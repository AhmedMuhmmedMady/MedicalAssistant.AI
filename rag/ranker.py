import re
from typing import List, Dict, Tuple, Optional, Set
from core.logging import log
from core.config import MIN_CONFIDENCE
from core.constants import MEDICAL_KEYWORDS
from models.schemas import KnowledgeMatch
from rag.query_processor import ArabicQueryProcessor

class MedicalReranker:
    """
    Production-grade hybrid re-ranker that refines vector similarity scores using
    category alignment, lexical token overlap, and medical keyword density.
    Also calculates adaptive thresholds dynamically to avoid soft hallucinations.
    """
    
    def __init__(
        self, 
        weight_cosine: float = 0.55,
        weight_category: float = 0.15,
        weight_lexical: float = 0.15,
        weight_medical: float = 0.15
    ):
        self.w_cosine = weight_cosine
        self.w_category = weight_category
        self.w_lexical = weight_lexical
        self.w_medical = weight_medical
        
    @staticmethod
    def _extract_medical_tokens(text_norm: str) -> Set[str]:
        """
        Extracts medical tokens from normalized text matching MEDICAL_KEYWORDS.
        """
        words = text_norm.split()
        med_tokens = set()
        for w in words:
            if len(w) >= 3:
                # Direct check or substring check against medical keywords
                if w in MEDICAL_KEYWORDS or any(kw in w for kw in MEDICAL_KEYWORDS):
                    med_tokens.add(w)
        return med_tokens

    @staticmethod
    def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Computes the standard Jaccard Similarity between two sets of tokens.
        """
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _containment_similarity(set_query: Set[str], set_doc: Set[str]) -> float:
        """
        Computes how much of the query tokens are contained in the document.
        """
        if not set_query:
            return 0.0
        return len(set_query & set_doc) / len(set_query)

    def calculate_adaptive_threshold(
        self, 
        raw_matches: List[any], 
        router_confidence: float
    ) -> float:
        """
        Calculates a dynamic similarity threshold. 
        Enforces a stricter threshold if top matches are extremely strong,
        and triggers safety overrides if the raw scores are weak.
        """
        base_threshold = float(MIN_CONFIDENCE) # Default 0.60
        
        if not raw_matches:
            return base_threshold
            
        top_raw_score = float(raw_matches[0].score)
        
        # 1. Excellent Quality Batch:
        # If vector database returns perfect matches (score >= 0.85), tighten the threshold
        # to filter out any noisy supporting matches.
        if top_raw_score >= 0.85:
            adaptive = max(base_threshold + 0.05, top_raw_score * 0.78)
            log.info(f"[Adaptive Threshold] 🎯 Excellent Match detected (top_score={top_raw_score:.2f}). Stricter threshold: {adaptive:.2f}")
            return round(adaptive, 3)
            
        # 2. Moderate/Low Quality Batch with strong specialty confidence:
        # If the category router is extremely confident but similarity is moderate (e.g. 0.65-0.75),
        # we can lower the threshold slightly (to e.g. 0.58) because the context is highly clinically relevant.
        if top_raw_score >= 0.65 and router_confidence >= 0.85:
            adaptive = base_threshold - 0.02
            log.info(f"[Adaptive Threshold] 🏥 Strong Category alignment. Relaxed threshold: {adaptive:.2f}")
            return round(adaptive, 3)
            
        # 3. Weak Quality Batch:
        # If vector scores are low and category router is not confident, raise the threshold
        # to prevent feeding garbage context to the LLM.
        if top_raw_score < 0.65 and router_confidence < 0.60:
            adaptive = base_threshold + 0.05
            log.info(f"[Adaptive Threshold] ⚠️ Weak signals detected. Increased safety threshold: {adaptive:.2f}")
            return round(adaptive, 3)
            
        return base_threshold

    def score_match(
        self,
        cosine_score: float,
        query_norm: str,
        match_question: str,
        match_answer: str,
        predicted_category: str,
        match_category: str,
        category_confidence: float,
        candidate_categories: List[str]
    ) -> Tuple[float, float, float, float]:
        """
        Computes the final four-factor hybrid reranked score.
        Returns (final_score, category_score, lexical_score, medical_score).
        """
        # --- 1. Category Alignment Score ---
        # Direct Match
        match_category_clean = (match_category or "").lower().strip()
        predicted_category_clean = (predicted_category or "").lower().strip()
        
        cat_alignment = 0.0
        if match_category_clean == predicted_category_clean:
            cat_alignment = 1.0
        elif match_category_clean in [c.lower() for c in candidate_categories]:
            cat_alignment = 0.7
        elif match_category_clean in ["general", "internal_medicine"]:
            cat_alignment = 0.4
            
        # Scale category alignment by router confidence
        category_score = cat_alignment * category_confidence

        # --- 2. Lexical Overlap Score ---
        q_tokens = set(query_norm.split())
        q_question_tokens = set(ArabicQueryProcessor.clean_and_normalize(match_question).split())
        q_answer_tokens = set(ArabicQueryProcessor.clean_and_normalize(match_answer).split())
        
        # Jaccard overlap on question (primary)
        jaccard_q = self._jaccard_similarity(q_tokens, q_question_tokens)
        # Containment overlap on answer (secondary)
        contain_a = self._containment_similarity(q_tokens, q_answer_tokens)
        
        lexical_score = (0.7 * jaccard_q) + (0.3 * contain_a)

        # --- 3. Medical Keyword Density Score ---
        q_med_tokens = self._extract_medical_tokens(query_norm)
        match_combined_norm = f"{ArabicQueryProcessor.clean_and_normalize(match_question)} {ArabicQueryProcessor.clean_and_normalize(match_answer)}"
        doc_med_tokens = self._extract_medical_tokens(match_combined_norm)
        
        medical_score = self._containment_similarity(q_med_tokens, doc_med_tokens) if q_med_tokens else 1.0

        # --- 4. Final Hybrid Reranked Score ---
        final_score = (
            (self.w_cosine * cosine_score) +
            (self.w_category * category_score) +
            (self.w_lexical * lexical_score) +
            (self.w_medical * medical_score)
        )
        
        # Ensure it fits between 0.0 and 1.0
        final_score = min(1.0, max(0.0, final_score))
        
        return (
            round(final_score, 4),
            round(category_score, 4),
            round(lexical_score, 4),
            round(medical_score, 4)
        )

    def rerank(
        self,
        raw_matches: List[any],
        query: str,
        routing_info: Dict[str, any]
    ) -> List[KnowledgeMatch]:
        """
        Reranks raw Pinecone matches based on the four-factor hybrid formula.
        """
        if not raw_matches:
            return []
            
        query_norm = ArabicQueryProcessor.clean_and_normalize(query)
        primary_ns = routing_info["primary_namespace"]
        conf = routing_info["confidence"]
        candidates = routing_info["all_candidate_namespaces"]
        
        reranked_matches = []
        for match in raw_matches:
            cosine = float(match.score) if match.score is not None else 0.0
            meta = match.metadata or {}
            
            q_text = meta.get("question", "")
            a_text = meta.get("answer", "")
            category = meta.get("category", "General")
            
            final, cat_s, lex_s, med_s = self.score_match(
                cosine_score=cosine,
                query_norm=query_norm,
                match_question=q_text,
                match_answer=a_text,
                predicted_category=primary_ns,
                match_category=category,
                category_confidence=conf,
                candidate_categories=candidates
            )
            
            reranked_matches.append(KnowledgeMatch(
                question=q_text,
                answer=a_text,
                confidence=final,
                category=category
            ))
            
            log.info(
                f"[Reranker] Chunk Q: '{q_text[:30]}' | "
                f"Scores: raw_cos={cosine:.3f} -> final_hybrid={final:.3f} "
                f"(cat={cat_s:.2f}, lex={lex_s:.2f}, med={med_s:.2f})"
            )
            
        # Sort by final hybrid score in descending order
        reranked_matches.sort(key=lambda x: x.confidence, reverse=True)
        return reranked_matches
