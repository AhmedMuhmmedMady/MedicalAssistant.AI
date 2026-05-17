import asyncio
import re
from typing import Any, List
from collections import Counter

from core.config import MIN_CONFIDENCE, PINECONE_NAMESPACE, MAX_RETRIES, RETRY_DELAY, TOP_K, MAX_CONTEXT_MATCHES
from core.constants import MEDICAL_KEYWORDS
from core.logging import log
from models.schemas import KnowledgeMatch
from rag.retriever import get_index, encode_async
from rag.scoring import compute_hybrid_score, token_similarity
from utils.text import normalize_text, NORMALIZED_SYMPTOM_MAP

def _extract_expected_categories(query: str) -> List[str]:
    q = normalize_text(query)
    cnt: dict = {}
    for kw, cats in NORMALIZED_SYMPTOM_MAP.items():
        if kw in q:
            for c in cats:
                cnt[c] = cnt.get(c, 0) + 1
    return sorted(cnt, key=lambda c: cnt[c], reverse=True)

class KnowledgeBaseService:
    @staticmethod
    def _sanitize_match(match: KnowledgeMatch) -> KnowledgeMatch:
        q = re.sub(r'\s+', ' ', match.question.strip())
        a = match.answer.strip()
        return KnowledgeMatch(question=q, answer=a, confidence=match.confidence, category=match.category)

    @staticmethod
    def _deduplicate_and_sanitize(matches: List[KnowledgeMatch]) -> List[KnowledgeMatch]:
        seen, out = set(), []
        for m in matches:
            if m.is_garbage: continue
            key = m.answer.lower().strip()
            if key in seen: continue
            seen.add(key)
            out.append(KnowledgeBaseService._sanitize_match(m))
        return out

    @staticmethod
    def _select_top_matches(matches: List[KnowledgeMatch], max_count: int = MAX_CONTEXT_MATCHES) -> List[KnowledgeMatch]:
        if not matches: return []
        sorted_m = sorted(matches, key=lambda m: m.confidence, reverse=True)
        return sorted_m[:max_count]

    @staticmethod
    def _calculate_garbage_ratio(matches: List[KnowledgeMatch]) -> float:
        if not matches: return 0.0
        return round(sum(1 for m in matches if m.is_garbage) / len(matches), 2)

    @staticmethod
    def _category_consistency(matches: List[KnowledgeMatch]) -> float:
        if not matches: return 0.0
        cats = [m.category for m in matches if m.category]
        if not cats: return 0.0
        return round(Counter(cats).most_common(1)[0][1] / len(cats), 2)

    @staticmethod
    def _extract_medical_tokens(text: str) -> set:
        clean = normalize_text(text)
        tokens = set(t for t in re.split(r"[\s\W]+", clean) if len(t) >= 3)
        out = set()
        for tok in tokens:
            if any(kw in tok or tok in kw for kw in MEDICAL_KEYWORDS):
                out.add(tok)
            elif len(tok) >= 4:
                out.add(tok)
        return out

    def __init__(self):
        from utils.cache import AsyncCache
        self._async_cache = AsyncCache(200)
        self.engine = None

    async def search(self, query: str, top_k: int = TOP_K) -> List[KnowledgeMatch]:
        if self.engine is None:
            from services.gemini_service import GeminiService
            from rag.retrieval_engine import ArabicMedicalRetrievalEngine
            self.engine = ArabicMedicalRetrievalEngine(gemini_service=GeminiService())
            
        return await self.engine.retrieve(query, top_k=top_k)


    @staticmethod
    def _parse_matches_hybrid(
        results: Any, query: str, expected_cats: List[str], top_k: int,
    ) -> List[KnowledgeMatch]:
        query_norm = normalize_text(query)
        candidates, filtered = [], 0

        for m in results.matches:
            cosine = float(m.score) if m.score is not None else 0.0
            if cosine < MIN_CONFIDENCE * 0.80:
                filtered += 1
                log.info(f"[KB-v2] REJECTED | raw={cosine:.4f} < pre-filter threshold | q='{m.metadata.get('question', '')[:30]}'")
                continue

            meta     = m.metadata or {}
            q_text   = meta.get("question", "")
            a_text   = meta.get("answer",   "")
            category = meta.get("category", "General")

            from rag.scoring import compute_hybrid_score
            final, semantic, cat_b, match_type = compute_hybrid_score(
                cosine=cosine, query_norm=query_norm, match_question=q_text,
                match_category=category, expected_categories=expected_cats,
            )

            if final < MIN_CONFIDENCE:
                log.info(f"[KB-v2] REJECTED | raw={cosine:.4f} reranked={final:.4f} < threshold | q='{q_text[:30]}'")
            else:
                icon = "🎯" if match_type == "exact" else "🔍" if match_type == "high_similarity" else "🌐"
                log.info(
                    f"[KB-v2] ACCEPTED | {icon} {match_type:<16} raw={cosine:.4f} reranked={final:.4f} "
                    f"(sem={semantic:.2f} cat={cat_b:.2f}) "
                    f"| cat={category} | q='{q_text[:40]}'"
                )

            candidates.append(KnowledgeMatch(
                question=q_text, answer=a_text, confidence=final, category=category,
            ))

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        kept = [c for c in candidates if c.confidence >= MIN_CONFIDENCE]
        log.info(
            f"[KB-v2] Re-ranked: {len(kept)} kept "
            f"(top={kept[0].confidence if kept else 0.0:.4f}), "
            f"{len(candidates)-len(kept)} below threshold, {filtered} pre-filtered"
        )
        return kept[:top_k]

    @staticmethod
    def relevance_ok(query: str, matches: List[KnowledgeMatch], expected_cats: List[str] = None, min_overlap: int = 1) -> bool:
        from core.config import EXACT_MATCH_THRESHOLD
        
        if not matches: return False
        
        if matches[0].confidence >= 0.55:
            log.info(f"[Relevance] ✅ High semantic confidence bypass (score={matches[0].confidence:.3f})")
            return True
            
        qn = normalize_text(query)
        for m in matches[:3]:
            sim = token_similarity(qn, normalize_text(m.question))
            if sim >= EXACT_MATCH_THRESHOLD:
                log.info(f"[Relevance] ✅ exact match (sim={sim:.3f})")
                return True
                
        if expected_cats is None:
            expected_cats = _extract_expected_categories(query)
            
        if expected_cats and matches:
            top_cat = (matches[0].category or "").lower()
            if top_cat in [c.lower() for c in expected_cats]:
                log.info(f"[Relevance] ✅ category match ({top_cat})")
                return True
                
        qt = KnowledgeBaseService._extract_medical_tokens(query)
        if not qt:
            log.info("[Relevance] no medical tokens — allowing")
            return True
            
        combined = " ".join(f"{m.question} {m.answer}" for m in matches[:3])
        mt = KnowledgeBaseService._extract_medical_tokens(combined)
        overlap = len(qt & mt)
        ok = overlap >= min_overlap
        log.info(f"[Relevance] tokens: q={len(qt)} m={len(mt)} overlap={overlap} → {'✅' if ok else '❌'}")
        return ok
