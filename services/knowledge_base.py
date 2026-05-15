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
        sorted_m, seen_cats, selected = sorted(matches, key=lambda m: m.confidence, reverse=True), set(), []
        for m in sorted_m:
            if len(selected) >= max_count: break
            if m.category not in seen_cats or len(selected) < 2:
                selected.append(m)
                if m.category: seen_cats.add(m.category)
        return selected

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

    async def search(self, query: str, top_k: int = TOP_K) -> List[KnowledgeMatch]:
        log.info(f"[KB-v2] Searching: '{query[:80]}'")
        expected_cats = _extract_expected_categories(query)
        log.info(f"[KB-v2] Expected categories: {expected_cats or ['unknown']}")

        try:
            vector = await encode_async(query)
        except Exception as exc:
            log.error(f"[KB-v2] Encoding failed: {exc}")
            return []

        index   = await get_index()
        fetch_k = min(top_k * 3, 30)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                def _call():
                    return index.query(
                        vector=vector, top_k=fetch_k,
                        include_metadata=True, namespace=PINECONE_NAMESPACE or ""
                    )
                results = await asyncio.wait_for(asyncio.to_thread(_call), timeout=10.0)

                raw_scores = [round(float(m.score), 4) for m in results.matches if m.score]
                log.info(f"[KB-v2] Pinecone raw scores ({len(raw_scores)}): {raw_scores}")

                return self._parse_matches_hybrid(results, query, expected_cats, top_k)

            except Exception as exc:
                log.warning(f"[KB-v2] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)

        log.error("[KB-v2] All retries exhausted")
        return []

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
                continue

            meta     = m.metadata or {}
            q_text   = meta.get("question", "")
            a_text   = meta.get("answer",   "")
            category = meta.get("category", "General")

            final, exact_b, cat_b, match_type = compute_hybrid_score(
                cosine=cosine, query_norm=query_norm, match_question=q_text,
                match_category=category, expected_categories=expected_cats,
            )

            icon = "🎯" if match_type == "exact" else "🔍" if match_type == "high_similarity" else "🌐"
            log.info(
                f"[KB-v2] {icon} {match_type:<16} cosine={cosine:.3f} "
                f"exact_b={exact_b:.2f} cat_b={cat_b:.2f} → final={final:.4f} "
                f"| cat={category} | q='{q_text[:50]}'"
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
