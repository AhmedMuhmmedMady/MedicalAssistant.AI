import asyncio
from typing import List, Dict, Any, Tuple
from core.logging import log
from core.config import TOP_K, PINECONE_NAMESPACE, MAX_RETRIES, RETRY_DELAY, MIN_CONFIDENCE
from models.schemas import KnowledgeMatch
from rag.retriever import get_index, encode_async
from rag.query_processor import ArabicQueryProcessor
from rag.router import MedicalNamespaceRouter
from rag.ranker import MedicalReranker
from utils.cache import AsyncCache

SAFE_FALLBACK_TEXT = "لا توجد معلومات طبية كافية داخل قاعدة البيانات الحالية للإجابة بدقة."

class ArabicMedicalRetrievalEngine:
    """
    Production-grade Retrieval Engine for Arabic Medical RAG.
    Orchestrates Query Normalization, Dialect Expansion, Namespace Routing,
    Parallel Pinecone Queries, Four-Factor Hybrid Re-ranking, Adaptive Thresholding,
    and Anti-Hallucination Safe fallback layers.
    """
    
    def __init__(self, gemini_service = None):
        self.gemini_service = gemini_service
        self.router = MedicalNamespaceRouter(gemini_service)
        self.reranker = MedicalReranker()
        self.cache = AsyncCache(maxsize=300, ttl_seconds=3600)

    def build_professional_context(self, matches: List[KnowledgeMatch]) -> str:
        """
        Builds and formats the RAG context block exactly in the requested layout:
        📍 [التخصص: ...]
        سؤال مشابه:
        ...
        الإجابة الطبية:
        ...
        """
        if not matches or any(SAFE_FALLBACK_TEXT in m.answer for m in matches):
            return SAFE_FALLBACK_TEXT
            
        blocks = []
        for m in matches:
            block = (
                f"📍 [التخصص: {m.category or 'عام'}]\n\n"
                f"سؤال مشابه:\n{m.question.strip()}\n\n"
                f"الإجابة الطبية:\n{m.answer.strip()}"
            )
            blocks.append(block)
            
        return "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(blocks)

    async def _query_single_namespace(
        self, 
        index: Any, 
        vector: List[float], 
        namespace: str, 
        fetch_k: int
    ) -> List[Any]:
        """
        Performs a query against a specific Pinecone namespace with error handling and retries.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                def _call():
                    return index.query(
                        vector=vector,
                        top_k=fetch_k,
                        include_metadata=True,
                        namespace=namespace
                    )
                results = await asyncio.wait_for(asyncio.to_thread(_call), timeout=8.0)
                return results.matches if results and results.matches else []
            except Exception as exc:
                log.warning(f"[Retrieval Engine] Namespace '{namespace}' query attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
        return []

    async def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K,
        enable_llm_expansion: bool = False
    ) -> List[KnowledgeMatch]:
        """
        High-performance retrieval pipeline with caching, multi-namespace routing,
        hybrid reranking, dynamic thresholding, and anti-hallucination fallbacks.
        """
        cache_key = f"rag_retrieve_{query}_{top_k}_{enable_llm_expansion}"
        
        async def _execute_retrieval() -> List[KnowledgeMatch]:
            log.info(f"[Retrieval Engine] Starting retrieval pipeline for: '{query[:60]}...'")
            
            # 1. Route Namespace
            routing_info = await self.router.route_query(query)
            primary_ns = routing_info["primary_namespace"]
            candidates = routing_info["all_candidate_namespaces"]
            conf = routing_info["confidence"]
            
            # 2. Query Normalization & Slang-to-Medical Expansion
            expanded_query = await ArabicQueryProcessor.process_and_expand(
                query=query, 
                gemini_service=self.gemini_service,
                enable_llm_expansion=enable_llm_expansion
            )
            
            # 3. Generate Embeddings (Reuse query embedding across namespaces)
            try:
                vector = await encode_async(expanded_query)
            except Exception as exc:
                log.error(f"[Retrieval Engine] Embedding generation failed: {exc}")
                return self._get_safe_fallback()
                
            index = await get_index()
            fetch_k = min(top_k * 3, 20)
            
            # 4. Fetch Chunks Concurrently across candidate namespaces
            tasks = [
                self._query_single_namespace(index, vector, ns, fetch_k)
                for ns in candidates
            ]
            
            results_batches = await asyncio.gather(*tasks)
            
            # Consolidate and deduplicate raw matches across namespaces by vector ID
            raw_matches_map = {}
            for batch in results_batches:
                for match in batch:
                    raw_matches_map[match.id] = match
                    
            raw_matches = list(raw_matches_map.values())
            log.info(f"[Retrieval Engine] Retrived {len(raw_matches)} raw matches from Pinecone.")
            
            if not raw_matches:
                log.warning("[Retrieval Engine] No matches returned from Pinecone. Triggering anti-hallucination fallback.")
                return self._get_safe_fallback()
                
            # 5. Hybrid Re-ranking
            reranked = self.reranker.rerank(raw_matches, query, routing_info)
            
            # 6. Dynamic/Adaptive Threshold Filtering
            adaptive_threshold = self.reranker.calculate_adaptive_threshold(raw_matches, conf)
            
            # Keep matches that survive our dynamic threshold
            filtered_matches = [
                match for match in reranked 
                if match.confidence >= adaptive_threshold
            ]
            
            log.info(
                f"[Retrieval Engine] Adaptive threshold: {adaptive_threshold:.3f} | "
                f"Matches surviving filter: {len(filtered_matches)} / {len(reranked)}"
            )
            
            # 7. Anti-Hallucination Safe Fallback if results are weak
            if not filtered_matches:
                log.warning(
                    f"[Retrieval Engine] All matches filtered out. Best reranked score was "
                    f"{reranked[0].confidence:.3f} (below threshold {adaptive_threshold:.3f}). "
                    f"Triggering safe fallback to protect medical accuracy."
                )
                return self._get_safe_fallback()
                
            # Return top_k best surviving matches
            return filtered_matches[:top_k]
            
        matches, is_hit = await self.cache.get_or_compute(cache_key, _execute_retrieval)
        if is_hit:
            log.info(f"[Retrieval Engine] Cache hit for query: '{query[:40]}'")
            
        return matches

    def _get_safe_fallback(self) -> List[KnowledgeMatch]:
        """
        Returns a single deterministic safe fallback KnowledgeMatch to prevent model hallucination.
        """
        return [
            KnowledgeMatch(
                question="لا توجد معلومات",
                answer=SAFE_FALLBACK_TEXT,
                confidence=0.0,
                category="General"
            )
        ]
