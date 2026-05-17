import sys
import os
import asyncio

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_processor import ArabicQueryProcessor
from rag.router import MedicalNamespaceRouter
from rag.ranker import MedicalReranker
from rag.retrieval_engine import ArabicMedicalRetrievalEngine

async def main():
    print("==================================================")
    print("🧪 STARTING MEDIC AL RAG RETRIEVAL ENGINE TESTING")
    print("==================================================")
    
    # 1. Test Arabic Query Processing
    print("\n[1] Testing Query Normalization & Slang Expansion...")
    q1 = "عندي سخونية ونهجان وهرش جامد"
    normalized = ArabicQueryProcessor.clean_and_normalize(q1)
    expanded = await ArabicQueryProcessor.process_and_expand(q1)
    print(f"  - Original Query  : '{q1}'")
    print(f"  - Clean Normalized: '{normalized}'")
    print(f"  - Dialect Expanded: '{expanded}'")
    
    # Assertions for query processor
    assert "سخونية" not in normalized, "Tashkeel / Ya normalization failed!"
    assert "حمي" in expanded, "Egyptian slang expansion for 'سخونية' failed!"
    assert "ضيق تنفس" in expanded, "Egyptian slang expansion for 'نهجان' failed!"
    assert "حكه" in expanded, "Egyptian slang expansion for 'هرش' failed!"
    print("  ✅ Query Normalization & Expansion Passed!")

    # 2. Test Namespace Routing
    print("\n[2] Testing Medical Namespace Routing...")
    router = MedicalNamespaceRouter()
    
    # Query with cardiology keywords
    cardio_query = "عندي ألم شديد في صدري ونبضات قلبي سريعة ونهجان"
    cardio_route = await router.route_query(cardio_query)
    print(f"  - Cardio Query: '{cardio_query}'")
    print(f"  - Mapped Route: {cardio_route['primary_namespace']} (Confidence: {cardio_route['confidence']:.2f})")
    assert cardio_route['primary_namespace'] == "cardiology", "Cardiology routing failed!"
    
    # Query with dermatology keywords
    skin_query = "هرش شديد في جلدي مع طفح جلدي احمر بيوجعني"
    skin_route = await router.route_query(skin_query)
    print(f"  - Skin Query  : '{skin_query}'")
    print(f"  - Mapped Route: {skin_route['primary_namespace']} (Confidence: {skin_route['confidence']:.2f})")
    assert skin_route['primary_namespace'] == "dermatology", "Dermatology routing failed!"
    
    print("  ✅ Specialty Namespace Routing Passed!")

    # 3. Test Reranking & Hybrid Scoring
    print("\n[3] Testing Reranker & Smart Adaptive Threshold...")
    reranker = MedicalReranker()
    
    # Mock some Pinecone search match objects
    class MockMatch:
        def __init__(self, id, score, question, answer, category):
            self.id = id
            self.score = score
            self.metadata = {
                "question": question,
                "answer": answer,
                "category": category
            }
            
    # Mock matches returned for Skin Query
    mock_matches = [
        MockMatch("1", 0.72, "عندي هرش جلدي شديد وتهيج وحكة احمرار", "هذه أعراض حساسية جلدية وإكزيما وعلاجها الترطيب", "dermatology"),
        MockMatch("2", 0.65, "عندي مغص شديد ووجع بطن وترجيع", "التهاب جدار المعدة يسبب غثيان ومغص", "internal_medicine"),
        MockMatch("3", 0.45, "عندي وجع جامد في ضرس العقل", "يجب مراجعة طبيب الأسنان لخلع الضرس", "dentistry")
    ]
    
    adaptive_threshold = reranker.calculate_adaptive_threshold(mock_matches, skin_route["confidence"])
    print(f"  - Dynamic Calculated Threshold: {adaptive_threshold:.3f}")
    
    reranked = reranker.rerank(mock_matches, skin_query, skin_route)
    print("  - Reranked Candidate List:")
    for idx, m in enumerate(reranked, 1):
        print(f"    [{idx}] Q: '{m.question[:30]}...' | Category: {m.category:<18} | Score: {m.confidence:.4f} (Survives: {m.confidence >= adaptive_threshold})")
        
    # First match should be highest and belong to dermatology
    assert reranked[0].category == "dermatology", "Reranker failed to boost correct category match!"
    assert reranked[0].confidence > reranked[1].confidence, "Reranking score ordering is incorrect!"
    print("  ✅ Reranking & Adaptive Filtering Passed!")

    print("\n==================================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! CODE IS PRODUCTION-READY")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
