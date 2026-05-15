from typing import List, Optional, Tuple
from core.config import EXACT_MATCH_THRESHOLD, WEIGHT_EXACT, WEIGHT_CATEGORY
from utils.text import normalize_text

def token_similarity(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def compute_hybrid_score(
    cosine: float,
    query_norm: str,
    match_question: str,
    match_category: Optional[str],
    expected_categories: List[str],
) -> Tuple[float, float, float, str]:
    semantic_overlap = token_similarity(query_norm, normalize_text(match_question))
    
    match_type = "semantic"
    if semantic_overlap >= EXACT_MATCH_THRESHOLD:
        match_type = "exact" if semantic_overlap == 1.0 else "high_similarity"

    category_match = 0.0
    if match_category and expected_categories:
        cl = (match_category or "").lower()
        ec = [c.lower() for c in expected_categories]
        if cl in ec:
            category_match = 1.0 if cl == ec[0] else 0.6

    final = min(1.0, (0.6 * cosine) + (0.2 * semantic_overlap) + (0.2 * category_match))
    return round(final, 4), round(semantic_overlap, 4), round(category_match, 4), match_type
