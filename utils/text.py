import re
from typing import Dict, List
from core.constants import SYMPTOM_CATEGORY_MAP

_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06ff\s]")
_SPACES_RE = re.compile(r"\s+")
_ARABIC_TRANS = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ة':'ه','ى':'ي','ؤ':'و','ئ':'ي'})

def normalize_text(text: str) -> str:
    """Unified normalization: lowercase + no punctuation + Arabic unification."""
    if not text: return ""
    t = _DIACRITICS_RE.sub("", text)
    t = t.translate(_ARABIC_TRANS).lower()
    t = _NON_WORD_RE.sub(" ", t)
    return _SPACES_RE.sub(" ", t).strip()

NORMALIZED_SYMPTOM_MAP = {
    normalize_text(k): v for k, v in SYMPTOM_CATEGORY_MAP.items()
}

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        if not text: return "ar"
        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic / max(len(text.strip()), 1) > 0.25 else "en"
