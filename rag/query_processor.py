import re
from typing import List, Dict, Set, Optional
from core.logging import log
from utils.text import normalize_text, normalize_arabic

# Comprehensive Egyptian Dialect & Slang to Standard Medical Terms Mapping
EGYPTIAN_DIALECT_MAP: Dict[str, List[str]] = {
    "بطني": ["ألم بطن", "معدة", "جهاز هضمي", "مغص"],
    "بطنى": ["ألم بطن", "معدة", "جهاز هضمي", "مغص"],
    "نهجان": ["ضيق تنفس", "صعوبة تنفس", "نهجة", "كرشة نفس"],
    "كرشة نفس": ["ضيق تنفس", "صعوبة تنفس", "نهجان"],
    "كرشه نفس": ["ضيق تنفس", "صعوبة تنفس", "نهجان"],
    "هرش": ["حكة", "طفح جلدي", "حساسية", "تهيج جلد"],
    "ترجيع": ["قيء", "غثيان", "معدة", "اضطراب هضمي"],
    "سخونية": ["حمى", "ارتفاع حرارة", "سخونة"],
    "سخونيه": ["حمى", "ارتفاع حرارة", "سخونة"],
    "زور": ["التهاب حلق", "لوز", "حنجرة"],
    "زوري": ["التهاب حلق", "لوز", "حنجرة"],
    "زورى": ["التهاب حلق", "لوز", "حنجرة"],
    "وجع": ["ألم", "مغص"],
    "مغص": ["ألم بطن", "معدة", "قولون"],
    "دوخة": ["دوار", "دوخة", "عدم اتزان"],
    "دوخه": ["دوار", "دوخة", "عدم اتزان"],
    "سنان": ["أسنان", "لثة", "ضرس"],
    "سناني": ["أسنان", "لثة", "ضرس"],
    "سنانى": ["أسنان", "لثة", "ضرس"],
    "ودني": ["ألم أذن", "التهاب أذن"],
    "ودنى": ["ألم أذن", "التهاب أذن"],
    "عيني": ["ألم عين", "رمد", "حساسية عين"],
    "عينى": ["ألم عين", "رمد", "حساسية عين"],
    "تنميل": ["خدر", "أعصاب", "فقرات"],
    "رعشة": ["رعشة", "ارتجاف", "تشنج"],
    "رعشه": ["رعشة", "ارتجاف", "تشنج"],
    "كحة": ["سعال", "بلغم", "كحة"],
    "كحه": ["سعال", "بلغم", "كحة"],
    "بلغم": ["سعال", "بلغم", "صدر"],
    "تهيج": ["تهيج", "حساسية"],
    "حرقان": ["حموضة", "حرقة معدة", "حرقان بول"],
    "اسهال": ["نزلة معوية", "إسهال"],
    "امساك": ["عسر هضم", "قولون", "إمساك"],
    "صدر": ["ألم صدر", "ذبحة", "قلب", "تنفس"],
    "صدري": ["ألم صدر", "ذبحة", "قلب", "تنفس"],
    "صدرى": ["ألم صدر", "ذبحة", "قلب", "تنفس"],
}

class ArabicQueryProcessor:
    """
    Handles medical query normalization, Egyptian dialect mapping, and 
    advanced medical query expansion for high-performance Arabic RAG.
    """
    
    _DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")
    _NON_WORD_RE = re.compile(r"[^\w\u0600-\u06ff\s]")
    _SPACES_RE = re.compile(r"\s+")
    _ARABIC_TRANS = str.maketrans({
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ة': 'ه', 'ى': 'ي', 'ؤ': 'و',
        'ئ': 'ي', 'ـ': ''
    })

    _COMMON_MEDICAL_WORDS = {
        "حامل", "حمل", "صداع", "سخونه", "سخونة", "مغص", "ترجيع", "اسهال",
        "كحه", "كحة", "نهجان", "وجع", "الم", "ألم", "هرش", "حكه", "حكة", "تنفس",
        "صدر", "بطن", "قلب", "ضغط", "سكر", "دوار", "دوخه", "دوخة", "حراره", "حرارة"
    }

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return ArabicQueryProcessor.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    @classmethod
    def correct_spelling(cls, query: str) -> str:
        """
        Applies a fuzzy Levenshtein spell-correction to common medical words
        to handle typological keyboard slips (e.g., 'حامب' -> 'حامل').
        """
        words = query.split()
        corrected_words = []
        
        for word in words:
            # If the word is already a known medical slang or in dictionary, keep it
            if word in EGYPTIAN_DIALECT_MAP or word in cls._COMMON_MEDICAL_WORDS or len(word) <= 2:
                corrected_words.append(word)
                continue
            
            best_match = word
            best_dist = 999
            
            for target in cls._COMMON_MEDICAL_WORDS:
                # Only check words of similar length to prevent wild corrections
                if abs(len(word) - len(target)) <= 1:
                    dist = cls.levenshtein_distance(word, target)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = target
            
            # If we found a very close match (distance of exactly 1), correct it!
            if best_dist == 1:
                log.info(f"[Query Processor] ⚡ Fuzzy Spell Corrected: '{word}' -> '{best_match}'")
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)
                
        return " ".join(corrected_words)

    @classmethod
    def clean_and_normalize(cls, text: str) -> str:
        """
        Removes Tashkeel, normalizes Arabic characters (Alif, Ya, Ta Marbouta, etc.),
        and removes punctuation and double spaces.
        """
        if not text:
            return ""
        
        # Remove diacritics (Tashkeel)
        text = cls._DIACRITICS_RE.sub("", text)
        
        # Translate Arabic variations
        text = text.translate(cls._ARABIC_TRANS)
        
        # Clean punctuation and extra spaces
        text = cls._NON_WORD_RE.sub(" ", text)
        text = cls._SPACES_RE.sub(" ", text.lower()).strip()
        
        return text

    @classmethod
    def expand_dialect_egyptian(cls, query: str) -> Set[str]:
        """
        Scans normalized query for Egyptian colloquial medical slang terms
        and maps them to standard clinical Arabic keywords. Handles common prefix conjunctions.
        """
        expanded_keywords = set()
        words = query.split()
        
        # Strip common Arabic prefixed conjunctions (و, ف, ب) before matching
        for word in words:
            # Try direct match
            if word in EGYPTIAN_DIALECT_MAP:
                expanded_keywords.update(EGYPTIAN_DIALECT_MAP[word])
                continue
            
            # Check prefixes if word is long enough
            if len(word) > 3:
                # Strip 'و' (and)
                if word.startswith("و"):
                    stripped = word[1:]
                    if stripped in EGYPTIAN_DIALECT_MAP:
                        expanded_keywords.update(EGYPTIAN_DIALECT_MAP[stripped])
                        continue
                # Strip 'ف' (then/so)
                if word.startswith("ف"):
                    stripped = word[1:]
                    if stripped in EGYPTIAN_DIALECT_MAP:
                        expanded_keywords.update(EGYPTIAN_DIALECT_MAP[stripped])
                        continue
        
        # Multi-word matching (e.g. "كرشة نفس")
        for slang, medical_terms in EGYPTIAN_DIALECT_MAP.items():
            if " " in slang and slang in query:
                expanded_keywords.update(medical_terms)
                
        return expanded_keywords

    @classmethod
    async def get_semantic_expansion(cls, query: str, gemini_service) -> List[str]:
        """
        Optionally uses Gemini for semantic query expansion to retrieve highly accurate clinical terms.
        """
        prompt = (
            "أنت خبير أنظمة استرجاع المعلومات الطبية (Medical Information Retrieval).\n"
            "مهمتك هي تحليل السؤال الطبي التالي للمريض وصياغة قائمة من المرادفات والمصطلحات الطبية بالفصحى.\n"
            "أعطني فقط الكلمات المفتاحية والأعراض المرتبطة مباشرة مفصولة بمسافات دون أي ترقيم أو شرح أو علامات ترقيم.\n\n"
            f"سؤال المريض: {query}\n\n"
            "الكلمات الطبية المفتاحية:"
        )
        try:
            expanded_text, _ = await gemini_service.generate(prompt)
            # Normalize and clean output
            normalized_expansion = cls.clean_and_normalize(expanded_text)
            return [word for word in normalized_expansion.split() if len(word) > 2]
        except Exception as exc:
            log.warning(f"[Query Expansion] LLM-based query expansion failed: {exc}. Continuing with dictionary-only expansion.")
            return []

    @classmethod
    async def process_and_expand(
        cls, 
        query: str, 
        gemini_service = None, 
        enable_llm_expansion: bool = False
    ) -> str:
        """
        Main entry point for processing and expanding Arabic medical queries.
        Combines character normalization, Egyptian slang translation, and LLM expansion.
        """
        # 1. Base Normalization
        normalized_query = cls.clean_and_normalize(query)
        
        # 1.5 Fuzzy Spelling Correction for Arabic keyboard slips (e.g. 'حامب' -> 'حامل')
        normalized_query = cls.correct_spelling(normalized_query)
        
        # 2. Local Slang Mapping (Egyptian Dialect)
        dialect_synonyms = cls.expand_dialect_egyptian(normalized_query)
        
        # 3. LLM semantic expansion if enabled and service provided
        llm_synonyms = []
        if enable_llm_expansion and gemini_service:
            llm_synonyms = await cls.get_semantic_expansion(query, gemini_service)
            
        # Combine all tokens uniquely
        final_terms = list(normalized_query.split())
        for term in dialect_synonyms:
            term_norm = cls.clean_and_normalize(term)
            if term_norm and term_norm not in final_terms:
                final_terms.append(term_norm)
                
        for term in llm_synonyms:
            if term not in final_terms:
                final_terms.append(term)
                
        result = " ".join(final_terms).strip()
        log.info(f"[Query Processor] Original: '{query}' -> Normalized & Expanded: '{result}'")
        return result
