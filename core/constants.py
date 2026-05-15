from typing import Dict, List

GEMINI_TEXT_MODELS   = ["gemini-1.5-flash", "gemini-1.5-pro"]
GEMINI_VISION_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"]

ALLOWED_IMAGE_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
})

MEDICAL_DISCLAIMER = "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط ولا تُغني عن استشارة طبيب متخصص."

MEDICAL_KEYWORDS: frozenset = frozenset({
    "ألم","وجع","مرض","دواء","طبيب","مستشفى","أعراض","علاج",
    "صداع","حمى","سعال","ضغط","سكر","قلب","كلى","معدة",
    "عظام","جلد","عين","أذن","أنف","رئة","كبد","دم",
    "تعب","إرهاق","دوار","غثيان","إسهال","إمساك","حرقة",
    "الم","عندي","عندى","اشعر","احس","اعاني","يؤلم",
    "بوجعني","بتوجعني","حاسس","حاسه","حبوب","طفح","حكة",
    "عملية","جراحة","منظار","تحليل","أشعة","نتيجة","تقرير",
    "برد","انفلونزا","رشح","زكام","كحة","بلغم","حرارة",
    "pain","ache","fever","cough","headache","nausea","dizzy",
    "vomit","diarrhea","symptom","disease","doctor","hospital",
    "medicine","drug","blood","heart","lung","kidney","liver",
    "diabetes","pressure","infection","allergy","rash","swelling",
    "fatigue","tired","breathe","chest","stomach","throat",
    "surgery","scan","test","result","report","prescription",
    "cold","flu","runny","nose","sneeze","congestion",
})

LOW_QUALITY_PATTERNS: frozenset = frozenset({
    "طبيعي","تم الاجابة","كل شيء ممكن","راجع الطبيب","استشر طبيب",
    "غير واضح","وضح اكثر","natural","answered","everything possible",
    "consult doctor","not clear","clarify more",
})

GARBAGE_PATTERNS: frozenset = frozenset({
    "تم الاجابة","راجع الطبيب","استشر طبيب","كل شيء ممكن","غير واضح",
    "وضح اكثر","طبيعي","لا يوجد","لا يوجد جواب","لا يوجد رد",
    "معلومات غير متوفرة","غير متوفر","لا اعرف","لا أستطيع","لا يمكنني",
    "معلومات محدودة","answered","consult doctor","everything possible",
    "not clear","clarify more","natural","no answer","no response",
    "information not available","not available","don't know","cannot","limited information",
})

EMERGENCY_KEYWORDS_AR: frozenset = frozenset({
    "ألم صدر","ضيق تنفس","نوبة قلبية","سكتة دماغية","نزيف شديد",
    "إغماء","فقدان وعي","صدمة","حروق شديدة","كسر عظم",
    "ألم حاد","طوارئ","إسعاف","علاج فوري","خطر على الحياة",
    "ضربة شمس","تسمم","جرح عميق","نزيف داخلي","انفجار",
    "ألم بطن حاد","صعوبة بلع","خدر","شلل","تشنج",
    "انتحار","أفكار انتحارية","إيذاء النفس",
})

EMERGENCY_KEYWORDS_EN: frozenset = frozenset({
    "chest pain","difficulty breathing","heart attack","stroke","severe bleeding",
    "fainting","loss of consciousness","shock","severe burns","broken bone",
    "severe pain","emergency","ambulance","immediate treatment","life threatening",
    "heat stroke","poisoning","deep wound","internal bleeding","explosion",
    "severe abdominal pain","difficulty swallowing","numbness","paralysis","seizure",
    "suicide","suicidal thoughts","self harm",
})

SYMPTOM_CATEGORY_MAP: Dict[str, List[str]] = {
    "حلق":["respiratory","general"],"سعال":["respiratory","general"],
    "كحة":["respiratory","general"],"رئة":["respiratory"],
    "ربو":["respiratory"],"أنف":["respiratory"],"زكام":["respiratory"],
    "رشح":["respiratory"],"ضيق تنفس":["respiratory","cardiology"],
    "throat":["respiratory","general"],"cough":["respiratory","general"],
    "breath":["respiratory","cardiology"],"asthma":["respiratory"],
    "حرارة":["general","pediatrics"],"حمى":["general","pediatrics"],
    "برد":["general","respiratory"],"انفلونزا":["general","respiratory"],
    "fever":["general","pediatrics"],"infection":["general"],
    "قلب":["cardiology"],"صدر":["cardiology","respiratory"],
    "ضغط":["cardiology"],"heart":["cardiology"],"chest":["cardiology","respiratory"],
    "صداع":["neurology","general"],"دوار":["neurology"],
    "أعصاب":["neurology"],"headache":["neurology","general"],"dizzy":["neurology"],
    "معدة":["gastroenterology"],"بطن":["gastroenterology"],
    "إسهال":["gastroenterology"],"غثيان":["gastroenterology"],
    "كبد":["gastroenterology"],"stomach":["gastroenterology"],
    "diarrhea":["gastroenterology"],"nausea":["gastroenterology"],
    "جلد":["dermatology"],"طفح":["dermatology"],"حكة":["dermatology"],
    "skin":["dermatology"],"rash":["dermatology"],"itching":["dermatology"],
    "عظام":["orthopedic"],"مفاصل":["orthopedic"],"ظهر":["orthopedic"],
    "bone":["orthopedic"],"joint":["orthopedic"],
    "كلى":["urology"],"بول":["urology"],"kidney":["urology"],"urine":["urology"],
    "عين":["ophthalmology"],"نظر":["ophthalmology"],"eye":["ophthalmology"],
    "طفل":["pediatrics"],"رضيع":["pediatrics"],"أطفال":["pediatrics"],
    "child":["pediatrics"],"infant":["pediatrics"],
    "نفس":["psychology"],"قلق":["psychology"],"اكتئاب":["psychology"],
    "anxiety":["psychology"],"depression":["psychology"],
    "سكري":["endocrinology"],"غدة":["endocrinology"],"هرمون":["endocrinology"],
    "diabetes":["endocrinology"],"thyroid":["endocrinology"],
    "حمل":["gynecology"],"دورة":["gynecology"],"رحم":["gynecology"],
    "pregnancy":["gynecology"],"period":["gynecology"],
    "أسنان":["dentistry"],"سن":["dentistry"],"لثة":["dentistry"],
    "tooth":["dentistry"],"gum":["dentistry"],
}

CAUSE_MAP = {
    "respiratory": "Respiratory infection or inflammation",
    "cardiology": "Cardiovascular condition or circulation issue",
    "neurology": "Neurological condition or nerve issue",
    "gastroenterology": "Gastrointestinal issue or digestive problem",
    "dermatology": "Skin condition or allergic reaction",
    "orthopedic": "Musculoskeletal injury or joint issue",
    "general": "General medical condition",
    "pediatrics": "Pediatric condition",
    "urology": "Urinary or kidney issue",
    "ophthalmology": "Eye condition or vision issue",
    "psychology": "Psychological or emotional factor",
    "endocrinology": "Hormonal or metabolic issue",
    "gynecology": "Gynecological or reproductive issue",
    "dentistry": "Dental or oral health issue",
}
