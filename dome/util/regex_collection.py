from dome.util.constants import MAIN_PHI_MAPPING, PHI_MAPPING

IMAGE_REGEX = r"ima.*?([0-9]+)\/([0-9]+(\-[0-9]+)?)"

STUDY_REGEX = r"(\b[0-9\-A-Za-züöäß]+?\b)\s*-?\s*Studie[^\w]"
SKIP_STUDIES = {
    "PET",
    "MRT",
    "CT",
    "PET/MRT",
    "PET/CT",
    "PET-MRT",
    "PET-CT",
    "Platform",
    "I",
    "1",
    "Phase-I",
    "Phase-1",
    "II",
    "2",
    "Phase-II",
    "Phase-2",
    "III",
    "3",
    "Phase-III",
    "Phase-3",
    "der",
    "die",
    "das",
    "des",
    "einer",
    "ein",
    "eines",
    "eine",
    "adjuvante",
    "andere",
    "beiden",
    "bzgl",
    "durch",
    "multizentrische",
    "monozentrische",
    "multicentrische",
    "monocentrische",
    "für",
    "in",
    "keine",
    "klinischen",
    "Klinische",
    "lehnt",
    "nicht-interventionelle",
    "randomisierte",
    "Einschluss",
    "Oder",
}

WEIGHT_REGEX = (
    r"(?<=[^\w])"
    r"(Körper)?[gG]ewicht.\s*(([0-9]{1,3}|[0-9]{1,2}[\.,][0-9]{1,2})\s*(kg|KG|Kg|kG)?)"
    r"(?=[^\w])"
)

SIZE_REGEX = (
    r"(?<=[^\w])"
    r"(Körper)?[gG]röße.\s*(([0-9]{3}|[1-2]{1}[\.,][0-9]{1,2})\s*(cm|m|CM|M|Cm|cM)?)"
    r"(?=[^\w])"
)

ADDITIONS_FOR_ALL = [
    (
        STUDY_REGEX,
        MAIN_PHI_MAPPING["ID_PHI_NAME"],
        PHI_MAPPING["STUDY_PHI_NAME"],
        1,
    ),
]

MELANOM_REDACT_REGEX = ADDITIONS_FOR_ALL

RADIO_REDACT_REGEX = [
    (
        r"([0-9]{2}\.[0-9]{2}\.([0-9]{2,4})?)",
        MAIN_PHI_MAPPING["DATE_PHI_NAME"],
        PHI_MAPPING["DATE_PHI_NAME"],
        0,
    ),
    (
        r"(2[0-9]{3})",
        MAIN_PHI_MAPPING["DATE_PHI_NAME"],
        PHI_MAPPING["DATE_PHI_NAME"],
        0,
    ),
    (
        r"ED ([0-9]{2}\/[0-9]{2,4})",
        MAIN_PHI_MAPPING["DATE_PHI_NAME"],
        PHI_MAPPING["DATE_PHI_NAME"],
        1,
    ),
    (
        r"([0-9]{2}\/[0-9]{4})",
        MAIN_PHI_MAPPING["DATE_PHI_NAME"],
        PHI_MAPPING["DATE_PHI_NAME"],
        0,
    ),
    (
        r"("
        r"("
        r"Jan(uar)?|"
        r"Feb(ruar)?|"
        r"Mär(z)?|"
        r"Apr(il)?|"
        r"Mai|"
        r"Jun(i)?|"
        r"Jul(i)?|"
        r"Aug(ust)?|"
        r"Sep(tember)?|"
        r"Oct(ober)?|"
        r"Nov(ember)?|"
        r"Dez(ember)?"
        r")[^\w][0-9]{4}"
        r")",
        MAIN_PHI_MAPPING["DATE_PHI_NAME"],
        PHI_MAPPING["DATE_PHI_NAME"],
        0,
    ),
    (
        r"[^\w](8[0-9]{4})[^\w]",
        MAIN_PHI_MAPPING["CONTACT_PHI_NAME"],
        PHI_MAPPING["PHONE_PHI_NAME"],
        1,
    ),
    (
        r"\(([0-9]{4})\)",
        MAIN_PHI_MAPPING["CONTACT_PHI_NAME"],
        PHI_MAPPING["PHONE_PHI_NAME"],
        1,
    ),
    (
        r"[^\w]([0-9]{3}[^\w][0-9]{4})[^\w]",
        MAIN_PHI_MAPPING["CONTACT_PHI_NAME"],
        PHI_MAPPING["PHONE_PHI_NAME"],
        1,
    ),
    (
        r"([A-Z][0-9]{2,7}\/[0-9]{2})",
        MAIN_PHI_MAPPING["ID_PHI_NAME"],
        PHI_MAPPING["ENUMBER_PHI_NAME"],
        0,
    ),
    (
        r"("
        r"Befunderstellung.+von|"
        r"Befundbesprechung.+vorab.+mit|"
        r"Telefonische.+Befundübermittlung.+vorab.+an"
        r")"
        r"[^\w](.+)\n",
        MAIN_PHI_MAPPING["NAME_PHI_NAME"],
        PHI_MAPPING["STAFF_PHI_NAME"],
        2,
    ),
] + ADDITIONS_FOR_ALL

SPLIT_DOC_REGEX = r"([A-ZÖÄÜ]\.\s[A-ZÖÄÜ][a-zöäüß]+)"


POSTPROCESS_REGEX = [
    # (18 Jahre, 15 Jahre)
    (
        r"([0-9]+)\sJahre",
        MAIN_PHI_MAPPING["AGE_PHI_NAME"],
        PHI_MAPPING["AGE_PHI_NAME"],
        1,
    ),
    # 14. Lebensjahr
    # 14. LJ
    (
        r"([0-9]+)\.\s(Lebensjahr|LJ)",
        MAIN_PHI_MAPPING["AGE_PHI_NAME"],
        PHI_MAPPING["AGE_PHI_NAME"],
        1,
    ),
] + ADDITIONS_FOR_ALL
