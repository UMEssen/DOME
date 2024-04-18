UIMA_ANNOTATION = "uima.tcas.Annotation"
NAMED_ENTITY_CAS = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
# NAMED_ENTITY_CAS = "de.ume.ner.NamedEntity"
DOCUMENT_TYPE = "de.ume.info"
NAMED_ENTITY_TYPE = "de.ume.ner.type"
SENTENCE_CHUNK_TYPE = "de.ume.chunk.type.Sentence"
AGE_ENTITY = "de.ume.ner.type.Age"
CONTACT_ENTITY = "de.ume.ner.type.Contact"
DATE_ENTITY = "de.ume.ner.type.Date"
ID_ENTITY = "de.ume.ner.type.ID"
LOCATION_ENTITY = "de.ume.ner.type.Location"
NAME_ENTITY = "de.ume.ner.type.Name"
PROFESSION_ENTITY = "de.ume.ner.type.Profession"
OTHER_ENTITY = "de.ume.ner.type.Other"


MAIN_PHI_MAPPING = {
    "CONTACT_PHI_NAME": "Contact",
    "AGE_PHI_NAME": "Age",
    "DATE_PHI_NAME": "Date",
    "NAME_PHI_NAME": "Name",
    "ID_PHI_NAME": "ID",
    "PROFESSION_PHI_NAME": "Profession",
    "LOCATION_PHI_NAME": "Location",
    "OTHER_PHI_NAME": "Other",
}

PHI_MAPPING = {
    "AGE_PHI_NAME": "AGE",
    "EMAIL_PHI_NAME": "EMAIL",
    "FAX_PHI_NAME": "FAX",
    "PHONE_PHI_NAME": "PHONE",
    "URL_PHI_NAME": "URL",
    "BIRTHDATE_PHI_NAME": "BIRTHDATE",
    "DATE_PHI_NAME": "DATE",
    "ENUMBER_PHI_NAME": "PATIENTID",
    "STUDY_PHI_NAME": "STUDYID",
    "WEIGHT_PHI_NAME": "WEIGHT",
    "SIZE_PHI_NAME": "SIZE",
    "CITY_PHI_NAME": "CITY",
    "COUNTRY_PHI_NAME": "CITY",
    "HOSPITAL_PHI_NAME": "HOSPITAL",
    "ORGANIZATION_PHI_NAME": "ORGANIZATION",
    "LOC_OTHER_PHI_NAME": "OTHER",
    "STATE_PHI_NAME": "CITY",
    "STREET_PHI_NAME": "STREET",
    "ZIP_PHI_NAME": "ZIP",
    "NAME_OTHER_PHI_NAME": "OTHER",
    "PATIENT_PHI_NAME": "PATIENT",
    "STAFF_PHI_NAME": "STAFF",
    "DOCTOR_PHI_NAME": "STAFF",
    "PROFESSION_PHI_NAME": "PROFESSION",
    "STATUS_PHI_NAME": "STATUS",
    "OTHER_OTHER_PHI_NAME": "OTHER",
}

PHI_VALUES_SUBCLASS = [
    "Age_AGE",
    "Contact_EMAIL",
    "Contact_FAX",
    "Contact_PHONE",
    "Contact_URL",
    "Date_BIRTHDATE",
    "Date_DATE",
    "ID_PATIENTID",
    "ID_STUDYID",
    "Location_CITY",
    # "Location_COUNTRY",
    "Location_HOSPITAL",
    "Location_ORGANIZATION",
    "Location_OTHER",
    # "Location_STATE",
    "Location_STREET",
    "Location_ZIP",
    "Name_OTHER",
    "Name_PATIENT",
    "Name_STAFF",
    "Profession_PROFESSION",
    "Profession_STATUS",
]


PHI_VALUES_SUPERCLASS = [
    "Age",
    "Contact",
    "Date",
    "ID",
    "Location",
    "Name",
    "Profession",
]


NO_DEID_DATES = [
    r"\d{1,2}(\.|\/) ?\d{4}",  # e.g. 12.2020
    r"\d{1,2}\/\d{1,2}",  # e.g. 10/20 <- 10.2020
    # r"\d{3}(\.|/)\d{4}", # e.g. ???
    r"\d{2}(\-)\d{2}(\/)\d{4}",  # e.g. 02-03/2020
    r"\d{2}\.\d{4}-\d{2}\.\d{4}",  # e.g. 02.2020-03.2020
    r"\d{2}\/\d{2}(\-| \- )\d{2}\/\d{2}",  # e.g. 02/20-03/20
    r"\d{4}",  # e.g. 2021
    r"\d{2}",  # e.g. 97
    r"\d{4}\/\d{4}",  # e.g. 2021/2022
    r"[A-Za-zäüö]+([^\w]+\d{2,4})?",  # e.g. Oktober, Januar 2021
    r"[A-Za-zäüö]+\.",  # e.g. Okt. Nov.
    r"[0-9](-[0-9])? [A-Za-zä]+\.",  # e.g. in 2-3 Mo.
]

DAYS_WEEK = {
    "montag",
    "dienstag",
    "mittwoch",
    "donnerstag",
    "freitag",
    "samstag",
    "sonntag",
    "sommerhitze",
    "mo",
    "fr",
}
