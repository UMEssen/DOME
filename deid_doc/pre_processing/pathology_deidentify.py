import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cassis
from cassis import Cas, TypeSystem
from cassis.typesystem import TYPE_NAME_STRING
from fhir_pyrate import Ahoy, Pirate

from deid_doc.util.cas_handling import (
    add_annotation_features,
    add_label_without_clashes,
)
from deid_doc.util.constants import (
    DOCUMENT_TYPE,
    MAIN_PHI_MAPPING,
    NAMED_ENTITY_CAS,
    NAMED_ENTITY_TYPE,
    PHI_MAPPING,
    UIMA_ANNOTATION,
)
from deid_doc.util.no_push_constants import MULTIPLE_DOCTORS, NAME_REGEX, PATIENT_REGEX
from deid_doc.util.regex_collection import PATHO_REGEX
from deid_doc.util.util import decode_base64, get_study_name

SEARCH_URL = "https://shipdev.uk-essen.de/app/FHIR/r4"
BASIC_AUTH = "https://shipdev.uk-essen.de/app/Auth/v1/basicAuth"
REFRESH_AUTH = "https://shipdev.uk-essen.de/app/Auth/v1/refresh"
separator = "--"


REMOVE_REDACT_RIGHT = [
    " familiär ",
    " übermittelt ",
    " im Rahmen ",
    " besteht ",
    " wurde ",
    " mit ",
    " diagnostiziert ",
    " telefonisch ",
    " zurück ",
    " erfolgt ",
    " war ",
    " zur ",
    " aus ",
    " um ",
    " klinisch ",
    " nochmals ",
    " mitgeteilt ",
    " und ",
    " wird ",
    " nach ",
    " Frau",
]
REMOVE_REDACT_END_RIGHT = [
    " telefonisch",
    " übermittelt",
    " umfangreich",
    " mitgeteilt",
    " am",
    " um",
    " mit",
    " vom",
    " und",
    # "PD ",
    " Brabeckstraße",
]
REMOVE_REDACT_START_LEFT = [
    "unter ",
    # "Doktor ",
    # "Dres. med. Prof. ",
    # "Prof",
    # "h.c.",
    # "Dres. med. ",
    # "med ",
    # "PD ",
]
PATIENT_OUTLIERS = [" Obduktion", " Fr", " Ob"]
SKIP_FIELDS = [
    # "Dr. Dr. med.",
    # "univ",
    # "Prof. Dr. med",
    # "Professor",
    # "Univ",
    # "Dr",
    "sieht man Blutbestandteile",
    "Herrn sieht man Blutbestandteile",
    "Herr sieht man Blutbestandteile",
    "",
]


def return_matched_regex(message: str, regex: str) -> Optional[str]:
    new_patient_regex = None
    for match in re.finditer(string=message, pattern=regex) or []:
        start = match.start()
        end = match.end()
        for out in PATIENT_OUTLIERS:
            if out in message[start:end]:
                end -= len(out)
            # print(message[start:end])
        new_patient_regex = (
            message[start:end].replace("-", r"\n?-\n?").replace(" ", r"[\n\r\s]+")
        )
    return new_patient_regex


def fix_indices(
    message: str,
    start: int,
    end: int,
    label: str,
    group: Optional[int],
    match: re.Match,
    preserve_number_letter: bool,
) -> Tuple[int, int]:
    if group is not None:
        # print(
        #     specific_class,
        #     message[start:end],
        #     len(match.groups()),
        #     [g for g in match.groups()],
        # )
        start, end = match.span(group)
        # print(message[start:end])
    for remove_redacted in REMOVE_REDACT_RIGHT:
        if remove_redacted in message[start:end]:
            # print(current_match)
            new_end_index = message[start:end].find(remove_redacted)
            end -= len(message[start:end]) - new_end_index
            # print(current_match)
    for remove_redacted in REMOVE_REDACT_END_RIGHT:
        if remove_redacted == message[end - len(remove_redacted) : end]:
            # print(message[start:end])
            end -= len(remove_redacted)
            # print(message[start:end])
    for remove_redacted in REMOVE_REDACT_START_LEFT:
        if remove_redacted == message[start : start + len(remove_redacted)]:
            # print(message[start:end])
            start += len(remove_redacted)
            # print(message[start:end])
    while message[start] in {" ", "\n", "."}:
        start += 1
    while message[end - 1] in {" ", "\n", "."}:
        end -= 1
    if message[start : start + 2] == ". ":
        start += 2
    if (
        preserve_number_letter
        and label == PHI_MAPPING["ENUMBER_PHI_NAME"]
        and message[start].isalpha()
    ):
        start += 1
    # Known errors
    if message[start:end] == "mstEine":
        end -= len("Eine")
    elif message[start:end].endswith("S. Teuber-"):
        end += len("Hanselmann")
    elif message[start:end].endswith("Topalidis Hannover"):
        end -= len(" Hannover")
    elif message[start:end].endswith("Gassel EKO"):
        end -= len(" EKO")
    return start, end


def anonymize_message(
    typesystem: TypeSystem,
    message: str,
    patient_regex: str = None,
    preserve_number_letter: bool = False,
) -> Any:
    cas = Cas(typesystem=typesystem)
    cas.sofa_string = message
    cas.sofa_mime = "text/plain"
    new_patient_regex = return_matched_regex(
        message, PATIENT_REGEX
    ) or return_matched_regex(message, NAME_REGEX)

    current_regex = PATHO_REGEX.copy()
    if patient_regex is not None:
        current_regex.insert(
            0,
            (
                patient_regex,
                MAIN_PHI_MAPPING["NAME_PHI_NAME"],
                PHI_MAPPING["PATIENT_PHI_NAME"],
                0,
            ),
        )
    if new_patient_regex is not None:
        current_regex.insert(
            0,
            (
                new_patient_regex,
                MAIN_PHI_MAPPING["NAME_PHI_NAME"],
                PHI_MAPPING["PATIENT_PHI_NAME"],
                0,
            ),
        )
    # print(patient_regex)
    # print(new_patient_regex)
    for regex, general_label, label, group in current_regex:
        for match in re.finditer(string=message, pattern=regex) or []:
            current_label = label
            start = match.start()
            end = match.end()
            # initial = message[start:end]
            # print(specific_class, message[start:end], start, end)
            if (
                " blau" in message[start - 5 : end + 5]
                or "blau " in message[start - 5 : end + 5]
            ) or message[start:end] in {"PD L", "Herr sieht"}:
                continue
            start, end = fix_indices(
                message, start, end, current_label, group, match, preserve_number_letter
            )
            if label == PHI_MAPPING["STUDY_PHI_NAME"]:
                maybe_start, maybe_end = get_study_name(message, start, end)
                if maybe_start is None or maybe_end is None:
                    continue
                start, end = maybe_start, maybe_end
            if "Dr. Jansen" in message[start:end]:
                # Add the raw part first
                add_label_without_clashes(
                    cas=cas,
                    main_class=general_label,
                    specific_class=current_label,
                    start=start - len("raw"),
                    end=start,
                    probability=1,
                    source="patho-regex+label split",
                    always_update=True,
                )
                # Then add the jansen part later
            elif "EngersPD" in message[start:end]:
                # Add the Engers part first
                add_label_without_clashes(
                    cas=cas,
                    main_class=general_label,
                    specific_class=current_label,
                    start=start,
                    end=start + len("Prof. Dr. R. Engers"),
                    probability=1,
                    source="patho-regex+label split",
                    always_update=True,
                )
                # Then add the other part later
                start += len("Prof. Dr. R. Engers")
                end = start + len("PD Dr. med. K. \nEngels")
            elif "SchulerProf" in message[start:end]:
                # Add the Engers part first
                add_label_without_clashes(
                    cas=cas,
                    main_class=general_label,
                    specific_class=current_label,
                    start=start,
                    end=start + len("Univ.-Prof. Dr. med. M. Schuler"),
                    probability=1,
                    source="patho-regex+label split",
                    always_update=True,
                )
                # Then add the other part later
                start += len("Univ.-Prof. Dr. med. M. Schuler")
                end = start + len("Prof. Schuler")
            elif "Univ. Prof. Dr. H.-U" == message[start:end]:
                end += len(". \nSchildhaus")

            if any(message[start:end] == skip for skip in SKIP_FIELDS):
                continue
            if (
                current_label == PHI_MAPPING["DATE_PHI_NAME"]
                and "geb." in message[start - 10 : start]
            ):
                current_label = PHI_MAPPING["BIRTHDATE_PHI_NAME"]

            if regex == MULTIPLE_DOCTORS:
                chosen_sep = ""
                for sep in [", ", ",", "/", " und "]:
                    if sep in match.group(3):
                        chosen_sep = sep
                        break
                for doc in message[start:end].split(chosen_sep):
                    add_label_without_clashes(
                        cas=cas,
                        main_class=general_label,
                        specific_class=current_label,
                        start=start + message[start:end].find(doc),
                        end=start + message[start:end].find(doc) + len(doc),
                        probability=1,
                        source="patho-regex",
                        always_update=True,
                    )
            else:
                add_label_without_clashes(
                    cas=cas,
                    main_class=general_label,
                    specific_class=current_label,
                    start=start,
                    end=end,
                    probability=1,
                    source="patho-regex",
                    always_update=True,
                )
    return cas


def return_redact_regex(
    search: Pirate,
    patient_id: str,
) -> str:
    df = search.steal_bundles_to_dataframe(
        resource_type="Patient",
        request_params={"_id": patient_id, "_sort": "_id"},
        fhir_paths=[
            ("family", "name.family"),
            ("given", "name.given"),
        ],
    ).explode("family")
    df["given"] = df["given"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    df["given"] = df["given"].str.replace("-", r"\n?-\n?")
    df["given"] = df["given"].str.replace(" ", r"[-\n\r\s]+")
    new_col = df["family"] + r",[\n\r\s]+" + df["given"]
    # patient_info = list(df.family.values) + list(df.given.values)
    return "|".join(new_col.values)


def main(opt: argparse.Namespace) -> None:  # noqa
    jsons = opt.input_folder.glob("*.json")
    reports = opt.output_folder
    reports.mkdir(parents=True, exist_ok=True)
    auth = Ahoy(auth_method="env", auth_url=BASIC_AUTH, refresh_url=REFRESH_AUTH)
    search = Pirate(
        num_processes=1,
        auth=auth,
        base_url=SEARCH_URL,
        cache_folder="cache",
        print_request_url=False,
    )
    # new_typesystem = True
    # if new_typesystem:
    types = sorted(MAIN_PHI_MAPPING.values())
    typesystem = cassis.load_typesystem(Path("TUDarmstadtTypeSystem.xml"))
    if typesystem.get_type(NAMED_ENTITY_CAS) is not None:
        ner_type = typesystem.get_type(NAMED_ENTITY_CAS)
    else:
        ner_type = typesystem.create_type(
            name=NAMED_ENTITY_CAS, supertypeName="uima.tcas.Annotation"
        )
        add_annotation_features(typesystem, ner_type)

    for x in types:
        t = typesystem.create_type(
            name=f"{NAMED_ENTITY_TYPE}.{x}", supertypeName=ner_type.name
        )
        add_annotation_features(typesystem, t)

    # Add some information about the versions
    t = typesystem.create_type(
        name=f"{DOCUMENT_TYPE}.VersionInfo", supertypeName=UIMA_ANNOTATION
    )
    typesystem.create_feature(domainType=t, name="version", rangeType=TYPE_NAME_STRING)
    typesystem.create_feature(
        domainType=t, name="gitVersion", rangeType=TYPE_NAME_STRING
    )
    typesystem.create_feature(domainType=t, name="gitHash", rangeType=TYPE_NAME_STRING)
    typesystem.create_feature(
        domainType=t, name="supermodelName", rangeType=TYPE_NAME_STRING
    )
    typesystem.create_feature(
        domainType=t, name="supermodelPath", rangeType=TYPE_NAME_STRING
    )
    typesystem.create_feature(
        domainType=t, name="submodelName", rangeType=TYPE_NAME_STRING
    )
    typesystem.create_feature(
        domainType=t, name="submodelPath", rangeType=TYPE_NAME_STRING
    )
    # else:
    #     typesystem = cassis.load_typesystem(
    #         Path.cwd() / "datasets" / "TUDarmstadtTypeSystem.xml"
    #     )

    typesystem.to_xml(reports / "TypeSystem.xml")
    for j in jsons:
        with j.open("r") as of:
            bundle: Dict = json.load(of)
        patient_redact_regex = None
        for entry in bundle.get("entry") or []:
            resource = entry.get("resource")
            if resource.get("resourceType") == "Observation":
                continue
            # Get the resource type
            r_id = resource.get("id")
            # if (
            #     r_id
            #     != "2627ffbec534dc308f33375997fc836031b98501f482f37999263950ec5da59d"
            # ):
            #     continue
            # print(i, r_id)
            if "subject" in resource:
                pat_id = (
                    resource.get("subject").get("reference").replace("Patient/", "")
                )
                if patient_redact_regex is None:
                    patient_redact_regex = return_redact_regex(
                        patient_id=pat_id,
                        search=search,
                    )
            # print(patient_redact_regex)
            positions_filename = reports / f"{r_id}.xmi"
            message, cas = None, None
            if "conclusion" in resource or []:
                message = resource.get("conclusion")
                cas = anonymize_message(
                    typesystem=typesystem, message=message, preserve_number_letter=False
                )

            assert patient_redact_regex is not None
            for form in resource.get("presentedForm") or []:
                message = decode_base64(form.get("data"))
                cas = anonymize_message(
                    typesystem=typesystem,
                    message=message,
                    patient_regex=patient_redact_regex,
                    preserve_number_letter=False,
                )
            if message is not None and cas is not None:
                cas.to_xmi(positions_filename)


# RUN: poetry run python -m deid_doc.deidentify --input_folder json_export --output_folder datasets/regex-new+manual-final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Folder where input JSONs are stored.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default="out",
        help="Folder where files will be stored.",
    )
    args = parser.parse_args()
    main(args)
