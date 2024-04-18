import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cassis
import pandas as pd
from cassis import TypeSystem, load_cas_from_xmi
from dotenv import load_dotenv

from dome.util.cas_handling import add_label_without_clashes, label_count
from dome.util.constants import NAMED_ENTITY_CAS, PHI_MAPPING
from dome.util.regex_collection import (
    MELANOM_REDACT_REGEX,
    RADIO_REDACT_REGEX,
    SPLIT_DOC_REGEX,
)
from dome.util.util import (
    generate_pathology_cas_without_letterhead,
    get_study_name,
    transform_averbis_into_cas,
)


def remove_space_begin_end(start: int, end: int, text: str) -> Tuple[int, int]:
    while text[start:end].endswith(" "):
        end -= 1
    while text[start:end].startswith(" "):
        start += 1
    return start, end


def split_radiology_data(
    type_system: TypeSystem,
    radiology_df: pd.DataFrame,
    percent_train_radio: int,
    max_radio_documents: int,
    min_percent_training: int,
    max_percent_training: int,
    output_folder: Path,
) -> Tuple[Dict[Tuple[str, str], int], List[Dict[str, str]]]:
    labels: Dict[Tuple[str, str], int] = {}
    filtered_df = radiology_df.loc[radiology_df["status"] == "final"].drop_duplicates(
        "diagnostic_report_id", keep="first"  # Keep only the oldest one
    )
    radio_annotations: Dict[str, List[Tuple[str, cassis.Cas]]] = {}
    current_docs = 0
    for row in filtered_df.itertuples():
        if current_docs >= max_radio_documents:
            break
        if len(row.text) < 1000:
            continue
        cas = cassis.Cas(typesystem=type_system)
        cas.sofa_string = row.text
        cas.sofa_mime = "text/plain"
        for regex, general_label, label, group in RADIO_REDACT_REGEX:
            for match in re.finditer(string=row.text, pattern=regex) or []:
                start = match.start()
                end = match.end()
                if group is not None:
                    start, end = match.span(group)
                start, end = remove_space_begin_end(start, end, row.text)
                if label == PHI_MAPPING["STUDY_PHI_NAME"]:
                    maybe_start, maybe_end = get_study_name(row.text, start, end)
                    if maybe_start is None or maybe_end is None:
                        continue
                    start, end = maybe_start, maybe_end
                if label == PHI_MAPPING["STAFF_PHI_NAME"]:
                    # print(regex, row.text[start:end])
                    for doc_regex in [SPLIT_DOC_REGEX]:
                        for sub_match in re.finditer(
                            string=row.text[start:end], pattern=doc_regex
                        ):
                            split_begin = start + sub_match.start()
                            split_end = split_begin + len(sub_match.group(0))
                            # print("sub", row.text[split_begin:split_end])
                            # split_begin, split_end = remove_space_begin_end(
                            #     split_begin, split_end, row.text
                            # )
                            add_label_without_clashes(
                                cas=cas,
                                main_class=general_label,
                                specific_class=label,
                                start=split_begin,
                                end=split_end,
                                probability=1,
                                source="radio-regex",
                                always_update=True,
                            )
                else:
                    add_label_without_clashes(
                        cas=cas,
                        main_class=general_label,
                        specific_class=label,
                        start=start,
                        end=end,
                        probability=1,
                        source="radio-regex",
                        always_update=True,
                    )
        if len(cas.select(NAMED_ENTITY_CAS)) == 0:
            continue
        label_count(cas, labels)
        radio_annotations.setdefault(row.per_number, [])
        radio_annotations[row.per_number].append((row.diagnostic_report_id, cas))
        current_docs += 1

    print("Radiology Findings")
    print("Total Patients", len(radio_annotations))
    print("Labels", labels)

    radio_documents = split_documents(
        annotations=radio_annotations,
        total_labels=labels,
        percent_train=percent_train_radio,
        max_percent_training=max_percent_training,
        min_percent_training=min_percent_training,
        document_type="radio",
        output_folder=output_folder,
    )

    return labels, radio_documents


def split_pathology_data(
    type_system: TypeSystem,
    pathology_folder: Path,
    percent_train_patho: int,
    output_folder: Path,
    min_percent_training: int,
    max_percent_training: int,
    remove_letterhead: bool = False,
) -> Tuple[Dict[Tuple[str, str], int], List[Dict[str, str]]]:
    labels: Dict[Tuple[str, str], int] = {}
    df = pd.read_csv("json_to_resource_mapping.csv")
    df = df.loc[df["type"] == "DiagnosticReport", ["per_number", "id"]].drop_duplicates(
        "id"
    )
    patho_annotations: Dict[str, List[Tuple[str, cassis.Cas]]] = {}
    for row in df.itertuples():
        doc = pathology_folder / f"{row.id}.xmi"
        if not doc.exists():
            continue
        cas = cassis.load_cas_from_xmi(doc, type_system)
        if remove_letterhead:
            cas = generate_pathology_cas_without_letterhead(cas)
        label_count(cas, labels)
        patho_annotations.setdefault(row.per_number, [])
        patho_annotations[row.per_number].append((row.id, cas))

    print("Pathology Findings")
    print("Total Patients", len(patho_annotations))
    print("Labels", labels)

    patho_documents = split_documents(
        annotations=patho_annotations,
        total_labels=labels,
        percent_train=percent_train_patho,
        max_percent_training=max_percent_training,
        min_percent_training=min_percent_training,
        document_type="patho",
        output_folder=output_folder,
    )
    return labels, patho_documents


def add_missing_annotations(original_text: str) -> List[Dict[str, Any]]:
    new_annotations = []
    for regex, general_label, label, group in MELANOM_REDACT_REGEX:
        for match in (
            re.finditer(string=original_text, pattern=regex, flags=re.I | re.M) or []
        ):
            start = match.start()
            end = match.end()
            if group is not None:
                start, end = match.span(group)
            if label == PHI_MAPPING["STUDY_PHI_NAME"]:
                maybe_start, maybe_end = get_study_name(original_text, start, end)
                if maybe_start is None or maybe_end is None:
                    continue
                start, end = maybe_start, maybe_end
            new_annotations.append(
                {
                    "begin": start,
                    "end": end,
                    "coveredText": original_text[start:end],
                    "type": general_label,
                    "kind": label,
                }
            )
    return new_annotations


def convert_json_to_cas(
    type_system: TypeSystem,
    melanom_batches: List[Dict],
    resource_type: str = "DiagnosticReport",
) -> Dict[str, List[Tuple[str, cassis.Cas]]]:
    annotations: Dict[str, List[Tuple[str, cassis.Cas]]] = {}
    for export in melanom_batches:
        for doc in export["payload"]["textAnalysisResultDtos"]:
            m = re.search(
                r"Patient:([0-9a-f]+)_" + resource_type + ":([0-9a-f]+).txt",
                doc["documentName"],
            )
            assert m is not None, doc["documentName"]
            patient_id = m.group(1)
            diagnostic_report_id = m.group(2)
            cas = transform_averbis_into_cas(
                doc["annotationDtos"],
                type_system,
                add_missing_annotations,
            )
            annotations.setdefault(patient_id, [])
            annotations[patient_id].append((diagnostic_report_id, cas))
    return annotations


def create_annotations_from_cas(
    type_system: TypeSystem, cas_folder: Path, resource_type: str = "DiagnosticReport"
) -> Dict[str, List[Tuple[str, cassis.Cas]]]:
    annotations: Dict[str, List[Tuple[str, cassis.Cas]]] = {}
    for f in cas_folder.glob("*xmi"):
        with open(f, "rb") as file:
            m = re.search(
                r"Patient:([0-9a-f]+)_" + resource_type + ":([0-9a-f]+).xmi", file.name
            )
            assert m is not None, file.name
            patient_id = m.group(1)
            resource_id = m.group(2)
            cas = load_cas_from_xmi(file, type_system)
            annotations.setdefault(patient_id, [])
            annotations[patient_id].append((resource_id, cas))
    return annotations


def find_documents_with_class(
    annotations: Dict[str, List[Tuple[str, cassis.Cas]]],
    annotation_type: Tuple[str, str],
) -> List[Tuple[str, int]]:
    return [
        (i, num)
        for i, num in sorted(
            [
                (
                    patient,
                    sum(
                        annotation_type[0] == token.type.name.split(".")[-1]
                        and annotation_type[1] == token.entityType
                        for cas_id, cas in annotations[patient]
                        for token in cas.select(NAMED_ENTITY_CAS)
                    ),
                )
                for patient in annotations
            ],
            key=lambda x: x[1],
        )
        if num > 0
    ]


def make_annotations_equal(
    annotations: Dict[str, List[Tuple[str, cassis.Cas]]],
    total_labels: Dict[Tuple[str, str], int],
    sampled_patients: List[str],
    max_percent: float,
    min_percent: float,
) -> None:
    removed_patient = []
    while True:
        train_labels: Dict[Tuple[str, str], int] = {}
        filtered_annotations = {
            pat_id: cases
            for pat_id, cases in annotations.items()
            if pat_id in sampled_patients
        }
        for patient in filtered_annotations:
            for _, cas in annotations[patient]:
                label_count(cas, train_labels)
        all_fine = True
        for label, total in sorted(total_labels.items(), key=lambda x: x[1]):
            train_total = train_labels[label] if label in train_labels else 0
            percentage = round((100 * train_total) / total, 2)
            # print(label, total, train_total, percentage, percentage > max_percent)
            if label[0] == "Name" and (label[1] == "PATIENT" or label[1] == "OTHER"):
                continue
            if label[1] == "PROFESSION" or label[1] == "ORGANIZATION":
                continue
            if total > 1 and percentage > max_percent:
                ordered_patients = find_documents_with_class(
                    filtered_annotations, label
                )
                # print(ordered_patients)
                if len(ordered_patients) == 1:
                    continue
                max_removed_number = train_total - ((min_percent * total) / 100)
                min_removed_number = train_total - ((max_percent * total) / 100)
                min_pat_ids = [
                    pat_id
                    for pat_id, num in ordered_patients
                    if min_removed_number < num < max_removed_number
                ]
                if len(min_pat_ids) > 0:
                    to_remove = min_pat_ids[0]
                    # if (
                    #     to_remove
                    #     == "10910b157f27b81a50e52f4ba98673a7b2d61b1cda6b64a3c44fbde3f9911d20"
                    # ):
                    #     to_remove = min_pat_ids[1]
                else:
                    to_remove = ordered_patients[0][0]
                sampled_patients.remove(to_remove)
                removed_patient.append((label, to_remove))
                all_fine = False
                break
        if all_fine:
            break
    # print(removed_patient)
    print(f"Min percentage allowed for training: {min_percent}")
    print(f"Max percentage allowed for training: {max_percent}")
    for label, total in sorted(total_labels.items(), key=lambda x: x[1]):
        train_total = train_labels[label] if label in train_labels else 0
        percentage = round((100 * train_total) / total, 2)
        print(
            f"label: {label}, "
            f"total: {total}, "
            f"training: {train_total}, "
            f"training percentage: {percentage}, "
            f"percentage more than max? {percentage > max_percent}"
        )


def split_melanom_documents(
    type_system: TypeSystem,
    melanom_batches: List[Dict],
    percent_train_melanoma: int,
    min_percent_training: int,
    max_percent_training: int,
    output_folder: Path,
) -> Tuple[Dict[Tuple[str, str], int], List[Dict[str, str]]]:
    labels: Dict[Tuple[str, str], int] = {}
    melanom_annotations = convert_json_to_cas(
        type_system, melanom_batches, resource_type="Observation"
    )
    for patient in melanom_annotations:
        for _, cas in melanom_annotations[patient]:
            label_count(cas, labels)
    patients_melanom = sorted(melanom_annotations.keys())
    print("Melanom Verlaufsdokumentation")
    print("Total Patients", len(patients_melanom))
    print("Labels", labels)

    melanoma_documents = split_documents(
        annotations=melanom_annotations,
        total_labels=labels,
        percent_train=percent_train_melanoma,
        max_percent_training=max_percent_training,
        min_percent_training=min_percent_training,
        document_type="melanoma",
        output_folder=output_folder,
    )

    return labels, melanoma_documents


def split_documents(
    annotations: Dict[str, List[Tuple[str, cassis.Cas]]],
    total_labels: Dict[Tuple[str, str], int],
    percent_train: int,
    max_percent_training: int,
    min_percent_training: int,
    document_type: str,
    output_folder: Path,
) -> List[Dict[str, Any]]:
    num_training = math.ceil((percent_train * len(annotations)) / 100)
    sampled = random.sample(sorted(annotations.keys()), num_training)
    make_annotations_equal(
        annotations,
        total_labels=total_labels,
        sampled_patients=sampled,
        max_percent=max_percent_training,
        min_percent=min_percent_training,
    )
    train_annotations = [
        (pat_id, obs_id, annotation)
        for pat_id in annotations
        for obs_id, annotation in annotations[pat_id]
        if pat_id in sampled
    ]
    test_annotations = [
        (pat_id, obs_id, annotation)
        for pat_id in annotations
        for obs_id, annotation in annotations[pat_id]
        if pat_id not in sampled
    ]
    print("Train Documents", len(train_annotations))
    print("Test Documents", len(test_annotations))
    document_records = []
    for target_folder, output_annotations in [
        (output_folder / "train", train_annotations),
        (output_folder / "test", test_annotations),
    ]:
        for pat_id, resource_id, annotation in output_annotations:
            document_records.append(
                {
                    "document_type": document_type,
                    "patient_id": pat_id,
                    "document_id": resource_id,
                    "phase": target_folder.name,
                }
            )
            annotation.to_xmi((target_folder / (resource_id + ".xmi")))
    return document_records


def main(
    type_system: TypeSystem,
    pathology_folder: Path,
    melanom_documents: List[Dict],
    radiology_df: pd.DataFrame,
    percent_train_patho: int,
    percent_train_melanoma: int,
    percent_train_radio: int,
    max_radio_documents: int,
    remove_pathology_letterhead: bool,
    min_percent_training: int,
    max_percent_training: int,
    output_folder: Path,
) -> None:

    train_folder = output_folder / "train"
    test_folder = output_folder / "test"
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    final_labels = set()
    records = []
    if percent_train_radio > 0:
        radio_labels, radio_records = split_radiology_data(
            type_system=type_system,
            radiology_df=radiology_df,
            percent_train_radio=percent_train_radio,
            max_radio_documents=max_radio_documents,
            max_percent_training=max_percent_training,
            min_percent_training=min_percent_training,
            output_folder=output_folder,
        )
        print()
        for kind in radio_labels:
            final_labels.add(kind)
        records += radio_records
    if percent_train_patho > 0:
        patho_labels, patho_records = split_pathology_data(
            type_system=type_system,
            pathology_folder=pathology_folder,
            percent_train_patho=percent_train_patho,
            output_folder=output_folder,
            remove_letterhead=remove_pathology_letterhead,
            max_percent_training=max_percent_training,
            min_percent_training=min_percent_training,
        )
        print()
        for kind in patho_labels:
            final_labels.add(kind)
        records += patho_records

    if percent_train_melanoma > 0:
        melanom_labels, melanoma_records = split_melanom_documents(
            type_system=type_system,
            melanom_batches=melanom_documents,
            percent_train_melanoma=percent_train_melanoma,
            output_folder=output_folder,
            max_percent_training=max_percent_training,
            min_percent_training=min_percent_training,
        )
        print()
        for kind in melanom_labels:
            final_labels.add(kind)
        records += melanoma_records

    type_system.to_xml(output_folder / "TypeSystem.xml")
    pd.DataFrame(records).to_csv(output_folder / "documents.csv", index=False)
    print("Final pairs", sorted(final_labels))
    print(sorted(f"{a}_{b}" for a, b in final_labels))


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    load_dotenv(args.env.absolute())

    m = Path(os.environ["MELANOMA_PATH"])
    main(
        type_system=cassis.load_typesystem(Path(os.environ["TYPESYSTEM_PATH"])),
        pathology_folder=Path(os.environ["PATHOLOGY_PATH"]),
        melanom_documents=[json.load(f.open("r")) for f in m.glob("*json")],
        radiology_df=pd.read_csv(os.environ["RADIO_PATH"]),
        percent_train_patho=int(os.environ["PERCENT_PATHO_PATIENTS"]),
        percent_train_melanoma=int(os.environ["PERCENT_MELANOMA_PATIENTS"]),
        percent_train_radio=int(os.environ["PERCENT_RADIO_PATIENTS"]),
        max_radio_documents=int(os.environ["MAX_RADIO"]),
        remove_pathology_letterhead=bool(int(os.environ["REMOVE_LETTERHEAD"])),
        output_folder=Path(os.environ["XMI_OUTPUT"]),
        max_percent_training=int(os.environ["MAX_PERCENT_TRAINING"]),
        min_percent_training=int(os.environ["MIN_PERCENT_TRAINING"]),
    )
