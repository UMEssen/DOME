import base64
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cassis
from cassis import Cas, TypeSystem, typesystem
from tqdm import tqdm

from deid_doc.util.cas_handling import (
    add_annotation_features,
    add_label_without_clashes,
)
from deid_doc.util.constants import (
    MAIN_PHI_MAPPING,
    NAMED_ENTITY_CAS,
    NAMED_ENTITY_TYPE,
)
from deid_doc.util.regex_collection import SKIP_STUDIES


def get_study_name(
    message: str, start: int, end: int
) -> Tuple[Optional[int], Optional[int]]:
    if any(a == message[start:end] for a in SKIP_STUDIES):
        return None, None

    if message[start:end].isdigit() or message[start:end] in {"Lung"}:
        # print("text", message[start - 30:end + 9])
        # Invert the text before the start and find the first space, that's the word before
        index = message[: start - 1][::-1].find(" ")
        # print("new word", message[start - index - 1:start-1])
        # Check if the word before is in the list of skippable elements
        if any(a == message[start - index + 1 : start - 1] for a in SKIP_STUDIES):
            return None, None
        # Otherwise add it
        start -= index + 1
        # print("new annotation", message[start:end])

    if message[end] == "-":
        end -= 1
    return start, end


def decode_base64(message: str) -> str:
    return base64.b64decode(message).decode("utf-8")


def check_intersection(bound1: Tuple[int, int], bound2: Tuple[int, int]) -> bool:
    init_1, end_1 = bound1
    init_2, end_2 = bound2
    if (
        init_1 <= init_2 < end_1
        or init_1 <= end_2 - 1 <= end_1
        or init_2 <= init_1 < end_2
        or init_2 <= end_1 - 1 <= end_2
    ):
        return True
    return False


def get_annotations_outside_letterhead_for_pathology(
    document: str,
    annotations: List[typesystem.Any],
) -> List[typesystem.Any]:
    letterhead_indices = get_letterhead_for_pathology(document)
    return [
        annotation
        for annotation in annotations
        if not any(
            check_intersection(index, (annotation.begin, annotation.end))
            for index in letterhead_indices
        )
    ]


def get_letterhead_for_pathology(document: str) -> List[Tuple[int, int]]:
    if (
        re.search(
            r"(Hufelandstr. 55, 45122 Essen|"
            r"Institut für (Pathologie|Neuropathologie|Pathologie und Neuropathologie)"
            r"\s*\nUniversitätsklinikum Essen)\n",
            document,
        )
        is None
    ):
        return []
    # These are just heuristics, will only work for these documents
    document_type_init, document_end = None, None
    for document_kind in [
        r"Korrekturbefund\n",
        r"Nachbericht:\n",
        r"Zytologischer Befundbericht\n",
        r"Befundbericht\n",
        r"Schnellschnittdiagnose( \(.*?\))?\n",
        r"Nachbericht mit Makroskopie\n",
        r"Nachbericht \(Korrekturbefund\):",
        r"Nachbericht \(nach Paraffineinbettung\):",
        r"[0-9]\.? Befundbericht",
        r"Untersuchter Block:\n",
        r"Sektionsbericht",
        r"Dermatopathologischer Beratungsfall\n",
        r"Neuropathologisch-autoptischer Befund",
        r"Dermatopathologischer Nachbericht\n",
        r"Dermatopathologisches Gutachten\n",
        r"Korrekturbefund \- Nachbericht\n",
        r"Korrekturbefund/Nachbericht\n",
        r"Klinische Angabe(n)?( \(.*?\))?:?\n",
        r"Schnellschnittgewebe zu .*?:\n",
        r"Makroskopie:\n",
    ]:
        # print(document_kind, re.search(document_kind, doc))
        if re.search(document_kind, document) is not None:
            chosen_match = [m for m in re.finditer(document_kind, document)][0]
            document_type_init = chosen_match.start()
            if chosen_match.end() == len(document):
                return [(0, document_type_init)]
            break

    assert document_type_init is not None, "Init Missing\n" + document[:200]

    # Find the end
    for final in [
        r"Kein Anhalt für Malignität.(\s*\n)",
        r"\n(Univ\.|Prof\.|PD |Dr\.)",
    ]:
        # print(final, re.search(final, doc))
        if re.search(final, document) is not None:
            document_end = [m.span(1)[0] for m in re.finditer(final, document)][-1]
            break

    assert document_end is not None, (
        f"Begin {document_type_init}, End {len(document)} End Missing\n"
        + document[-200:]
    )

    init_date = (
        r"(Name: Seite Eingang am:"  # |Eingang am:"
        r"|Name: Seite Eingangsdatum:"  # |Eingangsdatum:"
        r"|Seite 1 von 2)"
    )
    end_date = (
        r"(\n(BarcodeDokumentNr)?\s+[A-Z][0-9]+\/[0-9]+(\+[0-9]+)?\n|Seite 2 von 2)"
    )
    dates_init = [
        m for m in re.finditer(init_date, document) if document_type_init < m.start()
    ]
    dates_end = [
        m for m in re.finditer(end_date, document) if document_type_init < m.end()
    ]
    assert len(dates_init) == len(dates_end), (
        f"Document: {document}\n"
        f"Begin: {document[document_type_init - 50:document_type_init]}, {document_type_init}\n"
        f"End: {document[document_end:document_end + 50]}, {document_end}\n"
        f"Start values: {dates_init}\n"
        f"End values: {dates_end}\n"
    )
    date_indices = list(
        zip([d.start() for d in dates_init], [d.end() for d in dates_end])
    )
    # Also add the beginning of the document
    # if check_intersection((0, document_type_init), date_indices[0]):
    #     date_indices[0] = (
    #         min(date_indices[0][0], 0),
    #         max(date_indices[0][1], document_type_init),
    #     )
    # else:
    date_indices.insert(0, (0, document_type_init))

    # And the end
    if check_intersection((document_end, len(document)), date_indices[-1]):
        date_indices[-1] = (
            min(date_indices[-1][0], document_end),
            max(date_indices[-1][1], len(document)),
        )
    else:
        date_indices.append((document_end, len(document)))

    return date_indices


def transform_averbis_into_cas(
    annotations: Dict,
    type_system: TypeSystem,
    fix_annotation_function: Callable = lambda x, _: [x],
    new_annotation_function: Callable = lambda x: [],
) -> Cas:
    document_text = [
        doc["coveredText"]
        for doc in annotations
        if doc["type"] == "de.averbis.types.health.DeidentifiedDocument"
    ]
    assert len(document_text) == 1, annotations
    document = document_text[0]
    # Create the CAS
    cas = Cas(typesystem=type_system)
    cas.sofa_string = document
    cas.sofa_mime = "text/plain"
    annotations += new_annotation_function(document)
    # Iterate through the other annotations
    for annotation in annotations:
        # Skip the text
        if annotation["type"] in [
            "de.averbis.types.health.DeidentifiedDocument",
            "de.averbis.types.health.DocumentAnnotation",
            "de.averbis.types.health.PatientInformation",
            "de.averbis.types.health.Hospitalisation",
            "de.averbis.types.health.ClinicalSection",
            "de.averbis.types.health.ClinicalSectionKeyword",
        ]:
            continue
        # Change the type to the actual last part of the type
        annotation["type"] = annotation["type"].split(".")[-1]
        # Some types don't have kinds
        if "kind" not in annotation and annotation["type"] in {
            MAIN_PHI_MAPPING["PROFESSION_PHI_NAME"],
            MAIN_PHI_MAPPING["AGE_PHI_NAME"],
        }:
            annotation["kind"] = annotation["type"]
        assert "kind" in annotation, annotation
        # Function that fixes the annotation and returns a list of annotations to add
        annotations_to_add = fix_annotation_function(annotation, document)
        for annotation_to_add in annotations_to_add:
            new_type = annotation_to_add["type"]
            # If the typesystem does not have the type yet
            if not type_system.contains_type(f"{NAMED_ENTITY_TYPE}.{new_type}"):
                t = type_system.create_type(
                    name=f"{NAMED_ENTITY_TYPE}.{new_type}",
                    supertypeName=NAMED_ENTITY_CAS,
                )
                add_annotation_features(type_system, t)
            # Add it to the List and check for clashes
            add_label_without_clashes(
                cas=cas,
                main_class=annotation_to_add["type"],
                specific_class=annotation_to_add["kind"],
                start=annotation_to_add["begin"],
                end=annotation_to_add["end"],
                probability=1,
                always_update=True,
                always_add=False,
                source="averbis",
            )
    return cas


def annotation_eval(
    type_system: TypeSystem, folder: Path
) -> Tuple[Dict[str, Set], Dict[str, Set]]:
    eval_type: Dict[str, Set] = {}
    eval_kind: Dict[str, Set] = {}
    for doc in tqdm(list(folder.rglob("*.xmi"))):
        data = cassis.load_cas_from_xmi(doc, type_system)
        for token in data.select(NAMED_ENTITY_CAS):
            assert len(token.get_covered_text()) > 0, doc.name
            ann_type = token.type.name.split(".")[-1]
            specific_type = f"{ann_type}_{token.entityType}"
            if ann_type not in eval_type:
                eval_type[ann_type] = set()
            if specific_type not in eval_kind:
                eval_kind[specific_type] = set()
            eval_type[ann_type].add(token.get_covered_text())
            eval_kind[specific_type].add(token.get_covered_text())
    return eval_type, eval_kind


def fix_annotations_after_span(
    annotations: List[cassis.typesystem.Any],
    offset: int,
    span_end: int,
) -> None:
    for annotation in annotations:
        # The annotation begins after the end
        if annotation.begin >= span_end:
            # we need to change the begin/end values
            annotation.begin = annotation.begin - offset
            annotation.end = annotation.end - offset
            assert annotation.begin >= 0 and annotation.end >= 0, annotation


def generate_pathology_cas_without_letterhead(
    cas: Cas,
) -> Cas:
    new_cas = Cas(typesystem=cas.typesystem)
    letterhead_indices = get_letterhead_for_pathology(cas.sofa_string)
    current_offset = 0
    new_cas.sofa_string = ""
    current_annotations = [
        deepcopy(annotation)
        for annotation in cas.select(NAMED_ENTITY_CAS)
        if not any(
            check_intersection((annotation.begin, annotation.end), span)
            for span in letterhead_indices
        )
    ]
    # for element in current_annotations:
    #     print(element, element.get_covered_text())
    updated_position = 0
    for start, end in letterhead_indices:
        new_cas.sofa_string += cas.sofa_string[current_offset:start]
        updated_position += end - start
        fix_annotations_after_span(
            annotations=current_annotations,
            offset=end - start,
            span_end=end - updated_position,
        )
        current_offset = end
    new_cas.sofa_string += cas.sofa_string[current_offset:]
    # print()
    # print()
    # print(new_cas.sofa_string)
    new_cas.add_annotations(current_annotations)
    return new_cas


# Thank you https://stackoverflow.com/questions/8733233/filtering-out-certain-bytes-in-python
def valid_xml_char_ordinal(c: str) -> bool:
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF
        or codepoint in (0x9, 0xA, 0xD)
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )
