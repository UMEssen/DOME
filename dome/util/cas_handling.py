import logging
import math
import re
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Tuple, Union

import cassis.typesystem
from cassis import Cas, TypeSystem
from cassis.typesystem import TYPE_NAME_FLOAT, TYPE_NAME_STRING

from dome.util.constants import (
    DAYS_WEEK,
    MAIN_PHI_MAPPING,
    NAMED_ENTITY_CAS,
    NAMED_ENTITY_TYPE,
    NO_DEID_DATES,
    PHI_MAPPING,
    SENTENCE_CHUNK_TYPE,
)

logger = logging.getLogger(__name__)
try:
    import locale

    locale.setlocale(locale.LC_ALL, "de_DE.utf8")
except Exception:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")


def get_entities(annotations: List[cassis.typesystem.Any]) -> Generator:
    for token in annotations:
        # print(annotation["type"], annotation["kind"])
        yield (
            token.begin,
            token.end,
        ), f"{token.type.name.split('.')[-1]}_{token.entityType}"


def get_entities_for_evaluation(
    annotations: List[cassis.typesystem.Any], blindness: int = 2
) -> Generator:
    for token in annotations:
        ann_type = token.type.name.split(".")[-1]
        yield {
            "start": token.begin,
            "end": token.end,
            "label": f"{ann_type}_{token.entityType}"
            if blindness == 2 and ann_type != "Other"
            else (f"{ann_type}" if blindness == 1 else "REDACTED"),
        }


def add_annotation_features(typesystem: TypeSystem, t: cassis.typesystem.Type) -> None:
    typesystem.create_feature(
        domainType=t, name="entityType", rangeType=TYPE_NAME_STRING
    )
    typesystem.create_feature(domainType=t, name="score", rangeType=TYPE_NAME_FLOAT)
    typesystem.create_feature(domainType=t, name="source", rangeType=TYPE_NAME_STRING)


def check_feasibility(cas: Cas, new_position: Tuple) -> List:
    start_new, end_new = new_position
    matches = []
    for token in cas.select(NAMED_ENTITY_CAS):
        if not (end_new <= token.begin or start_new >= token.end):
            matches.append(token)
    matches.sort(key=lambda x: (x.begin, x.end))  # type: ignore
    return matches


def add_label_without_clashes(
    cas: Cas,
    main_class: str,
    specific_class: str,
    start: int,
    end: int,
    probability: float,
    source: str,
    always_update: bool = False,
    always_add: bool = False,
    verbose: bool = False,
) -> None:
    entity = cas.typesystem.get_type(f"{NAMED_ENTITY_TYPE}.{main_class}")
    new_entity = entity(
        begin=start,
        end=end,
        entityType=specific_class,
        score=probability,
        source=source,
    )

    clashes = check_feasibility(
        cas=cas,
        new_position=(start, end),
    )
    if len(clashes) == 0 or always_add:
        cas.add(new_entity)
    else:
        for value in clashes:
            existing_entity_text = value.get_covered_text()
            new_entity_text = cas.sofa_string[new_entity.begin : new_entity.end]
            if (
                existing_entity_text != new_entity_text
                or value.entityType != new_entity.entityType
            ) and verbose:
                print(
                    f"Current entity type: {value.entityType} | "
                    f"other entity type: {new_entity.entityType},\n"
                    f"Current entity value: '{existing_entity_text}' | "
                    f"other entity value: '{new_entity_text}'\n"
                    f"New entity type: {value.entityType}, "
                    f"new entity value "
                    f"{cas.sofa_string[min(value.begin, new_entity.begin): max(value.end, new_entity.end)]}."
                )
            if always_update:
                cas.remove(value)
                value.begin = min(value.begin, new_entity.begin)
                value.end = max(value.end, new_entity.end)
                value.source = "Merge overlapping entities"
                # The entity remains the one of the original
                new_entity = value
            elif len(existing_entity_text) > len(new_entity_text):
                cas.remove(value)
                new_entity = value
            else:
                cas.remove(value)
        cas.add(new_entity)


def generate_chunks(
    cas: Cas,
    ignore_ne: bool,
    label_map: Dict[str, int] = None,
    chunk_size: int = None,
    overlap_words: int = None,
) -> Generator[List[Dict[str, Union[str, int]]], None, None]:
    word_labels: List[Dict[str, Union[str, int]]] = []
    if ignore_ne:
        # If we ignore the Named Entities, we don't have actual tokens
        generate_tokens_from_string(
            cas.sofa_string,
            pos_init=0,
            pos_end=-1,
            superclass=0,
            subclass=0,
            word_labels=word_labels,
        )
    else:
        tokens_from_ground_truth(
            cas,
            word_labels,
            label_map,
        )

    if chunk_size is None or overlap_words is None:
        ranges = [(0, len(word_labels) - 1)]
    else:
        ranges = [
            (
                max((i * chunk_size) - overlap_words, 0),
                min(
                    ((i + 1) * chunk_size) + overlap_words,
                    len(word_labels) - 1,
                ),
            )
            for i in range(0, math.ceil(len(word_labels) / chunk_size))
        ]
    for chunk_init, chunk_end in ranges:
        chunked_list = []
        for chunk_id in range(chunk_init, chunk_end + 1):
            pos_init = int(word_labels[chunk_id]["begin"])
            pos_end = int(word_labels[chunk_id]["end"])
            matches = [
                (pos_init + match.start(), pos_init + match.end())
                for match in re.finditer(
                    r"([\wäöüßÄÜÖ]+|[^\s\w]+)", cas.sofa_string[pos_init:pos_end]
                )
            ]
            # print(
            #     cas.sofa_string[pos_init:pos_end],
            #     word_labels[chunk_id]["superclass"],
            #     word_labels[chunk_id]["subclass"],
            # )
            for i, (start, end) in enumerate(matches):
                base_dict = {
                    "text": cas.sofa_string[start:end],
                    "begin": start,
                    "end": end,
                }
                # We set this smaller labels to the same in three cases:
                # 1. The label is odd (B-), and we are at beginning of the token, it should also
                # stay B-
                # 2. The label is 0
                # 3. The label is even (I-), then we are intermediate, so everything should just
                # be intermediate
                if (
                    i == 0
                    or word_labels[chunk_id]["superclass"] == 0
                    or word_labels[chunk_id]["superclass"] % 2 == 0
                ):
                    base_dict["superclass"] = word_labels[chunk_id]["superclass"]
                    base_dict["subclass"] = word_labels[chunk_id]["subclass"]
                else:
                    # Otherwise it means we are at the beginning of the token, but we are looking
                    # at the characters that follow
                    base_dict["superclass"] = (
                        int(word_labels[chunk_id]["superclass"]) + 1
                    )
                    base_dict["subclass"] = int(word_labels[chunk_id]["subclass"]) + 1
                # print(
                #     cas.sofa_string[start:end],
                #     base_dict["superclass"],
                #     base_dict["subclass"],
                # )
                chunked_list.append(base_dict)
        yield chunked_list


def get_begin_intermediate_token(
    label: Union[str, int]
) -> Tuple[Union[str, int], Union[str, int]]:
    return (
        label if isinstance(label, int) else "B-" + label,
        label + bool(label) if isinstance(label, int) else "I-" + label,
    )


def generate_tokens_from_string(
    document: str,
    pos_init: int,
    pos_end: int,
    superclass: Union[str, int],
    subclass: Union[str, int],
    word_labels: List[Dict[str, Union[str, int]]],
    sentence_split: bool = False,
) -> None:
    matches = [
        (pos_init + match.start(), pos_init + match.end())
        for match in re.finditer(r"\S+", document[pos_init:pos_end])
        # for match in re.finditer(r"(\w+|[^\w\s]+)", document[pos_init:pos_end])
    ]
    # print("Text:", document[pos_init:pos_end], superclass, subclass)

    begin_superclass, inter_superclass = get_begin_intermediate_token(superclass)
    begin_subclass, inter_subclass = get_begin_intermediate_token(subclass)
    for i, (start, end) in enumerate(matches):
        if i == 0:
            # At the beginning the tokens start with B-
            word_labels.append(
                {
                    "text": document[start:end],
                    "superclass": begin_superclass,
                    "subclass": begin_subclass,
                    "begin": start,
                    "end": end,
                }
            )
            # print(f"B: {document[start:end]}: ({begin_superclass}, {begin_subclass})")
        else:
            # Then they start with I-
            word_labels.append(
                {
                    "text": document[start:end],
                    "superclass": inter_superclass,
                    "subclass": inter_subclass,
                    "begin": start,
                    "end": end,
                }
            )
            # print(f"I: {document[start:end]}: ({inter_superclass}, {inter_subclass})")


def tokens_from_ground_truth(
    cas: cassis.Cas,
    word_labels: List[Dict[str, Union[str, int]]],
    label_map: Dict[str, int] = None,
) -> None:
    annotations = sorted(cas.select(NAMED_ENTITY_CAS), key=lambda x: x.begin)  # type: ignore
    pos_before = 0
    for annotation in annotations:
        superclass = annotation.type.name.split(".")[-1]
        subclass = f"{annotation.type.name.split('.')[-1]}_{annotation.entityType}"
        # First generate the tokens for the part before the annotation
        generate_tokens_from_string(
            cas.sofa_string,
            pos_init=pos_before,
            pos_end=annotation.begin,
            superclass=0,
            subclass=0,
            word_labels=word_labels,
        )
        # Then the tokens for the annotated part
        generate_tokens_from_string(
            cas.sofa_string,
            pos_init=annotation.begin,
            pos_end=annotation.end,
            superclass=label_map[superclass] if label_map is not None else superclass,
            subclass=label_map[subclass] if label_map is not None else subclass,
            word_labels=word_labels,
        )
        pos_before = annotation.end
    # Then for the last part
    generate_tokens_from_string(
        cas.sofa_string,
        pos_init=pos_before,
        pos_end=len(cas.sofa_string),
        superclass=0,
        subclass=0,
        word_labels=word_labels,
    )


def generate_sentences(
    cas: Cas,
    type_system: TypeSystem,
    chunks: Generator[List[Dict[str, str]], None, None],
) -> None:
    for chunk in chunks:
        # Each chunk has
        # The actual word, then the specific_class, then a (begin, end) tuple
        # We want the begin of the very first token and the end of the very last
        sentence_entity = type_system.get_type(SENTENCE_CHUNK_TYPE)
        cas.add(sentence_entity(begin=chunk[0]["begin"], end=chunk[-1]["end"]))


def label_count(
    cas: Cas,
    labels: Dict[Tuple[str, str], int],
) -> None:
    for token in cas.select(NAMED_ENTITY_CAS):
        t = (token.type.name.split(".")[-1], token.entityType)
        if t in labels:
            labels[t] += 1
        else:
            labels[t] = 1


def identify_date(date: str) -> Optional[str]:
    # e.g. 2019.08.15 or 2019/08/15 or 2019-8-15
    match = re.fullmatch(r"\d{4}(\.|\/|\-)\d{1,2}(\.|\/|\-)\d{1,2}", date)
    if match is not None:
        return f"%Y{match.group(1)}%m{match.group(2)}%d"
    # e.g. 20.07.2021 or 20.7.2021 or 20072021
    match = re.fullmatch(r"\d{1,2}(\W*?)\d{1,2}(\W*?)\d{4}", date)
    if match is not None:
        return f"%d{match.group(1)}%m{match.group(2)}%Y"
    # e.g. 19.09.22 or 19/9/22 or 19-09-22
    match = re.fullmatch(r"\d{1,2}(\.|\/|\-)\d{1,2}(\.|\/|\-)\d{2}", date)
    if match is not None:
        return f"%d{match.group(1)}%m{match.group(2)}%y"
    # e.g. 08.2022 or 08/2022 or 08-2022
    match = re.fullmatch(r"\d{1,2}(\.|\/|\-)\d{4}", date)
    if match is not None:
        return f"%m{match.group(1)}%y"
    # e.g 30.08 or 30/8 or 30-8.
    match = re.fullmatch(r"\d{1,2}(\.|\/|\-)\d{1,2}(\.?)", date)
    if match is not None:
        return f"%d{match.group(1)}%m{match.group(2)}"
    # e.g. 29. or 9.
    match = re.fullmatch(r"\d{1,2}(\.?)", date)
    if match is not None:
        return f"%d{match.group(1)}"
    # e.g. 20. Jul or 8. Aug or 20. Dez
    match = re.fullmatch(r"\d{1,2}(\.?) [A-Z][äa-z]{2}", date)
    if match is not None:
        return f"%d{match.group(1)} %b"
    # e.g. 20. Juli or 8. August or 20. Dezember
    match = re.fullmatch(r"\d{1,2}(\.?) [A-Z][äa-z]+( (\d{2,4}))?", date)
    if match is not None:
        if match.group(2) is not None:
            if len(match.group(2)) == 2:
                return f"%d{match.group(1)} %B %y"
            else:
                return f"%d{match.group(1)} %B %Y"
        return f"%d{match.group(1)} %B"
    # e.g Juli 20. or Aug 8. or Dezember 20.
    match = re.fullmatch(r"[A-Z][äa-z]+ \d{1,2}(\.?)( (\d{2,4}))?", date)
    if match is not None:
        if match.group(2) is not None:
            if len(match.group(2)) == 2:
                return f"%B %d{match.group(1)} %y"
            else:
                return f"%B %d{match.group(1)} %Y"
        return f"%B %d{match.group(1)}"
    return None


def identify_timeframe_split(date: str) -> str:
    # match = re.fullmatch(
    #     r"\d{1,2}(\-| \- )\d{1,2}(.*?)\d{1,2}(.*?)\d{4}", date
    # )
    # if match is not None:
    #     return match.group(1)
    # e.g. 20.08-30.08 or 7.7.-8.8. or 12.9-15.09.
    match = re.fullmatch(
        r"\d{1,2}(\.|\/)(\d{1,2}(\.?))?(\-| \- )\d{1,2}(\.|\/)\d{1,2}(\.?)", date
    )
    if match is not None:
        return match.group(4)
    # e.g. Mittwoch 20.12.2020
    # TODO: Add handling for the days of the week
    match = re.fullmatch(
        r"[A-Z]*[a-z]+(.*?)\d{1,2}(\.|\/|\-)\d{1,2}(\.?)((\.|\/|\-)\d{2,4})?", date
    )
    if match is not None:
        return match.group(1)
    # e.g. 15-20.08.2020 or 15-20/8/2020
    match = re.fullmatch(r"\d{1,2}(\-| \- )\d{1,2}(\.|\/)\d{1,2}(\.|\/)\d{4}", date)
    if match is not None:
        return match.group(1)
    return ""


def substitute_value(
    annotation: cassis.typesystem.Any,
    config: Dict,
    document_id: str,
    fake_name: Tuple[str, str] = None,
    days_offset_ms: int = None,
    years_offset_ms: int = None,
    date_warning: int = 2007,
    only_superclass: bool = False,
) -> Tuple[str, bool]:
    tag_type = annotation.type.name.split(".")[-1]
    full_tag = f"{tag_type}_{annotation.entityType}"
    result = tag_type if only_superclass else full_tag, True
    annotation_error = (
        f"{document_id}: Error for annotation with type: {full_tag}, "
        f"at begin {annotation.begin} and end {annotation.end} "
        f"with value '{annotation.get_covered_text()}'."
    )
    substitution_method = (
        config["PHI"].get(tag_type.upper(), {}).get(annotation.entityType, "replace")
    )
    if substitution_method == "keep":
        result = annotation.get_covered_text(), False
    elif (
        substitution_method == "substitute"
        and annotation.entityType in PHI_MAPPING["PATIENT_PHI_NAME"]
        and fake_name is not None
    ):
        result = substitute(
            annotation=annotation,
            fake_name=fake_name,
        )
    elif substitution_method == "shift":
        result = shift(
            annotation=annotation,
            PHI_MAPPING=PHI_MAPPING,
            years_offset_ms=years_offset_ms,
            days_offset_ms=days_offset_ms,
            tag_type=tag_type,
            annotation_error=annotation_error,
            date_warning=date_warning,
            only_superclass=only_superclass,
            config=config,
            full_tag=full_tag,
        )
    elif substitution_method == "redact":
        result = "X" * len(annotation.get_covered_text()), False    
    else:
        result = tag_type if only_superclass else full_tag, True
    return result


def substitute(
    annotation: cassis.typesystem.Any,
    fake_name: Tuple[str, str],
) -> Tuple[str, bool]:
    result = None
    # TODO: This has not been tested
    title = None
    fake_first_name, fake_surname = fake_name
    to_substitute = annotation.get_covered_text()
    for possible_title in ["Herrn", "Herr", "Frau"]:
        if possible_title in to_substitute:
            title = possible_title
            break
    if title is not None:
        result = None
        if to_substitute == title:
            result = to_substitute, False
        else:
            to_substitute = re.sub("(Herr|Herrn|Frau) ", "", to_substitute)
    # TODO: Write a regex here instead
    names = to_substitute.split(" ")
    if len(names) == 1:
        result = (
            f"{title} {fake_surname}" if title is not None else fake_surname,
            False,
        )
    elif 1 < len(names) < 3:
        result = (
            f"{title} {fake_first_name} {fake_surname}"
            if title is not None
            else f"{fake_first_name} {fake_surname}"
        ), False
    else:
        raise Exception(names)
    return result


def shift(
    annotation: cassis.typesystem.Any,
    PHI_MAPPING: Dict,
    years_offset_ms: Optional[int],
    days_offset_ms: Optional[int],
    tag_type: str,
    annotation_error: str,
    date_warning: int,
    only_superclass: bool,
    config: Dict,
    full_tag: str,
) -> Tuple[str, bool]:
    if tag_type == MAIN_PHI_MAPPING["AGE_PHI_NAME"] and years_offset_ms is not None:
        try:
            new_age = (
                int(annotation.get_covered_text()) + int(years_offset_ms)
                if years_offset_ms != 0
                else annotation.get_covered_text()
            )
            if type(new_age) is int and new_age < 18:
                logger.warning(
                    annotation_error + " Detected age lower than 18. "
                    "Offset may be too small to deidentify!"
                )
            result = str(new_age), False
        except ValueError:
            # traceback.print_exc()
            logger.warning(
                annotation_error + " Detected non-numeric age. "
                "Age will be replaced with generic tag!"
            )
            result = f"{tag_type}_WRONG", True
    elif (
        annotation.entityType == PHI_MAPPING["BIRTHDATE_PHI_NAME"]
        and days_offset_ms is not None
    ):
        assert years_offset_ms is not None
        found_format = identify_date(annotation.get_covered_text())
        try:
            if found_format is None:
                raise ValueError(annotation_error)
            new_date = datetime.strptime(annotation.get_covered_text(), found_format)
            new_date = new_date.replace(
                year=int(new_date.year) + int(years_offset_ms)
            ) + timedelta(milliseconds=days_offset_ms)
            result = new_date.date().strftime(found_format), False
        except ValueError:
            # traceback.print_exc()
            logger.warning(
                annotation_error + " Date will be replaced with generic tag!"
            )
            result = f"{tag_type}_WRONG", True
    elif (
        annotation.entityType == PHI_MAPPING["DATE_PHI_NAME"]
        and days_offset_ms is not None
    ):
        date_value = (
            annotation.get_covered_text().strip().replace("(", "").replace(")", "")
        )
        try:
            if any(re.fullmatch(pattern, date_value) for pattern in NO_DEID_DATES):
                return annotation.get_covered_text(), False
            # Check if it is a time frame or a simple date
            timeframe_split = identify_timeframe_split(date_value)
            # Simple date, there is only one
            if timeframe_split == "":
                date_values = [date_value]
            else:
                # Split in multiple dates
                date_values = date_value.split(timeframe_split)
            date_values = [d for d in date_values if d.lower() not in DAYS_WEEK]
            new_dates = []
            for d in date_values:
                # Find the correct format
                found_format = identify_date(d)
                if found_format is None:
                    raise ValueError(f"No matching format found for date {d}")
                new_dates.append(
                    (
                        datetime.strptime(d, found_format)
                        + timedelta(milliseconds=days_offset_ms),
                        found_format,
                    )
                )
            new_date_formatted = timeframe_split.join(
                [
                    new_date.date().strftime(found_format)
                    for new_date, found_format in new_dates
                ]
            )
            if (
                any(1901 < new_date.year < date_warning for new_date, _ in new_dates)
                and config is not None
                and config["PHI"].get("DATE", {}).get("BIRTHDATE", None) == "replace"
            ):
                logger.warning(
                    annotation_error
                    + f"Beware, date with year before {date_warning} detected, "
                    f"possibly a birthdate! Date will be replaced with birthday tag!"
                )
                result = (
                    tag_type
                    if only_superclass
                    else f"{tag_type}_{PHI_MAPPING['BIRTHDATE_PHI_NAME']}",
                    True,
                )
            else:
                result = new_date_formatted, False
        except ValueError:
            # traceback.print_exc()
            logger.warning(
                annotation_error + " Date will be replaced with generic tag!"
            )
            result = f"{tag_type}_WRONG", True
    else:
        result = tag_type if only_superclass else full_tag, True
    return result


def deidentify_cas(
    cas: Cas,
    document_id: str,
    config: Dict,
    fake_name: Tuple[str, str] = None,
    days_offset_ms: int = None,
    years_offset_ms: int = None,
    date_warning: int = 2007,
    show_original_text: bool = False,
    only_superclass: bool = False,
) -> str:
    annotations = sorted(cas.select(NAMED_ENTITY_CAS), key=lambda d: (d.begin, d.end))  # type: ignore
    assert isinstance(cas.sofa_string, str)
    offset = 0
    deidentified_document = ""
    for annotation in annotations:
        substituted, is_tag = substitute_value(
            annotation=annotation,
            document_id=document_id,
            only_superclass=only_superclass,
            fake_name=fake_name,
            days_offset_ms=days_offset_ms,
            years_offset_ms=years_offset_ms,
            date_warning=date_warning,
            config=config,
        )
        deidentified_document += cas.sofa_string[offset : annotation.begin]
        if show_original_text:
            deidentified_document += f"<{substituted}:{annotation.get_covered_text()}>"
        elif is_tag:
            deidentified_document += f"<{substituted}>"
        else:
            deidentified_document += substituted
        offset = annotation.end
    deidentified_document += cas.sofa_string[offset:]

    return deidentified_document
