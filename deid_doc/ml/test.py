import logging
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cassis
import numpy as np
import torch
from cassis import Cas
from tqdm import tqdm

from deid_doc._version import __githash__, __gitversion__, __version__
from deid_doc.ml.util import (
    get_model_name_from_path,
    get_models_from_path,
    get_output_folder,
    prepare_dataset,
)
from deid_doc.util.cas_handling import add_label_without_clashes
from deid_doc.util.constants import DOCUMENT_TYPE, MAIN_PHI_MAPPING, PHI_MAPPING
from deid_doc.util.regex_collection import IMAGE_REGEX, POSTPROCESS_REGEX
from deid_doc.util.token_classification import CustomTokenClassificationPipeline
from deid_doc.util.util import check_intersection, get_study_name

logger = logging.getLogger(__name__)


def entity_matching(
    superclass_result: List[Dict[str, Any]], subclass_result: List[Dict[str, Any]]
) -> Generator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]], None, None]:
    superclass_result.sort(key=lambda x: x["start"])  # type: ignore
    subclass_result.sort(key=lambda x: x["start"])  # type: ignore
    found_super = np.zeros(len(superclass_result), dtype=np.uint8)
    found_sub = np.zeros(len(subclass_result), dtype=np.uint8)
    for i, superclass in enumerate(superclass_result):
        for j, subclass in enumerate(subclass_result):
            # If the start of the subclass is already bigger than then end of the superclass
            # then we can already move to the next iteration
            if subclass["start"] > superclass["end"]:
                break
            if check_intersection(
                (superclass["start"], superclass["end"]),
                (subclass["start"], subclass["end"]),
            ):
                yield superclass, subclass
                found_super[i] += 1
                found_sub[j] += 1
    for i in np.where(found_super == 0)[0]:
        yield superclass_result[i], None

    for j in np.where(found_sub == 0)[0]:
        yield None, subclass_result[j]


def get_labels(
    sample: Dict[str, Any], value: Optional[Dict[str, Any]]
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    if value is None:
        return None, None, None, None
    start = sample["part_start"] + value["start"]
    end = sample["part_start"] + value["end"]
    if "_" in value["entity_group"]:
        superclass, subclass = value["entity_group"].split("_")
    else:
        superclass = value["entity_group"]
        subclass = ""
    return start, end, superclass, subclass


def add_regex_annotations(cas: cassis.Cas, text: str, only_superclass: bool) -> None:
    for regex, general_label, label, group in POSTPROCESS_REGEX:
        for match in re.finditer(string=text, pattern=regex, flags=re.I | re.M) or []:
            start, end = match.span(group)
            if label == PHI_MAPPING["STUDY_PHI_NAME"]:
                maybe_start, maybe_end = get_study_name(text, start, end)
                if maybe_start is None or maybe_end is None:
                    continue
            add_label_without_clashes(
                cas=cas,
                main_class=general_label,
                specific_class=label if not only_superclass else "",
                start=start,
                end=end,
                probability=1.0,
                always_update=False,
                source="RegEx-post",
            )


def process_annotations(
    annotations: List[Dict[str, Any]], sample: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = sample["fulltext"]
    prefixes = ["Ehefrau", "Ehemann", "Herrn", "Herr", "Frau", "Dr", "Prog"]
    annotations = [
        annotation
        for annotation in annotations
        # Check if the ima regex is present
        if re.search(
            pattern=IMAGE_REGEX,
            string=text[
                sample["part_start"]
                + annotation["start"]
                - 6 : sample["part_start"]
                + annotation["end"]
            ],
            flags=re.I | re.M,
        )
        is None
        # and ignore the annotations that only contain prefixes
        or not any(
            text[
                sample["part_start"]
                + annotation["start"] : sample["part_start"]
                + annotation["end"]
            ]
            == s
            for s in prefixes
        )
    ]
    prefix_offset = -10
    for annotation in annotations:
        if "Name" not in annotation["entity_group"]:
            continue
        start = sample["part_start"] + annotation["start"]
        end = sample["part_start"] + annotation["end"]
        match = re.search(
            pattern=rf"({'|'.join(prefixes)})\s{re.escape(text[start:end])}",
            string=text[start + prefix_offset : end],
            flags=re.I | re.M,
        )
        if match is not None:
            annotation["start"] += prefix_offset + match.start()
            # print(match, text[sample["part_start"] + annotation["start"] : end])
    return annotations


def test(
    dataset_info: Dict[str, Any],
    model_choice: int,
    aggregation_strategy: str,
    model_superclass_path: Path = None,
    model_subclass_path: Path = None,
) -> None:
    dataset = prepare_dataset(dataset_info)
    test_dataset = dataset["test"]
    # test_dataset = [test_dataset[5]]

    typesystem = cassis.load_typesystem(Path("config") / "TypeSystem.xml")
    # MODEL_CHOICE
    # 0 = only supermodel
    # 1 = only submodel
    # 2 = both models
    timestamp_super, timestamp_sub = None, None
    nlp_sub: Optional[CustomTokenClassificationPipeline] = None
    nlp_super: Optional[CustomTokenClassificationPipeline] = None
    if model_choice > 0 and model_subclass_path is not None:
        (model_sub, tokenizer_sub), timestamp_sub = get_models_from_path(
            model_subclass_path
        )
        nlp_sub = CustomTokenClassificationPipeline(
            model=model_sub,
            tokenizer=tokenizer_sub,
            aggregation_strategy=aggregation_strategy,
            device=0 if torch.cuda.is_available() else -1,
        )

    if model_choice % 2 == 0 and model_superclass_path is not None:
        (
            model_super,
            tokenizer_super,
        ), timestamp_super = get_models_from_path(model_superclass_path)
        nlp_super = CustomTokenClassificationPipeline(
            model=model_super,
            tokenizer=tokenizer_super,
            aggregation_strategy=aggregation_strategy,
            device=0 if torch.cuda.is_available() else -1,
        )

    if nlp_super is None:
        additional_name = "subclass"
        timestamp = timestamp_sub
    elif nlp_sub is None:
        additional_name = "superclass"
        timestamp = timestamp_super
    else:
        additional_name = "both"
        timestamp = f"{timestamp_super}_{timestamp_sub}"

    model_super_name, model_sub_name = get_model_name_from_path(
        model_superclass_path=model_superclass_path,
        model_subclass_path=model_subclass_path,
    )
    if len(model_super_name) > 0:
        if len(model_sub_name) > 0:
            model_name = f"{model_super_name}_{model_sub_name}"
        else:
            model_name = model_super_name
    else:
        model_name = model_sub_name

    output_folder = (
        dataset_info["OUTPUT_FOLDER"]
        if dataset_info.get("USE_GIVEN_OUTPUT")
        else (
            get_output_folder(
                output_folder=dataset_info["OUTPUT_FOLDER"],
                model_name=model_name,
                dataset_name=dataset_info["DATASET_NAME"],
            )
            / f"results_{dataset_info['DATASET_NAME']}_{timestamp}_{additional_name}"
        )
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    version_info = dict(
        # TODO: Fix it in __init__
        version=__version__,
        gitVersion=__gitversion__,
        gitHash=__githash__,
        supermodelName=model_super_name,
        supermodelPath=model_superclass_path.name
        if model_superclass_path is not None
        else None,
        submodelName=model_sub_name,
        submodelPath=model_subclass_path.name
        if model_subclass_path is not None
        else None,
    )

    currents_files: Dict[str, Any] = {}
    for sample in tqdm(test_dataset, total=len(test_dataset)):
        filename = sample["document_id"]
        if filename in currents_files:
            currents_files[filename]["parts"] += 1
        else:
            currents_files[filename] = {
                "cas": Cas(typesystem=typesystem),
                "parts": 1,
            }
        if currents_files[filename]["cas"].sofa_string is None:
            currents_files[filename]["cas"].sofa_string = sample["fulltext"]
            currents_files[filename]["cas"].sofa_mime = "text/plain"

        annotations = None
        if nlp_sub is None and nlp_super is not None:
            annotations = process_annotations(
                nlp_super(sample["text"]),
                sample,
            )
        elif nlp_super is None and nlp_sub is not None:
            annotations = process_annotations(
                nlp_sub(sample["text"]),
                sample,
            )

        # for annotation in annotations:
        #     print(annotation)
        # continue
        if annotations is not None:
            for annotation in annotations:
                start, end, superclass, subclass = get_labels(sample, annotation)
                assert (
                    start is not None
                    and end is not None
                    and superclass is not None
                    and subclass is not None
                )
                add_label_without_clashes(
                    cas=currents_files[filename]["cas"],
                    main_class=superclass,
                    specific_class=subclass,
                    start=start,
                    end=end,
                    probability=float(annotation["score"]),
                    always_update=False,
                    source=additional_name + "-model",
                )
        else:
            assert nlp_sub is not None and nlp_super is not None
            annotations_super = process_annotations(
                nlp_super(sample["text"]),
                sample,
            )
            annotations_sub = process_annotations(
                nlp_sub(sample["text"]),
                sample,
            )
            # Match the entities from the sub and super class predictions
            # if the entities don't match (e.g. "Müller" is predicted for super but not for sub)
            # then None is returned (e.g. (Profession, None))
            for sup, sub in entity_matching(annotations_super, annotations_sub):
                start_sup, end_sup, superclass, _ = get_labels(sample, sup)
                start_sub, end_sub, superclass_from_sub, subclass = get_labels(
                    sample, sub
                )
                # Both models have a prediction at the same spot and they have the same superclass
                if (
                    sup is not None
                    and sub is not None
                    and superclass == superclass_from_sub
                ):
                    assert (
                        isinstance(superclass, str)
                        and isinstance(subclass, str)
                        and isinstance(start_sup, int)
                        and isinstance(start_sub, int)
                        and isinstance(end_sup, int)
                        and isinstance(end_sub, int)
                    )
                    add_label_without_clashes(
                        cas=currents_files[filename]["cas"],
                        main_class=superclass,
                        specific_class=subclass,
                        start=min(start_sup, start_sub),
                        end=max(end_sup, end_sub),
                        probability=float(sub["score"]),
                        always_update=False,
                        source="both-models-agree",
                    )
                else:
                    # Either the superclass is not the same, or one of the results is None
                    logger.debug(
                        "\n"
                        f"f{filename}\n"
                        f"Superclass: {sup['entity_group'] if sup is not None else None} "
                        f"({sup['score'] if sup is not None else None}): "
                        f"{sample['fulltext'][start_sup:end_sup] if sup is not None else None}\n"
                        f"Subclass: {sub['entity_group'] if sub is not None else None} "
                        f"({sub['score'] if sub is not None else None}): "
                        f"{sample['fulltext'][start_sub:end_sub] if sub is not None else None}"
                    )
                    # If only the subclass model is none, then we take the prediction from the
                    # superclass
                    if sup is not None and sub is None:
                        assert (
                            isinstance(superclass, str)
                            and isinstance(start_sup, int)
                            and isinstance(end_sup, int)
                        )
                        add_label_without_clashes(
                            cas=currents_files[filename]["cas"],
                            main_class=superclass,
                            specific_class="",
                            start=start_sup,
                            end=end_sup,
                            probability=float(sup["score"]),
                            always_update=False,
                            source="supermodel-vs-submodel",
                        )
                    else:
                        # Either:
                        # The models have different predictions
                        # The submodel predicted something, but the super model did not (None)
                        add_label_without_clashes(
                            cas=currents_files[filename]["cas"],
                            main_class=MAIN_PHI_MAPPING["OTHER_PHI_NAME"],
                            specific_class="",
                            start=min(
                                start_sup
                                if start_sup is not None
                                else len(sample["fulltext"]),
                                start_sub
                                if start_sub is not None
                                else len(sample["fulltext"]),
                            ),
                            end=max(
                                end_sup if end_sup is not None else 0,
                                end_sub if end_sub is not None else 0,
                            ),
                            probability=0.0,
                            always_update=False,
                            source="submodel-vs-supermodel",
                        )

        if currents_files[filename]["parts"] == sample["num_parts"]:
            cas = currents_files[filename]["cas"]
            add_regex_annotations(cas, cas.sofa_string, only_superclass=nlp_sub is None)
            cas.add(typesystem.get_type(f"{DOCUMENT_TYPE}.VersionInfo")(**version_info))
            cas.to_xmi(output_folder / (filename + ".xmi"))
            del currents_files[filename]
    typesystem.to_xml(output_folder / "TypeSystem.xml")
    logger.debug(currents_files)
