import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from dome.util.constants import PHI_VALUES_SUBCLASS, PHI_VALUES_SUPERCLASS

logger = logging.getLogger(__name__)


def get_class_labels(phi_values: List[str]) -> List[str]:
    tmp = [[add + val for add in ["B-", "I-"]] for val in sorted(phi_values)]
    labels = [item for sublist in tmp for item in sublist]
    labels.insert(0, "O")
    return labels


def prepare_dataset(dataset_info: Dict[str, Any]) -> datasets.DatasetDict:
    labels_superclass = get_class_labels(PHI_VALUES_SUPERCLASS)
    labels_subclass = get_class_labels(PHI_VALUES_SUBCLASS)
    label_map = {
        **{val[2:]: i for i, val in enumerate(labels_superclass) if val[0] == "B"},
        **{val[2:]: i for i, val in enumerate(labels_subclass) if val[0] == "B"},
    }

    logger.info(f"Superclass {labels_superclass}")
    logger.info(f"Subclass {labels_subclass}")
    logger.info(f"Mapping {label_map}")

    # TODO: I have put it here because I still have not found a way to specify the dataset config
    #  name later on
    from dome.util.ner_dataset import NERDataset

    ds = NERDataset(
        name=dataset_info["DATASET_NAME"],
        data_dir=dataset_info["OUTPUT_FOLDER"],
        input_folder=dataset_info["INPUT_FOLDER"],
        labels=labels_superclass,
        labels_subclass=labels_subclass,
        label_map=label_map,
        chunk_size=dataset_info["CHUNK_SIZE"],
        overlap_words=dataset_info["OVERLAP_WORDS"],
        longformer=dataset_info["LONGFORMER"],
        only_inference=dataset_info["INFERENCE"],
        skip_pathology_letterhead=dataset_info["SKIP_LETTERHEAD"],
    )
    ds.download_and_prepare()
    data = ds.as_dataset()
    train_dataset = data["train"]
    test_dataset = data["test"]
    logger.info(f"train {len(train_dataset)}, test {len(test_dataset)}")
    return datasets.DatasetDict({"train": train_dataset, "test": test_dataset})


def get_model_and_tokenizer_for_class(
    dataset: datasets.DatasetDict,
    model_name: str,
    token_name: str,
    ner_tags_name: str,
) -> Tuple[AutoModelForTokenClassification, PreTrainedTokenizerFast]:
    label_list = dataset["train"].features[ner_tags_name].feature.names
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        ignore_mismatched_sizes=True,
    ), AutoTokenizer.from_pretrained(
        token_name,
        num_labels=len(label_list),
        ignore_mismatched_sizes=True,
        add_prefix_space=True,
    )


def get_models_from_path(
    chosen_model: Path,
) -> Tuple[Tuple[AutoModelForTokenClassification, PreTrainedTokenizerFast], str]:
    logger.info(f"The chosen model is {chosen_model}")
    timestamp = "_".join(chosen_model.name.split("_")[-2:])
    return (
        AutoModelForTokenClassification.from_pretrained(
            chosen_model, local_files_only=True
        ),
        AutoTokenizer.from_pretrained(chosen_model, local_files_only=True),
    ), timestamp


def get_model_name_from_path(
    model_superclass_path: Optional[Path],
    model_subclass_path: Optional[Path],
) -> Tuple[str, str]:
    model_super, model_sub = "", ""
    if model_superclass_path is not None:
        with (model_superclass_path / "config.json").open("r") as fp:
            model_super = json.load(fp)["_name_or_path"].split("/")[-1]
    if model_subclass_path is not None:
        with (model_subclass_path / "config.json").open("r") as fp:
            model_sub = json.load(fp)["_name_or_path"].split("/")[-1]
    return model_super, model_sub


def get_model_name_from_name(
    model_superclass_name: str,
    model_subclass_name: str,
) -> str:
    return (
        f"{model_superclass_name.split('/')[-1]}"
        f"_{model_subclass_name.split('/')[-1]}"
    )


def get_output_folder(output_folder: Path, model_name: str, dataset_name: str) -> Path:
    return output_folder / model_name / dataset_name
