import json
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from datasets import load_metric
from transformers import (
    DataCollatorForTokenClassification,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.tokenization_bert_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import EvalPrediction

from deid_doc.ml.util import (
    get_model_and_tokenizer_for_class,
    get_model_name_from_name,
    get_output_folder,
    prepare_dataset,
)


def tokenize_and_align_labels(
    samples: Dict,
    tokenizer: PreTrainedTokenizerFast,
    class_level: str,
    label_all_tokens: bool = False,
    **tokenizer_params: Any,
) -> BatchEncoding:
    tokenized_inputs = tokenizer(samples["tokens"], **tokenizer_params)
    labels = []
    for i, label in enumerate(samples[f"ner_tags_{class_level}"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None.
            # We set the specific_class to -100, so they are
            # automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the specific_class for the first token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the specific_class to either the current
            # specific_class or -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(
    predictions: np.ndarray, labels: np.ndarray, metric: Any, label_list: List[str]
) -> Dict[str, float]:
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l > -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l > -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    label_results = {
        f"{label}_{class_results}": float(results[label][class_results])
        for label in results
        if "overall" not in label
        for class_results in results[label]
    }
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        **label_results,
    }


def monitor(p: EvalPrediction, metric: Any, label_list: List[str]) -> Dict[str, float]:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    return compute_metrics(predictions, labels, metric, label_list)


def train(
    dataset_info: Dict[str, Any],
    training_info: Dict[str, Any],
    model_superclass_name: str,
    token_superclass_name: str,
    model_subclass_name: str,
    token_subclass_name: str,
) -> None:
    print(
        "CUDA Available",
        torch.cuda.is_available(),
    )
    if "LOCAL_RANK" in os.environ:
        print("Using local rank", os.environ["LOCAL_RANK"])
    model_name = get_model_name_from_name(
        model_superclass_name=model_superclass_name,
        model_subclass_name=model_subclass_name,
    )
    output_folder = get_output_folder(
        output_folder=dataset_info["OUTPUT_FOLDER"],
        model_name=model_name,
        dataset_name=dataset_info["DATASET_NAME"],
    )

    common_train_args = dict(
        # Checkpoints
        overwrite_output_dir=True,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        save_total_limit=5,
        # Training Params
        learning_rate=training_info["LEARNING_RATE"],
        fp16=torch.cuda.is_available(),
        per_device_train_batch_size=training_info["BATCH_SIZE"],
        per_device_eval_batch_size=training_info["BATCH_SIZE"],
        num_train_epochs=training_info["EPOCHS"],
        weight_decay=training_info["WEIGHT_DECAY"],
        # Logging
        report_to=["tensorboard"],
        logging_dir=str(output_folder / "logs"),  # TB logs
        logging_first_step=True,
        logging_strategy=IntervalStrategy.EPOCH,
        # Evaluation
        evaluation_strategy=IntervalStrategy.EPOCH,
        # Other
        push_to_hub=False,
    )
    # model.gradient_checkpointing_enable()
    metric = load_metric("seqeval")
    for class_type in ["superclass", "subclass"]:
        train_args = TrainingArguments(
            output_dir=str(output_folder / f"checkpoints_{class_type}"),
            **common_train_args,
        )
        with train_args.main_process_first(desc="Create Dataset"):
            dataset = prepare_dataset(dataset_info)

        model, tokenizer = get_model_and_tokenizer_for_class(
            dataset,
            model_name=model_superclass_name
            if class_type == "superclass"
            else model_subclass_name,
            token_name=token_superclass_name
            if class_type == "superclass"
            else token_subclass_name,
            ner_tags_name=f"ner_tags_{class_type}",
        )
        data_collator = DataCollatorForTokenClassification(tokenizer)

        with train_args.main_process_first(desc="Dataset Map Pre-Processing"):
            wandb.init(project="deid_doc-pmr")
            tokenized_datasets = dataset.map(
                partial(
                    tokenize_and_align_labels,
                    tokenizer=tokenizer,
                    label_all_tokens=True,
                    class_level=class_type,
                    **{"is_split_into_words": True},
                ),
                batched=True,
            )
            label_list = (
                dataset["train"].features[f"ner_tags_{class_type}"].feature.names
            )
        trainer = Trainer(
            model,
            train_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=partial(monitor, metric=metric, label_list=label_list),
        )
        train_output = trainer.train()
        # val_output = trainer.evaluate()
        with train_args.main_process_first(desc="Store Model and Run Prediction"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            model_folder = (
                output_folder
                / f"{class_type}_model_{dataset_info['DATASET_NAME']}_{timestamp}"
            )
            model_folder.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(model_folder))

            config = trainer.model.config
            config.name = model_name
            config.id2label = {i: val for i, val in enumerate(label_list)}
            config.label2id = {val: i for i, val in enumerate(label_list)}
            config.to_json_file(str(model_folder / "config.json"))

            with (model_folder / "train_metrics.json").open("w") as of:
                json.dump(train_output, of)
            # with (model_folder / "evaluate_metrics.json").open("w") as of:
            #     json.dump(val_output, of)
