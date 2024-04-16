import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from deid_doc.ml.test import test
from deid_doc.ml.train import train


def main() -> None:
    dataset_info = {
        "DATASET_NAME": os.environ["DATASET_NAME"],
        "OUTPUT_FOLDER": Path(os.environ["OUTPUT_FOLDER"]),
        "INPUT_FOLDER": Path(os.environ["INPUT_FOLDER"]),
        "CHUNK_SIZE": int(os.environ["CHUNK_SIZE"]),
        "OVERLAP_WORDS": int(os.environ["OVERLAP_WORDS"]),
        "LONGFORMER": int(os.environ["LONGFORMER"]),
        "INFERENCE": int(os.environ["INFERENCE"]),
        "SKIP_LETTERHEAD": int(os.environ["SKIP_LETTERHEAD"]),
    }
    if os.environ["PHASE"] == "train":
        training_info = {
            "LEARNING_RATE": float(os.environ["LEARNING_RATE"]),
            "BATCH_SIZE": int(os.environ["BATCH_SIZE"]),
            "EPOCHS": int(os.environ["EPOCHS"]),
            "WEIGHT_DECAY": float(os.environ["WEIGHT_DECAY"]),
        }
        train(
            dataset_info=dataset_info,
            training_info=training_info,
            model_superclass_name=os.environ["MODEL_NAME_SUPERCLASS"],
            token_superclass_name=os.environ["TOKEN_MODEL_NAME_SUPERCLASS"],
            model_subclass_name=os.environ["MODEL_NAME_SUBCLASS"],
            token_subclass_name=os.environ["TOKEN_MODEL_NAME_SUBCLASS"],
        )
    else:
        test(
            dataset_info=dataset_info,
            model_choice=int(os.environ["MODEL_CHOICE"]),
            aggregation_strategy=os.environ["AGGREGATION_STRATEGY"],
            model_superclass_path=Path(os.environ["MODEL_SUPERCLASS_PATH"])
            if "MODEL_SUPERCLASS_PATH" in os.environ
            else None,
            model_subclass_path=Path(os.environ["MODEL_SUBCLASS_PATH"])
            if "MODEL_SUBCLASS_PATH" in os.environ
            else None,
        )


if __name__ == "__main__":
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
    main()
