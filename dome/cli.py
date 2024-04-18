import argparse
import logging
import random
import shutil
import tempfile
from pathlib import Path

import cassis
import yaml

from dome.ml.test import test
from dome.util.cas_handling import deidentify_cas

logger = logging.getLogger(__name__)


def main(arguments: argparse.Namespace) -> None:
    if arguments.keep_cas:
        cas_folder = arguments.output_folder / "cas"
        cas_folder.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning("The CAS files will not be available after deidentification!")
        cas_folder = Path(tempfile.mktemp(prefix="cas_"))
    dataset_info = {
        "DATASET_NAME": "test",
        "USE_GIVEN_OUTPUT": True,
        "OUTPUT_FOLDER": cas_folder,
        "INPUT_FOLDER": arguments.input_folder,
        "CHUNK_SIZE": 50,
        "OVERLAP_WORDS": 10,
        "LONGFORMER": 0,
        "INFERENCE": 1,
        "SKIP_LETTERHEAD": 0,
    }
    model_choice = {
        "superclass": 0,
        "subclass": 1,
        "both": 2,
    }
    test(
        dataset_info=dataset_info,
        model_choice=model_choice[arguments.model_choice],
        aggregation_strategy=arguments.aggregation_strategy,
        model_superclass_path=arguments.model_superclass_path
        if arguments.model_superclass_path.exists()
        else None,
        model_subclass_path=arguments.model_subclass_path
        if arguments.model_subclass_path.exists()
        else None,
    )

    assert (
        arguments.deid_config.exists()
    ), f"The deidentification {arguments.deid_config} config does not exist."

    with arguments.deid_config.open("r") as of:
        config = yaml.safe_load(of)

    assert (
        arguments.typesystem.exists()
    ), f"The type system {arguments.typesystem} does not exist."

    type_system = cassis.load_typesystem(arguments.typesystem)

    random.seed(42)
    allowed_values_date = list(
        range(
            int(config["OFFSET"]["DATE_NEGATIVE_BOUNDARY"]),
            int(config["OFFSET"]["DATE_POSITIVE_BOUNDARY"]),
        )
    )
    allowed_values_date.remove(0)
    allowed_values_year = list(
        range(
            int(config["OFFSET"]["AGE_NEGATIVE_BOUNDARY"]),
            int(config["OFFSET"]["AGE_POSITIVE_BOUNDARY"]),
        )
    )
    allowed_values_year.remove(0)
    ms_in_day = 24 * 60 * 60 * 1000

    for cas_path in sorted(cas_folder.glob("*.xmi")):
        cas = cassis.load_cas_from_xmi(cas_path, typesystem=type_system)
        offset_date = random.choice(allowed_values_date) * ms_in_day
        offset_year = random.choice(allowed_values_year)
        doc_id = cas_path.name.replace(".xmi", "")
        deidentified_document = deidentify_cas(
            cas=cas,
            document_id=doc_id,
            days_offset_ms=offset_date,
            years_offset_ms=offset_year,
            config=config,
        )

        with open(arguments.output_folder / f"{doc_id}.txt", "w") as f:
            f.write(deidentified_document)

    if not arguments.keep_cas:
        shutil.rmtree(cas_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="The folder where the output will be stored",
        required=True,
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        help="The folder containing the documents to be anonymized",
        required=True,
    )
    parser.add_argument(
        "--model-choice",
        type=str,
        choices=["superclass", "subclass", "both"],
        default="both",
        help="Which model to use for anonymization",
    )
    parser.add_argument(
        "--aggregation-strategy",
        type=str,
        choices=[
            "none",
            "simple",
            "first",
            "average",
            "max",
            "any_max",
            "any_average",
            "any_first",
            "exact",
        ],
        default="any_max",
        help="How to aggregate the predictions from different tokens",
    )
    parser.add_argument(
        "--model-superclass-path",
        type=Path,
        help="The path to the superclass model",
        default="/models/superclass_model_melanoma_patho_2023-04-05_13-56-05",
    )
    parser.add_argument(
        "--model-subclass-path",
        type=Path,
        help="The path to the subclass model",
        default="/models/subclass_model_melanoma_patho_2023-04-06_15-02-26",
    )
    parser.add_argument(
        "--deid-config",
        type=Path,
        help="The path to the deid config",
        default="/app/config/default-config.yaml",
    )
    parser.add_argument(
        "--typesystem",
        type=Path,
        help="The path to the deid config",
        default="/app/config/TypeSystem.xml",
    )
    parser.add_argument(
        "--keep-cas",
        default=False,
        action="store_true",
        help="Keep the CAS files after deidentification",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Print additional information for debugging purposes",
    )
    args = parser.parse_args()

    logging.basicConfig()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    main(args)
