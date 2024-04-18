import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cassis
import pandas as pd
import yaml

from dome.pre_processing.current_data_splits import (
    convert_json_to_cas,
    create_annotations_from_cas,
)
from dome.util.cas_handling import deidentify_cas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(str(args.config.absolute())))
    input_format = config["INPUT"]["FORMAT"]
    input_folder = Path(config["INPUT"]["PATH"])
    output_folder = Path(config["RESULTS"]["TXT_OUTPUT_PATH"])
    output_folder.mkdir(parents=True, exist_ok=True)
    type_system = cassis.load_typesystem(Path(config["INPUT"]["TS"]))
    if config["OFFSET"].get("TABLE_PATH", None) is not None:
        time_offset_df = pd.read_csv(config["OFFSET"]["TABLE_PATH"], sep=",")
    else:
        time_offset_df = None
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

    if "averbis" in input_format:
        annotations = convert_json_to_cas(
            type_system=type_system,
            melanom_batches=[
                json.load(f.open("r")) for f in input_folder.glob("*json")
            ],
            resource_type=config["INPUT"]["RESOURCE_TYPE"],
        )
    else:
        annotations = create_annotations_from_cas(
            type_system=type_system,
            cas_folder=input_folder,
            resource_type=config["INPUT"]["RESOURCE_TYPE"],
        )
    print(datetime.now())
    print("Token Model: ", config["NAMES"]["TOKENIZER"])
    print("Model: ", config["NAMES"]["MODEL"])
    print("Dataset: ", config["NAMES"]["DATASET"])
    print("")
    print("")

    for patient_id in annotations:
        if time_offset_df is not None:
            offset_date = int(
                time_offset_df.loc[
                    time_offset_df["patientid"] == patient_id, "timeoffset"
                ].values[0]
            )
            offset_year = 0
        else:
            offset_date = random.choice(allowed_values_date) * ms_in_day
            offset_year = random.choice(allowed_values_year)
        if config["RESULTS"].get("JSON_INPUT_PATH") is not None:
            obs_json_path = config["RESULTS"]["JSON_INPUT_PATH"]
            with open(obs_json_path + patient_id + ".json", "r") as f:
                bundle = json.load(f)
                for entry, annotation in zip(
                    sorted(bundle.get("entry"), key=lambda x: x["resource"]["id"]),
                    sorted(list(annotations[patient_id]), key=lambda x: x[0]),
                ):
                    resource = entry.get("resource")
                    if resource.get("id") == annotation[0]:
                        deidentified_document = deidentify_cas(
                            cas=annotation[1],
                            document_id=annotation[0],
                            days_offset_ms=offset_date,
                            years_offset_ms=offset_year,
                            config=config,
                        )
                        resource["valueString"] = deidentified_document
                        with open(output_folder / f"{annotation[0]}.txt", "w") as f:
                            f.write(deidentified_document)
                    else:
                        raise ValueError(
                            "JSON Bundles and Annotations are not identical. Bundles cannot be de-identified!"
                        )
                with open(
                    config["RESULTS"]["JSON_OUTPUT_PATH"] + patient_id + ".json",
                    "w",
                    encoding="utf-8",
                ) as json_file:
                    json.dump(bundle, json_file, ensure_ascii=False)
        else:
            for obs_id, cas in annotations[patient_id]:
                deidentified_document = deidentify_cas(
                    cas=cas,
                    document_id=obs_id,
                    days_offset_ms=offset_date,
                    years_offset_ms=offset_year,
                    config=config,
                )
                with open(output_folder / f"{obs_id}.txt", "w") as f:
                    f.write(deidentified_document)
