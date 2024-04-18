import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fhir_pyrate import Ahoy, Pirate


def main() -> None:
    split_document = Path(os.environ["INPUT_FILE"])
    df = pd.read_csv(split_document)
    # Filter rows where 'document_type' is 'patho' and 'phase' is 'train'
    patho_train = df[(df["document_type"] == "patho") & (df["phase"] == "train")]
    # Filter rows where 'document_type' is 'melanoma' and 'phase' is 'test'
    melanoma_test = df[(df["document_type"] == "melanoma") & (df["phase"] == "test")]
    # Find out pids for fhir_ids in melanoma testset
    auth = Ahoy(
        username=os.environ["FHIR_USER"],
        auth_type="token",
        auth_method="password",
        auth_url=os.environ["BASIC_AUTH"],  # The URL for authentication
        refresh_url=os.environ["REFRESH_URL"],
    )

    search = Pirate(auth=auth, base_url=os.environ["BASE_URL"], print_request_url=True)

    pid_df = search.trade_rows_for_dataframe(
        melanoma_test, resource_type="Patient", df_constraints={"_id": "patient_id"}
    )
    pid_df
    melanoma_test.rename(columns={"patient_id": "id"}, inplace=True)
    merged_df = melanoma_test.merge(
        pid_df[["id", "identifier_0_value"]], on="id", how="left"
    )
    merged_df.rename(columns={"identifier_0_value": "patient_id"}, inplace=True)
    # Find the intersection of unique values in 'patient_id'
    common_values = set(patho_train["patient_id"].unique()) & set(
        merged_df["patient_id"].unique()
    )
    df["is_repeating"] = df["patient_id"].isin(common_values)
    df.to_csv(os.environ["OUTPUT_FILE"], sep=",")


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
