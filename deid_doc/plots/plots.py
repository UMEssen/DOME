import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

eval_dict = {
    "ent_type": "pos=overlap, type=same",
    "partial": "pos=overlap, type=ignored",
    "strict": "pos=same, type=same",
    "exact": "pos=same, type=ignored",
}


def main(opt: argparse.Namespace) -> None:
    plots = Path.cwd() / "results" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    new_data = []
    for modus in ["only_type", "with_kind"]:
        for eval_type in ["ent_type", "partial", "strict", "exact"]:
            for letterhead in ["without_LH", "with_LH"]:
                sheet_name = f"doc_{modus}_{letterhead}"
                df = pd.read_excel(opt.excel, sheet_name=sheet_name)
                # Filter
                df = df.loc[
                    (df["possible"] > 0) & (df["eval_type"] == eval_type),
                    ["precision", "recall", "f1"],
                ]
                df.rename(
                    {"precision": "Precision", "recall": "Recall", "f1": "F1-Score"},
                    inplace=True,
                    axis=1,
                )
                df["Methode"] = eval_dict[eval_type]
                if letterhead == "without_LH":
                    df["Briefkopf"] = "ohne Briefkopf"
                else:
                    df["Briefkopf"] = "mit Briefkopf"
                new_data.append(df)
        df = pd.concat(new_data)
        for label in ["F1-Score"]:  # ["Precision", "Recall", "F1-Score"]:
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.boxplot(
                ax=ax,
                x="Methode",
                y=label,
                hue="Briefkopf",
                data=df,
                palette="Set2",
            )
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=4)
            # sns.despine(trim=True, left=True)
            plt.savefig(
                plots / f"{opt.excel.name.split('.')[0]}_{modus}_{label}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--excel",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    main(args)
