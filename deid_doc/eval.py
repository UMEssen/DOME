import argparse
import difflib
from pathlib import Path

import cassis
import pandas as pd

from deid_doc.util.constants import PHI_VALUES_SUBCLASS
from deid_doc.util.evaluation import get_documents, process_documents, strict_evaluation


def diff(a: str, b: str) -> str:
    sb = ""
    for s in difflib.unified_diff(a, b):
        if s != " ":
            sb += s
        # if s[0] == " ":
        #     continue
        # elif s[0] == "-":
        #     sb += 'Delete "{}" from position {}\n'.format(s[-1], i)
        # elif s[0] == "+":
        #     sb += 'Add "{}" to position {}\n'.format(s[-1], i)
    return sb


def main(opt: argparse.Namespace) -> None:
    output_folder = Path.cwd() / "results"
    output_folder.mkdir(parents=True, exist_ok=True)
    typesystem = cassis.load_typesystem(opt.typesystem)
    if opt.second_ts is not None:
        second_ts = cassis.load_typesystem(opt.second_ts)
    else:
        second_ts = None
    if opt.inception_curation is not None:
        inception_curation = opt.inception_curation
    else:
        inception_curation = False
    if opt.ignore_classes is not None:
        current_phi = [
            phi for phi in PHI_VALUES_SUBCLASS if phi not in opt.ignore_classes
        ]
    else:
        current_phi = PHI_VALUES_SUBCLASS
    with pd.ExcelWriter(
        output_folder / f"{opt.eval_name}_results.xlsx", engine="xlsxwriter"
    ) as excel:
        _, names, initial_gt, initial_pred = get_documents(
            typesystem=typesystem,
            ground_truth_folder=opt.ground_truth,
            prediction_folder=opt.prediction,
            inception=opt.inception,
            inception_user=opt.inception_user,
            source_folder=opt.source,
            second_ts=second_ts,
            inception_curation=inception_curation,
        )
        for blindness in [1, 2]:
            if blindness == 0:
                classes = ["REDACTED"]
                blind_name = "blind"
            elif blindness == 1:
                classes = sorted(set(p.split("_")[0] for p in current_phi))
                blind_name = "only_type"
            else:
                classes = current_phi
                blind_name = "with_kind"
            print(opt.eval_name, classes)
            gt, pred = process_documents(
                ground_truth=initial_gt,
                prediction=initial_pred,
                expected_phi=classes,
                blindness=blindness,
            )
            single_final = []
            for name, d1, d2 in zip(names, gt, pred):
                metric = strict_evaluation([d1], [d2])
                for new_result in metric.to_json():
                    new_result["document"] = name
                    single_final.append(new_result)
            pd.DataFrame(single_final).to_excel(
                excel,
                sheet_name=f"doc_{blind_name}",
                index=False,
            )
            metric_all = strict_evaluation(gt, pred)
            print(metric_all)
            pd.DataFrame(metric_all.to_json()).to_excel(
                excel,
                sheet_name=f"sum_{blind_name}",
                index=False,
            )


# RUN: poetry run python -m deid_doc.eval --original datasets/original-data --ground-truth datasets/regex-new+manual-final/positions --averbis datasets/averbis-results --type av1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--typesystem",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--second-ts",
        type=Path,
        required=False,
    )
    parser.add_argument("--inception-curation", type=bool, required=False)
    parser.add_argument("--inception", type=bool, required=False)
    parser.add_argument("--inception-user", type=str, required=False)
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--ignore-classes", nargs="+", type=str)

    args = parser.parse_args()
    main(args)
