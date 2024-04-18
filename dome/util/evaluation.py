"""Strict NER evaluation (Tjong Kim Sang, CoNLL 2002).

We count precision and recall as follows:
    - TP: the predicted entity matches with a gold entity both in type and offset
    - FP: the predicted entity does not exist in the set of gold entities
    - FN: the gold entity does not exist in the set of predicted entities
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import ZipFile

import cassis
from tqdm import tqdm

from dome.util.cas_handling import get_entities_for_evaluation
from dome.util.constants import (
    AGE_ENTITY,
    CONTACT_ENTITY,
    DATE_ENTITY,
    ID_ENTITY,
    LOCATION_ENTITY,
    NAME_ENTITY,
    NAMED_ENTITY_CAS,
    OTHER_ENTITY,
    PROFESSION_ENTITY,
)


@dataclass(eq=True, frozen=True)
class Entity:
    doc_id: int
    label: str
    start: int
    end: int


class Metric:
    """
    Helper class to tally tp/fp/fn per label and to compute evaluation metrics.

    Source: https://github.com/zalandoresearch/flair/blob/master/flair/training_utils.py
    """

    def __init__(self, name: str) -> None:
        self.name = name

        self._total_gt: Dict[str, int] = defaultdict(int)
        self._total_pred: Dict[str, int] = defaultdict(int)
        self._tps: Dict[str, int] = defaultdict(int)
        self._fps: Dict[str, int] = defaultdict(int)
        self._tns: Dict[str, int] = defaultdict(int)
        self._fns: Dict[str, int] = defaultdict(int)

    def add_tp(self, class_name: str, n: int = 1) -> None:
        self._tps[class_name] += n
        self._total_gt[class_name] += n
        self._total_pred[class_name] += n

    def add_tn(self, class_name: str, n: int = 1) -> None:
        self._tns[class_name] += n

    def add_fp(self, class_name: str, n: int = 1) -> None:
        self._fps[class_name] += n
        self._total_pred[class_name] += n

    def add_fn(self, class_name: str, n: int = 1) -> None:
        self._fns[class_name] += n
        self._total_gt[class_name] += n

    def get_tp(self, class_name: str = None) -> int:
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name: str = None) -> int:
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name: str = None) -> int:
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name: str = None) -> int:
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def get_total_gt(self, class_name: str = None) -> int:
        if class_name is None:
            return sum(
                [self._total_gt[class_name] for class_name in self.get_classes()]
            )
        return self._total_gt[class_name]

    def get_total_pred(self, class_name: str = None) -> int:
        if class_name is None:
            return sum(
                [self._total_pred[class_name] for class_name in self.get_classes()]
            )
        return self._total_pred[class_name]

    def precision(self, class_name: str = None) -> float:
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name: str = None) -> float:
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0

    def f_score(self, class_name: str = None) -> float:
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0

    def get_classes(self) -> List:
        class_names = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in class_names if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_json(self) -> List[Dict[str, Any]]:
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        return [
            {
                "eval_name": self.name if class_name is None else class_name,
                "tp": self.get_tp(class_name),
                "fp": self.get_fp(class_name),
                "fn": self.get_fn(class_name),
                "tn": self.get_tn(class_name),
                "total_gt": self.get_total_gt(class_name),
                "total_pred": self.get_total_pred(class_name),
                "precision": self.precision(class_name),
                "recall": self.recall(class_name),
                "f1": self.f_score(class_name),
            }
            for class_name in all_classes
        ]

    def __str__(self) -> str:
        results = self.to_json()
        all_lines = [
            f"{d['eval_name']:<25} "
            f"tp: {d['tp']:<5} fp: {d['fp']:<5} fn: {d['fn']:<5} tn: {d['tn']:<5} - "
            f"sum gt: {d['total_gt']:<7} sum pred: {d['total_pred']:<7} - "
            f"precision: {d['precision']:.3f} recall: {d['recall']:.3f} f1 {d['f1']:.3f}"
            for d in results
        ]
        return "\n".join(all_lines)


def strict_evaluation(
    ground_truth: List[List[Entity]], predictions: List[List[Entity]]
) -> Metric:
    metric = Metric("strict")

    gold = set(entity for doc in ground_truth for entity in doc)
    pred = set(entity for doc in predictions for entity in doc)

    for ent in pred:
        if ent in gold:
            metric.add_tp(class_name=ent.label)
        else:
            metric.add_fp(class_name=ent.label)

    for ent in gold:
        if ent not in pred:
            metric.add_fn(class_name=ent.label)

    return metric


def get_documents(
    ground_truth_folder: Path,
    prediction_folder: Path,
    typesystem: cassis.TypeSystem,
    inception: bool = False,
    inception_curation: bool = False,
    inception_user: str = "",
    source_folder: Path = None,
    second_ts: cassis.TypeSystem = None,
) -> Tuple[
    List[str],
    List[str],
    List[List[cassis.typesystem.Any]],
    List[List[cassis.typesystem.Any]],
]:
    all_files = list(prediction_folder.glob("*"))
    document_text = []
    names = []
    ground_truth_documents = []
    prediction_documents = []
    ner_list = [
        AGE_ENTITY,
        CONTACT_ENTITY,
        DATE_ENTITY,
        ID_ENTITY,
        LOCATION_ENTITY,
        NAME_ENTITY,
        PROFESSION_ENTITY,
        OTHER_ENTITY,
    ]
    for document_path in tqdm(all_files, total=len(all_files)):
        document_name = document_path.name.split(".")[0]
        if inception:
            if inception_curation:
                for zip_file in (ground_truth_folder / (document_name + ".xmi")).glob(
                    "*"
                ):
                    with ZipFile(zip_file, "r") as zip:
                        for info in zip.infolist():
                            if info.filename == "CURATION_USER.xmi":
                                ground_truth = cassis.load_cas_from_xmi(
                                    zip.read(info.filename).decode("utf-8"), typesystem
                                )
                gt_entities = list()
                for entity in ground_truth.select_all():
                    if entity.type.name in ner_list:
                        gt_entities.append(entity)
                ground_truth_documents.append(
                    sorted(gt_entities, key=lambda x: (x.begin, x.end))
                )
            else:
                ground_truth = cassis.load_cas_from_xmi(
                    ground_truth_folder / (document_name + ".xmi"), second_ts
                )
                ground_truth_documents.append(
                    sorted(
                        ground_truth.select(NAMED_ENTITY_CAS),
                        key=lambda x: (x.begin, x.end),
                    )
                )
            if not source_folder:
                for zip_file in document_path.glob("*"):
                    with ZipFile(zip_file, "r") as zip:
                        for info in zip.infolist():
                            if info.filename == inception_user + ".xmi":
                                prediction = cassis.load_cas_from_xmi(
                                    zip.read(info.filename).decode("utf-8"), typesystem
                                )
                pred_entities = list()
                for entity in prediction.select_all():
                    if entity.type.name in ner_list:
                        pred_entities.append(entity)
                prediction_documents.append(
                    sorted(pred_entities, key=lambda x: (x.begin, x.end))
                )
            else:
                prediction = cassis.load_cas_from_xmi(
                    source_folder / (document_name + ".xmi"), second_ts
                )
                prediction_documents.append(
                    sorted(
                        prediction.select(NAMED_ENTITY_CAS),
                        key=lambda x: (x.begin, x.end),
                    )
                )
        else:
            ground_truth = cassis.load_cas_from_xmi(
                ground_truth_folder / (document_name + ".xmi"), typesystem
            )
            prediction = cassis.load_cas_from_xmi(
                prediction_folder / (document_name + ".xmi"), typesystem
            )
            ground_truth_documents.append(
                sorted(
                    ground_truth.select(NAMED_ENTITY_CAS),
                    key=lambda x: (x.begin, x.end),
                )
            )
            prediction_documents.append(
                sorted(
                    prediction.select(NAMED_ENTITY_CAS), key=lambda x: (x.begin, x.end)
                )
            )
        assert ground_truth.sofa_string == prediction.sofa_string, document_path.name
        document_text.append(ground_truth.sofa_string)
        # The entities need sorting. In nerevaluate the true entities are only considered once for
        # comparison, so if we have something as follows
        # True: Dr. Abc: PER, Atlanta: LOC
        # Predicted: Dr. Abc/Atlanta: PER
        # Then the error is only counted once (for example, only for Dr. Abc), which means that it
        # may depend on the order of the annotations
        # So we need to sort them to ensure that the ordering stays the same

        names.append(document_name)
    return document_text, names, ground_truth_documents, prediction_documents


def process_documents(
    ground_truth: List[List[cassis.typesystem.Any]],
    prediction: List[List[cassis.typesystem.Any]],
    expected_phi: List[str],
    blindness: int = 2,
) -> Tuple[List[List[Entity]], List[List[Entity]]]:
    ground_truth_documents = []
    prediction_documents = []
    for document_id, (g, p) in enumerate(zip(ground_truth, prediction)):
        # if g:
        #    print(g)
        # if p:
        #    print(p)
        ground_truth_documents.append(
            [
                Entity(doc_id=document_id, **annotation)
                for annotation in get_entities_for_evaluation(
                    g,
                    blindness=blindness,
                )
                if annotation["label"] in expected_phi
            ]
        )
        prediction_documents.append(
            [
                Entity(doc_id=document_id, **annotation)
                for annotation in get_entities_for_evaluation(
                    p,
                    blindness=blindness,
                )
                if annotation["label"] in expected_phi
            ]
        )
    return ground_truth_documents, prediction_documents
