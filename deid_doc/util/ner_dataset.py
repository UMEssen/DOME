import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cassis
import datasets

from deid_doc.util.cas_handling import generate_chunks
from deid_doc.util.util import (
    generate_pathology_cas_without_letterhead,
    valid_xml_char_ordinal,
)

logger = logging.getLogger(__name__)


class NERConfig(datasets.BuilderConfig):
    def __init__(
        self,
        input_folder: Path,
        labels: List[str],
        labels_subclass: Optional[List[str]],
        label_map: Dict[str, int],
        chunk_size: int,
        overlap_words: int,
        longformer: bool,
        only_inference: bool,
        skip_pathology_letterhead: bool,
        **kwargs: Any,
    ) -> None:
        super(NERConfig, self).__init__(**kwargs)
        self.input_folder = input_folder
        self.labels = labels
        self.labels_subclass = labels_subclass
        self.label_map = label_map
        self.chunk_size = chunk_size
        self.overlap_words = overlap_words
        self.longformer = longformer
        self.only_inference = only_inference
        self.skip_pathology_letterhead = skip_pathology_letterhead


class NERDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        NERConfig(
            name=os.environ["DATASET_NAME"] if "DATASET_NAME" in os.environ else "test",
            input_folder=Path.cwd(),
            labels=["O"],
            labels_subclass=None,
            label_map={"O": 0},
            chunk_size=100,
            overlap_words=10,
            longformer=False,
            only_inference=True,
            skip_pathology_letterhead=False,
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        feature_dict = {
            "document_id": datasets.Value("string"),
            "num_parts": datasets.Value("uint32"),
            "part_id": datasets.Value("uint32"),
            "part_start": datasets.Value("uint32"),
            "part_end": datasets.Value("uint32"),
            "text": datasets.Value("string"),
            "fulltext": datasets.Value("string"),
            "tokens": datasets.Sequence(feature=datasets.Value("string")),
        }
        if self.config.labels_subclass is not None:
            feature_dict["ner_tags_superclass"] = datasets.Sequence(
                datasets.ClassLabel(
                    names=self.config.labels,
                    num_classes=len(self.config.labels),
                )
            )
            feature_dict["ner_tags_subclass"] = datasets.Sequence(
                datasets.ClassLabel(
                    names=self.config.labels_subclass,
                    num_classes=len(self.config.labels_subclass),
                )
            )
        else:
            feature_dict["ner_tags"] = datasets.Sequence(
                datasets.ClassLabel(
                    names=self.config.labels,
                    num_classes=len(self.config.labels),
                )
            )
        return datasets.DatasetInfo(
            description="TBD",
            features=datasets.Features(feature_dict),
            supervised_keys=None,
            homepage="TBD",
            citation="TBD",
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        typesystem = cassis.load_typesystem(Path("config") / "TypeSystem.xml")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "documents": [
                        doc
                        for doc in (self.config.input_folder / "train").glob("*.xmi")
                    ],
                    "txt_documents": [
                        doc
                        for doc in (self.config.input_folder / "train").glob("*.txt")
                    ],
                    "type_system": typesystem,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "documents": [
                        doc for doc in (self.config.input_folder / "test").glob("*.xmi")
                    ]
                    + (
                        [doc for doc in self.config.input_folder.glob("*.xmi")]
                        if self.config.only_inference
                        else []
                    ),
                    "txt_documents": [
                        doc for doc in (self.config.input_folder / "test").glob("*.txt")
                    ]
                    + (
                        [doc for doc in self.config.input_folder.glob("*.txt")]
                        if self.config.only_inference
                        else []
                    ),
                    "type_system": typesystem,
                },
            ),
        ]

    def _generate_examples(
        self,
        documents: List[Path],
        txt_documents: List[Path],
        type_system: cassis.TypeSystem,
    ) -> Generator:
        logger.info(
            f"Generating splits for {len(documents + txt_documents)} documents."
        )
        for document_path in documents + txt_documents:
            document_name = document_path.name
            doc_name = document_name.replace(".xmi", "").replace(".txt", "")
            if document_name.endswith(".txt"):
                cas = cassis.Cas(typesystem=type_system)
                with document_path.open("r") as fp:
                    # Current workaround to filter non-xml characters in case they appear
                    cas.sofa_string = "".join(
                        c for c in fp.read() if valid_xml_char_ordinal(c)
                    )
                cas.sofa_mime = "text/plain"
            else:
                cas = cassis.load_cas_from_xmi(document_path, typesystem=type_system)
            if self.config.skip_pathology_letterhead:
                cas = generate_pathology_cas_without_letterhead(cas)
            token_chunks = list(
                generate_chunks(
                    cas,
                    ignore_ne=self.config.only_inference,
                    label_map=self.config.label_map,
                    chunk_size=self.config.chunk_size,
                    overlap_words=self.config.overlap_words,
                )
            )
            for p_id, chunk_list in enumerate(token_chunks):
                # For a range of chunks, get the beginning of the first chunk
                # and the end of the last chunk
                string_index_init = chunk_list[0]["begin"]
                string_index_end = chunk_list[-1]["end"]
                assert isinstance(string_index_init, int) and isinstance(
                    string_index_end, int
                )
                result_dict = {
                    "document_id": doc_name,
                    "num_parts": len(token_chunks),
                    "part_id": p_id,
                    "part_start": string_index_init,
                    "part_end": string_index_end,
                    "text": cas.sofa_string[string_index_init:string_index_end],
                    "fulltext": cas.sofa_string,
                    "tokens": [chunk["text"] for chunk in chunk_list],
                }
                if self.config.labels_subclass is not None:
                    result_dict["ner_tags_superclass"] = (
                        [chunk["superclass"] for chunk in chunk_list]
                        if not self.config.only_inference
                        else []
                    )
                    result_dict["ner_tags_subclass"] = (
                        [chunk["subclass"] for chunk in chunk_list]
                        if not self.config.only_inference
                        else []
                    )
                else:
                    result_dict["ner_tags"] = (
                        [chunk["subclass"] for chunk in chunk_list]
                        if not self.config.only_inference
                        else []
                    )
                yield f"{doc_name}_{string_index_init}_{string_index_end}", result_dict
