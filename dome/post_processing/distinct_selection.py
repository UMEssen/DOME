import argparse
from pathlib import Path

import pandas as pd
from cassis import load_cas_from_xmi, load_typesystem
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main(opt: argparse.Namespace) -> None:
    patho_documents = []
    melanoma_documents = []
    input_folder = opt.input_folder
    with open(str(input_folder / "TypeSystem.xml"), "rb") as f:
        typesystem = load_typesystem(f)
    csv = pd.read_csv(input_folder / "documents.csv")
    for elem in csv.itertuples():
        if elem.phase == "test" and elem.document_type == "patho":
            with Path(str(input_folder) + "/test/" + elem.document_id + ".xmi").open(
                "rb"
            ) as c:
                cas = load_cas_from_xmi(c, typesystem=typesystem)
                patho_documents.append(
                    dict(text=cas.sofa_string, document=elem.document_id)
                )
        elif elem.phase == "test" and elem.document_type == "melanoma":
            with Path(str(input_folder) + "/test/" + elem.document_id + ".xmi").open(
                "rb"
            ) as c:
                cas = load_cas_from_xmi(c, typesystem=typesystem)
                melanoma_documents.append(
                    dict(text=cas.sofa_string, document=elem.document_id)
                )
    # Extract the text and ID elements from each tuple in melanoma_documents
    melanoma_document_texts = []
    melanoma_document_ids = []
    for doc in melanoma_documents:
        if (
            len(doc["text"]) >= 1500
        ):  # Filter out documents with less than 1000 characters
            melanoma_document_texts.append(doc["text"])
            melanoma_document_ids.append(doc["document"])

    # Convert the list of document texts to a matrix of token counts
    melanoma_vectorizer = CountVectorizer(token_pattern=r"[^\n]+")
    melanoma_count_matrix = melanoma_vectorizer.fit_transform(melanoma_document_texts)

    # Compute pairwise cosine similarities between documents
    melanoma_similarity_matrix = cosine_similarity(melanoma_count_matrix)

    # Compute pairwise distances between documents based on cosine similarity
    melanoma_distance_matrix = 1 - melanoma_similarity_matrix

    # Compute the 50 most distinct documents
    n_clusters = 50
    melanoma_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="average"
    ).fit(melanoma_distance_matrix)
    melanoma_representative_ids = []
    for i in range(n_clusters):
        indices = [
            idx for idx, label in enumerate(melanoma_clustering.labels_) if label == i
        ]
        distances_to_others = [
            sum(
                melanoma_distance_matrix[idx, other_idx]
                for other_idx in indices
                if other_idx != idx
            )
            for idx in indices
        ]
        representative_idx = indices[
            distances_to_others.index(max(distances_to_others))
        ]
        melanoma_representative_ids.append(melanoma_document_ids[representative_idx])
    melanoma_representative_docs = [
        doc
        for doc in melanoma_documents
        if doc["document"] in melanoma_representative_ids
    ]
    output_folder = opt.output_folder
    for _index, dictionary in enumerate(melanoma_representative_docs):
        with open(
            str(output_folder) + "/melanom/" + dictionary["document"], "w"
        ) as file:
            file.write(dictionary["text"])

    patho_document_texts = []
    patho_document_ids = []
    for doc in patho_documents:
        if (
            len(doc["text"]) >= 1500
        ):  # Filter out documents with less than 1000 characters
            patho_document_texts.append(doc["text"])
            patho_document_ids.append(doc["document"])

    patho_vectorizer = CountVectorizer(token_pattern=r"[^\n]+")
    patho_count_matrix = patho_vectorizer.fit_transform(patho_document_texts)

    patho_similarity_matrix = cosine_similarity(patho_count_matrix)
    patho_distance_matrix = 1 - patho_similarity_matrix

    # Compute the 50 most distinct documents
    n_clusters = 50
    patho_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="average"
    ).fit(patho_distance_matrix)
    patho_representative_ids = []
    for i in range(n_clusters):
        indices = [
            idx for idx, label in enumerate(patho_clustering.labels_) if label == i
        ]
        distances_to_others = [
            sum(
                patho_distance_matrix[idx, other_idx]
                for other_idx in indices
                if other_idx != idx
            )
            for idx in indices
        ]
        representative_idx = indices[
            distances_to_others.index(max(distances_to_others))
        ]
        patho_representative_ids.append(patho_document_ids[representative_idx])
    patho_representative_docs = [
        doc for doc in patho_documents if doc["document"] in patho_representative_ids
    ]
    for _index, dictionary in enumerate(patho_representative_docs):
        with open(str(output_folder) + "/patho/" + dictionary["document"], "w") as file:
            file.write(dictionary["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)
