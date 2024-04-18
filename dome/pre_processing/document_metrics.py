import csv
import json
import os

from nltk.tokenize import sent_tokenize, word_tokenize


# Generalized function to compute document metrics
def compute_document_metrics(text: str) -> tuple:
    tokens = word_tokenize(text, language="german")
    sentences = sent_tokenize(text, language="german")
    num_tokens = len(tokens)
    num_sentences = len(sentences)
    avg_word_length = (
        sum(len(word) for word in tokens) / num_tokens if num_tokens > 0 else 0
    )
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
    num_characters = len(text)
    return (
        num_tokens,
        num_sentences,
        num_characters,
        avg_word_length,
        avg_sentence_length,
    )


def compute_overall_metrics(
    total_num_tokens: int,
    total_num_sentences: int,
    total_characters: int,
    total_word_length: int,
    total_sentence_length: int,
    text: str,
) -> tuple:
    (
        num_tokens,
        num_sentences,
        num_characters,
        avg_word_length,
        avg_sentence_length,
    ) = compute_document_metrics(text)
    total_num_tokens += num_tokens
    total_num_sentences += num_sentences
    total_word_length += num_tokens * avg_word_length
    total_sentence_length += num_sentences * avg_sentence_length
    total_characters += num_characters
    return (
        total_num_tokens,
        total_num_sentences,
        total_word_length,
        total_sentence_length,
        total_characters,
    )


# Function to process text files in a folder
def process_files(
    base_path: str, is_json: bool = False, documents_key: str = None
) -> tuple:
    total_num_tokens = 0
    total_num_sentences = 0
    total_word_length = 0
    total_sentence_length = 0
    total_characters = 0

    for filename in os.listdir(base_path):
        if filename.endswith(".json") and is_json:
            file_path = os.path.join(base_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_data = json.load(file)
                for document in batch_data[documents_key]["textAnalysisResultDtos"]:
                    for annotation in document["annotationDtos"]:
                        if (
                            annotation["type"]
                            == "de.averbis.types.health.DocumentAnnotation"
                        ):
                            text = annotation["coveredText"]
                            (
                                total_num_tokens,
                                total_num_sentences,
                                total_word_length,
                                total_sentence_length,
                                total_characters,
                            ) = compute_overall_metrics(
                                total_num_tokens,
                                total_num_sentences,
                                total_characters,
                                total_word_length,
                                total_sentence_length,
                                text,
                            )

        elif filename.endswith(".txt") and not is_json:
            file_path = os.path.join(base_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                (
                    total_num_tokens,
                    total_num_sentences,
                    total_word_length,
                    total_sentence_length,
                    total_characters,
                ) = compute_overall_metrics(
                    total_num_tokens,
                    total_num_sentences,
                    total_characters,
                    total_word_length,
                    total_sentence_length,
                    text,
                )

    # Calculate average word length and average sentence length across all documents
    avg_word_length_overall = (
        total_word_length / total_num_tokens if total_num_tokens > 0 else 0
    )
    avg_sentence_length_overall = (
        total_sentence_length / total_num_sentences if total_num_sentences > 0 else 0
    )

    # Return computed metrics over all documents
    return (
        total_num_tokens,
        total_num_sentences,
        total_characters,
        avg_word_length_overall,
        avg_sentence_length_overall,
    )


# Function to write metrics to CSV
def write_metrics_to_csv(metrics: tuple, output_csv_file: str) -> None:
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Total Num Tokens",
            "Total Num Sentences",
            "Total Characters",
            "Average Word Length (Overall)",
            "Average Sentence Length (Overall)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "Total Num Tokens": metrics[0],
                "Total Num Sentences": metrics[1],
                "Total Characters": metrics[2],
                "Average Word Length (Overall)": metrics[3],
                "Average Sentence Length (Overall)": metrics[4],
            }
        )


# Example usage for processing all JSON batch files in a folder
json_batches_folder_path = "/path/to/json"
documents_key = "payload"  # Replace 'payload' with the appropriate key that holds the list of documents in your JSON

total_metrics_json = process_files(
    json_batches_folder_path, is_json=True, documents_key=documents_key
)

# Write metrics for JSON batch files to CSV
output_csv_file_json = "output_json.csv"
write_metrics_to_csv(total_metrics_json, output_csv_file_json)

# Example usage for processing all plain text files in a folder
plain_text_files_folder_path = "/path/to/txt"

total_metrics_text = process_files(plain_text_files_folder_path)

# Write metrics for plain text files to CSV
output_csv_file_text = "output_txt.csv"
write_metrics_to_csv(total_metrics_text, output_csv_file_text)
