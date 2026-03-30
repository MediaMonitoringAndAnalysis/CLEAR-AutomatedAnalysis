from ast import literal_eval
import pandas as pd
import os
from src.analysis.documents_based_analysis import _perform_documents_based_analysis

countries_to_analyze = ["Iran", "Lebanon"]


def _preprocess_classification_results(classification_results: str) -> list[str]:
    number_tags = {
        "Pillars 2D->At Risk->Number Of People At Risk": "Pillars 2D->At Risk->Risk And Vulnerabilities",
        "Pillars 2D->Impact->Number Of People Affected": "Pillars 2D->Impact->Impact On People",
        "Pillars 2D->Humanitarian Conditions->Number Of People In Need": "Pillars 2D->Humanitarian Conditions->Living Standards",
    }
    final_classification = []
    classification_results_list = literal_eval(classification_results)
    for tag in classification_results_list:
        if tag in number_tags:
            final_classification.append(number_tags[tag])
        else:
            final_classification.append(
                tag.replace(
                    "Pillars 2D->Priority Interventions", "Pillars 2D->Priority Needs"
                )
            )
    return sorted(list(set(final_classification)))


def _import_classification_dataset(
    classification_dataset_path: str,
    classification_column: str = "First Level Classification",
) -> pd.DataFrame:
    df = pd.read_csv(classification_dataset_path)
    df[classification_column] = df[classification_column].apply(
        _preprocess_classification_results
    )
    df = df.sort_values(by="Extraction Text", key=lambda col: col.fillna("").str.len())
    return df


if __name__ == "__main__":
    classification_dataset_path = "data/WestAsia2026/classification_dataset.csv"
    classification_column = "First Level Classification"
    n_kept_entries = 15
    save_folder = "data/WestAsia2026/analysis"
    answers_save_path = "answers.json"
    risk_list_save_path = "risk_list.json"
    key_indicator_numbers_save_path = "key_indicator_numbers.json"
    classification_df = _import_classification_dataset(
        classification_dataset_path, classification_column
    )
    countries = classification_df["Primary Country"].unique()
    for country in countries_to_analyze:
        one_country_classification_df = classification_df[
            classification_df["Primary Country"].apply(lambda x: country in x)
        ]
        answer_df, risk_list_df, key_indicator_numbers_df = (
            _perform_documents_based_analysis(
                one_country_classification_df,
                country,
                classification_column,
                n_kept_entries,
                os.path.join(save_folder, country),
                answers_save_path,
                risk_list_save_path,
                key_indicator_numbers_save_path,
            )
        )
