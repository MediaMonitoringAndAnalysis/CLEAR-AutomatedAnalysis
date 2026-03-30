from ast import literal_eval
import pandas as pd
from data_connectors import get_reliefweb_leads
from src.analysis.documents_based_analysis import _perform_documents_based_analysis
import argparse
import dotenv
import os

dotenv.load_dotenv()

openai_api_key = os.getenv("openai_api_key")
RW_url = "https://reliefweb.int/updates?advanced-search=%28PC121.PC137%29_%28DO20260315-20260321%29&page={}"


def _extract_entries(
    leads: pd.DataFrame,
    text_column: str = "text",
    entries_column: str = "Extraction Text",
) -> pd.DataFrame:

    from entry_extraction import SemanticEntriesExtractor

    extractor = SemanticEntriesExtractor(max_sentences=5, overlap=2)
    entries = extractor(leads[text_column].tolist())
    leads[entries_column] = entries
    leads = leads.explode(entries_column)
    leads = leads.drop_duplicates(subset=[entries_column])
    leads = leads[leads[entries_column].apply(lambda x: len(str(x)) > 3)].drop(columns=[text_column])
    return leads


def _classify_entries(
    entries: pd.DataFrame,
    entries_column: str = "Extraction Text",
    classification_column: str = "First Level Classification",
    prediction_ratio: float = 1.05,
) -> pd.DataFrame:
    from humanitarian_extract_classificator import humbert_classification

    entries[classification_column] = humbert_classification(
        entries[entries_column].tolist(), prediction_ratio=prediction_ratio
    )
    return entries


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_bool", type=str, default="true")
    parser.add_argument("--project_name", type=str, default="WestAsia2026")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--entries_column", type=str, default="Extraction Text")
    parser.add_argument(
        "--classification_column", type=str, default="First Level Classification"
    )
    parser.add_argument("--prediction_ratio", type=float, default=1.05)
    parser.add_argument("--countries_to_analyze", type=str, nargs="+", default=["Iran", "Lebanon"])
    parser.add_argument("--n_kept_entries", type=int, default=15)
    parser.add_argument("--answers_save_path", type=str, default="answers.json")
    parser.add_argument("--risk_list_save_path", type=str, default="risk_list.json")
    parser.add_argument("--key_indicator_numbers_save_path", type=str, default="key_indicator_numbers.json")
    args = parser.parse_args()

    sample = args.sample_bool.lower() == "true"

    leads_df = get_reliefweb_leads(
        project_page_starting_url=RW_url,
        project_name=args.project_name,
        data_folder="data",
        extracted_data_path="data/leads.csv",
        openai_api_key=openai_api_key,  # falls back to OPENAI_API_KEY env var
        extract_pdf_text=True,
        save=True,
        sample=sample,
    )

    classification_dataset_path = os.path.join(
        "data", args.project_name, "classification_dataset.csv"
    )

    if not os.path.exists(classification_dataset_path):
        entries_df = _extract_entries(
            leads_df, text_column=args.text_column, entries_column=args.entries_column
        )
        entries_df.to_csv(classification_dataset_path, index=False)
    else:
        entries_df = pd.read_csv(classification_dataset_path)

    if args.classification_column not in entries_df.columns:
        entries_df = _classify_entries(
            entries_df, classification_column=args.classification_column, prediction_ratio=args.prediction_ratio
        )
        entries_df.to_csv(classification_dataset_path, index=False)

    if "Entry ID" not in entries_df.columns:
        entries_df["Entry ID"] = entries_df.index
        entries_df.to_csv(classification_dataset_path, index=False)

    save_folder = os.path.join("data", args.project_name, "analysis")
    classification_df = _import_classification_dataset(
        classification_dataset_path, args.classification_column
    )
    for country in args.countries_to_analyze:
        one_country_classification_df = classification_df[
            classification_df["Primary Country"].apply(lambda x: country in x)
        ]
        answer_df, risk_list_df, key_indicator_numbers_df = (
            _perform_documents_based_analysis(
                one_country_classification_df,
                country,
                args.classification_column,
                args.n_kept_entries,
                os.path.join(save_folder, country),
                args.answers_save_path,
                args.risk_list_save_path,
                args.key_indicator_numbers_save_path,
            )
        )
