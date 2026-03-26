import pandas as pd
from data_connectors import get_reliefweb_leads
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
) -> pd.DataFrame:
    from humanitarian_extract_classificator import humbert_classification

    entries[classification_column] = humbert_classification(
        entries[entries_column].tolist(), prediction_ratio=0.9
    )
    return entries


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_bool", type=str, default="true")
    parser.add_argument("--project_name", type=str, default="WestAsia2026")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--entries_column", type=str, default="Extraction Text")
    parser.add_argument(
        "--classification_column", type=str, default="First Level Classification"
    )
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
            entries_df, classification_column=args.classification_column
        )
        entries_df.to_csv(classification_dataset_path, index=False)

    