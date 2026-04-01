import pandas as pd
import os
import dotenv
from llm_multiprocessing_inference import get_answers

dotenv.load_dotenv()

openai_api_key = os.getenv("openai_api_key")

date_precision_possible_values = ["Day", "Week", "Month", "Year"]

quantifier_possible_values = [
    "Exact",
    "Approximately",
    "Less Than",
    "More Than",
    "At Least",
]

# Quantifier mapping examples used in the system prompt
quantifier_examples = """
  - "500 people" → "Exact"
  - "at least 500" → "At Least"
  - "around 200" / "approximately 200" → "Approximately"
  - "more than 1,000" → "More Than"
  - "fewer than 50" / "less than 50" → "Less Than"
"""

generic_prompt = """
You are a humanitarian data analyst specializing in extracting displacement and crisis figures from field reports.

I will provide you with an excerpt from a humanitarian document.
Your output must be ONLY a valid JSON array of dictionaries — one dictionary per extracted number.
Do not include any prose, explanation, or markdown code fences.

A number is relevant only if BOTH its associated "unit" and "what_happened" can be found in the excerpt.
Exclude numbers that represent: percentages without a raw count, years, reference codes, phone numbers, or purely administrative figures (e.g., "article 3", "section 2").
If no relevant number can be extracted, return an empty array: []
"""

unknown_output_prompt = """
- If only one of start_date or end_date is available, return the available date in the start_date field.
- If only one of start_location or end_location is available, return the available location in the start_location field.
- If you cannot find any information in any field, return an empty string: "-".
"""

few_shot_example = """
Example input:
"At least 3,500 families were displaced from Idlib governorate to Aleppo in March 2024. Around 200 homes were destroyed."

Example output:
[
  {
    "number": 3500,
    "unit": "families",
    "what_happened": "displaced",
    "start_date": "2024-03-01",
    "start_date_precision": "Month",
    "end_date": "-",
    "end_date_precision": "-",
    "start_location": "Idlib governorate",
    "end_location": "Aleppo",
    "quantifier": "At Least",
    "risk_score": 5
  },
  {
    "number": 200,
    "unit": "homes",
    "what_happened": "destroyed",
    "start_date": "2024-03-01",
    "start_date_precision": "Month",
    "end_date": "-",
    "end_date_precision": "-",
    "start_location": "Idlib governorate",
    "end_location": "-",
    "quantifier": "Approximately",
    "risk_score": 3
  }
]
"""

numerical_data_extraction_system_prompt = f"""
Each extracted dictionary must contain the following fields:

"number": int — the numeric value mentioned in the excerpt, returned as an integer.
  For ranges (e.g., "3,000–5,000"), return the lower bound.

"unit": str — the entity counted by the number (e.g., individuals, men, women, children, families/households, houses, homes, shelters, buildings, etc.)

"what_happened": str — the humanitarian impact linked to the number (e.g., displaced, injured, affected, evacuated, killed, missing, destroyed, damaged, uninhabitable, etc.)

"start_date": str — the date the event began: YYYY-MM-DD
"start_date_precision": Literal{date_precision_possible_values} — precision of the start date
"end_date": str — the date the event ended or was last reported: YYYY-MM-DD
"end_date_precision": Literal{date_precision_possible_values} — precision of the end date

"start_location": str — location where the event occurred (up to administrative level 4 precision)
"end_location": str — destination location, only relevant for displacement events (origin vs. destination); otherwise "-"

"quantifier": Literal{quantifier_possible_values} — how the number is qualified in the text:
"risk_score": int — the risk score of the number, from 0 to 10 (0 = no risk, 10 = catastrophic risk)

{quantifier_examples}

{unknown_output_prompt}

{few_shot_example}

Ensure all returned values match the specified types. If unsure about an entry, return it anyway (high recall is crucial).
"""

_NUMBERS_TAGS = [
    "Pillars 2D->At Risk->Number Of People At Risk",
    "Pillars 2D->Impact->Number Of People Affected",
    "Pillars 2D->Humanitarian Conditions->Number Of People In Need",
    "Pillars 1D->Displacement->Type-Numbers-Movements",
    "Pillars 1D->Casualties->Dead",
    "Pillars 1D->Casualties->Injured",
    "Pillars 1D->Casualties->Missing",
]


def _extract_numbers(classification_df: pd.DataFrame) -> pd.DataFrame:
    numbers_extraction_entries_df = (
        classification_df[
            classification_df["First Level Classification"].apply(
                lambda x: max([x.get(tag, 0) for tag in _NUMBERS_TAGS]) >= 1.05
            )
        ][["Entry ID", "Extraction Text", "First Level Classification"]]
        .drop_duplicates("Extraction Text")
        .reset_index(drop=True)
        .copy()
    )
    numbers_extraction_entries_df["tag"] = numbers_extraction_entries_df["First Level Classification"].apply(
        lambda x: [tag for tag in x.keys() if x.get(tag, 0) >= 1.05]
    )

    # print(f"Extracting numbers for {len(numbers_extraction_entries_df)} entries")
    # numbers_extraction_entries_df = numbers_extraction_entries_df.iloc[:25]

    llm_entries = []
    for _, row in numbers_extraction_entries_df.iterrows():
        one_entry = [
            {
                "role": "system",
                "content": numerical_data_extraction_system_prompt,
            },
            {"role": "user", "content": row["Extraction Text"]},
        ]
        llm_entries.append(one_entry)

    llm_entries_results = get_answers(
        llm_entries,
        model="gpt-4.1-mini",
        response_type="structured",
        default_response=[],
        api_key=openai_api_key,
        api_pipeline="OpenAI",
    )

    final_entry_df = pd.DataFrame()
    for i in range(len(llm_entries_results)):
        id_ = numbers_extraction_entries_df.iloc[i]["Entry ID"]
        entry_text = numbers_extraction_entries_df.iloc[i]["Extraction Text"]
        llm_entry_result = llm_entries_results[i]
        
        if len(llm_entry_result) > 0:
            for one_entry in llm_entry_result:
                one_entry_df = pd.DataFrame([one_entry])
                one_entry_df["Entry ID"] = id_
                one_entry_df["Extraction Text"] = entry_text
                one_entry_df["tag"] = numbers_extraction_entries_df.iloc[i]["tag"]
                final_entry_df = pd.concat(
                    [final_entry_df, one_entry_df], ignore_index=True
                )

    final_entry_df = final_entry_df.reset_index(
        drop=True
    ).copy()

    return final_entry_df


def performs_numbers_extraction(classification_df: pd.DataFrame, output_path: os.PathLike) -> pd.DataFrame:

    if not os.path.exists(output_path):
        numbers_extraction_entries_df = _extract_numbers(classification_df)
        numbers_extraction_entries_df.to_csv(output_path, index=False)
