import pandas as pd
import json
import os
from llm_multiprocessing_inference import get_answers
from src.analysis.analytical_questions import (
    web_based_questions,
    situation_analysis_1d,
    situation_analysis_2d,
    needs_and_response_questions,
)
from src.analysis.analytical_questions import sectors
import dotenv

dotenv.load_dotenv()

answer_default_response = {
    "answer": {
        "text": "-",
        "relevance": 0,
        "ID": [],
    },
    "risk_list": [],
    "key_indicator_numbers": [],
}

documents_based_analysis_system_prompt = """
I am writing a secondary data analysis for a humanitarian situation.

I will provide you with a list of inputs in the form of a JSON dictionary with the following keys:
- topic: the topic of the analysis
- questions: a list of questions to answer
- context: a JSON dictionary where the keys are the IDs of the extracts and the values are the text of the extracts. Extracts may be in English, French, or Spanish.

You will answer the questions based solely on the provided context. Always write your output in English.

Return the answer as a JSON dictionary with the following keys:

- answer:
  - text: a complete, precise, and synthesised answer to the questions in markdown format (use paragraphs and line breaks for clarity). Only include information relevant to the questions.
  - relevance: a score from 0 to 1 indicating how relevant the answer is to the question. Scores under 0.5 mean that the answer is not relevant. The score of an answer that doesn't answer any part of the question is 0 and the score of an answer that answers all parts of the question is 1. Partially relevant answers should be scored between 0.5 and 1. It is better to return unnecessary information than missing important ones.
  - ID: list of all extract IDs that contributed to the answer, even partially.

- risk_list: a list of risk objects, each with the following keys:
  - risk: name of the risk (a risk to life, dignity, or basic needs of the affected population)
  - risk_score: integer 0–10 indicating severity (0 = no risk, 5 = moderate humanitarian concern, 10 = catastrophic/life-threatening risk)
  - ID: list of extract IDs supporting this risk assessment

- key_indicator_numbers: a list of key indicator objects, each with the following keys:
  - key_indicator: name of the indicator (e.g. "IDPs", "acute malnutrition rate", "schools destroyed")
  - number: the numeric value
  - unit: the unit of the number (e.g. "people", "%", "USD", "MT")
  - location: locations associated with this indicator, comma-separated (or "-" if not specified)
  - specific_population: specific population group (e.g. "children under 5", "women", "refugees") (or "-" if not specified)
  - date: date of the indicator in dd-mm-YYYY format (or "-" if not specified)
  - risk_score: integer 0–10 indicating the severity implied by this indicator (same scale as above)
  - ID: list of extract IDs supporting this data point

Rules:
- Base your answers strictly on the provided extracts. Do not infer, extrapolate, or combine facts not explicitly stated together in the source material.
- If the context does not contain enough information to answer a question, return an empty list or empty string for the relevant field.
- If a field value is not available in the extracts, return "-".
- If no response if relevant, just respond with "-". 
- Do not answer with "N/A" or "Not available" or any other similar phrase in the type "not enough information", just say "-".
"""


def _create_analysis_prompts(
    classification_df: pd.DataFrame,
    country: str,
    classification_column: str = "First Level Classification",
    n_kept_entries: int = 15,
):
    analysis_prompts = []
    analysis_df = pd.DataFrame()
    # 1d prompts
    for pillar_1d_name, pillar_1d_questions in situation_analysis_1d.items():
        for subpillar_1d_name, subpillar_1d_questions in pillar_1d_questions.items():
            subpillar_df = classification_df[
                classification_df[classification_column].apply(
                    lambda x: subpillar_1d_name in str(x)
                )
            ].copy()
            print(pillar_1d_name, subpillar_1d_name, len(subpillar_df))
            if subpillar_df.empty:
                continue
            else:

                topic = f"{pillar_1d_name} -> {subpillar_1d_name} in {country}"
                context = {
                    str(subpillar_df.iloc[i]["Entry ID"]): str(
                        subpillar_df.iloc[i]["Extraction Text"]
                    )
                    for i in range(min(n_kept_entries, len(subpillar_df)))
                }

                analysis_prompts.append(
                    [
                        {
                            "role": "system",
                            "content": documents_based_analysis_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "topic": topic,
                                    "questions": subpillar_1d_questions,
                                    "context": context,
                                }
                            ),
                        },
                    ]
                )
                analysis_df = pd.concat(
                    [
                        analysis_df,
                        pd.DataFrame(
                            [
                                {
                                    "task": "situation_analysis_1d",
                                    "country": country,
                                    "pillar": pillar_1d_name,
                                    "subpillar": subpillar_1d_name,
                                    "sector": "-",
                                    "context": context,
                                }
                            ]
                        ),
                    ]
                )

    # 2d prompts
    for pillar_2d_name, pillar_2d_questions in situation_analysis_2d.items():
        for subpillar_2d_name, subpillar_2d_questions in pillar_2d_questions.items():
            for sector in sectors:
                sector_df = classification_df[
                    classification_df[classification_column].apply(
                        lambda x: sector in str(x) and subpillar_2d_name in str(x)
                    )
                ].copy()
                print(pillar_2d_name, subpillar_2d_name, sector, len(sector_df))
                context = {
                    str(sector_df.iloc[i]["Entry ID"]): str(
                        sector_df.iloc[i]["Extraction Text"]
                    )
                    for i in range(min(n_kept_entries, len(sector_df)))
                }
                if sector_df.empty:
                    continue
                else:
                    topic = f"{pillar_2d_name} -> {subpillar_2d_name} for the {sector} sector in {country}"
                    analysis_prompts.append(
                        [
                            {
                                "role": "system",
                                "content": documents_based_analysis_system_prompt,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "topic": topic,
                                        "questions": subpillar_2d_questions,
                                        "context": context,
                                    }
                                ),
                            },
                        ]
                    )

                    analysis_df = pd.concat(
                        [
                            analysis_df,
                            pd.DataFrame(
                                [
                                    {
                                        "task": "situation_analysis_2d",
                                        "country": country,
                                        "pillar": pillar_2d_name,
                                        "subpillar": subpillar_2d_name,
                                        "sector": sector,
                                        "context": context,
                                    }
                                ]
                            ),
                        ]
                    )

    return analysis_prompts, analysis_df


def _perform_documents_based_analysis(
    classification_df: pd.DataFrame,
    country: str,
    classification_column: str = "First Level Classification",
    n_kept_entries: int = 15,
    save_folder: str = "analysis",
    answers_save_path: str = "answers.json",
    risk_list_save_path: str = "risk_list.json",
    key_indicator_numbers_save_path: str = "key_indicator_numbers.json",
):

    os.makedirs(save_folder, exist_ok=True)
    analysis_prompts, analysis_df = _create_analysis_prompts(
        classification_df, country, classification_column, n_kept_entries
    )
    answers = get_answers(
        prompts=analysis_prompts,
        default_response=answer_default_response,
        response_type="structured",
        api_pipeline="OpenAI",
        api_key=os.getenv("openai_api_key"),
        model="gpt-4.1-mini",
        additional_progress_bar_description=f"documents-based analysis for {country}",
    )

    # from the answers structured format, append to 3 dataframes:
    # - answer_df: the answers in a structured format
    # - risk_list_df: the risk list in a structured format
    # - key_indicator_numbers_df: the key indicator numbers in a structured format
    answer_df = pd.DataFrame([answer["answer"] for answer in answers])
    risk_list_df = pd.DataFrame([answer["risk_list"] for answer in answers])
    key_indicator_numbers_df = pd.DataFrame(
        [answer["key_indicator_numbers"] for answer in answers]
    )

    for col in ["task", "pillar", "subpillar", "sector"]:
        answer_df[col] = analysis_df[col].values
        risk_list_df[col] = analysis_df[col].values
        key_indicator_numbers_df[col] = analysis_df[col].values

    answer_df.to_json(
        os.path.join(save_folder, answers_save_path), orient="records", indent=4
    )
    risk_list_df.to_json(
        os.path.join(save_folder, risk_list_save_path), orient="records", indent=4
    )
    key_indicator_numbers_df.to_json(
        os.path.join(save_folder, key_indicator_numbers_save_path),
        orient="records",
        indent=4,
    )

    return answer_df, risk_list_df, key_indicator_numbers_df
