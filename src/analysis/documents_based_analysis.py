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
    "priority_needs": [],
    "priority_interventions": [],
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
  - overall_risk_score: a score from 0 to 10 indicating the overall risk score of the answer. The score is the average of the risk scores of the risks in the risk_list.
  - ID: list of all extract IDs that contributed to the answer, even partially.

- risk_list: a list of risk objects, each with the following keys:
  - risk: name of the risk (a risk to life, dignity, or basic needs of the affected population)
  - risk_score: integer 0–10 indicating severity (0 = no risk, 5 = moderate humanitarian concern, 10 = catastrophic/life-threatening risk)
  - ID: list of extract IDs supporting this risk assessment

- key_indicator_numbers: a list of key numerical indicators, each with the following keys: (each indicator is a numerical data point that is relevant to the topic and questions)
  - key_indicator: name of the indicator (e.g. "IDPs", "acute malnutrition rate", "schools destroyed")
  - number: the numeric value
  - unit: the unit of the number (e.g. "people", "%", "USD", "MT")
  - location: locations associated with this indicator, comma-separated (or "-" if not specified)
  - specific_population: specific population group (e.g. "children under 5", "women", "refugees") (or "-" if not specified)
  - date: date of the indicator in dd-mm-YYYY format (or "-" if not specified)
  - risk_score: integer 0–10 indicating the severity implied by this indicator (same scale as above)
  - ID: list of extract IDs supporting this data point

- priority_needs: a list of priority needs, each with the following keys:
  - priority_need: from the list of risks and key indicator numbers, name of the priority need, keeping it detailed and specific but not more than 2 sentences.
  - priority_need_score: integer 0–10 indicating the severity of the priority need (same scale as above)

- priority_interventions: a list of recommended interventions, each with the following keys:
  - priority_intervention: from the list of risks and priority needs, name of the recommended intervention, keeping it detailed and specific but not more than 2 sentences.
  - priority_intervention_score: integer 0–10 indicating the severity of the recommended intervention (same scale as above)

Rules:
- Base your answers strictly on the provided extracts. Do not infer, extrapolate, or combine facts not explicitly stated together in the source material.
- If the context does not contain enough information to answer a question, return an empty list or empty string for the relevant field.
- If a field value is not available in the extracts, return "-".
- If no response if relevant, just respond with "-". 
- Do not answer with "N/A" or "Not available" or any other similar phrase in the type "not enough information", just say "-".

When answering, focus only on the topic discussed. nothing else.
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
            tag_final_name = f"Pillars 1D->{pillar_1d_name}->{subpillar_1d_name}"
            # print(tag_final_name)
            # print(classification_df[classification_column].iloc[0][tag_final_name])

            mask = classification_df[classification_column].apply(
                    lambda x: x.get(tag_final_name, 0) >= 1
                )
            subpillar_df = classification_df[
                mask
            ].copy()
            subpillar_df["tag_value"] = subpillar_df[classification_column].apply(
                lambda x: x[tag_final_name]
            )
            subpillar_df = subpillar_df.sort_values(by="tag_value", ascending=False)
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
                tag_sector_final_name = f"Sectors->{sector}"
                tag_final_name = (
                    f"Pillars 2D->{pillar_2d_name}->{subpillar_2d_name}"
                )
                sector_df = classification_df[
                    classification_df[classification_column].apply(
                        lambda x: x.get(tag_sector_final_name, 0) >= 1
                        and x.get(tag_final_name, 0) >= 1
                    )
                ].copy()
                sector_df["tag_value"] = sector_df[classification_column].apply(
                    lambda x: min(x.get(tag_sector_final_name, 0), x.get(tag_final_name, 0))
                )
                sector_df = sector_df.sort_values(by="tag_value", ascending=False)
                print(subpillar_2d_name, sector, len(sector_df))

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
    priority_needs_save_path: str = "priority_needs.json",
    priority_interventions_save_path: str = "priority_interventions.json",
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

    # print(answers[0])

    # from the answers structured format, append to 3 dataframes:
    # - answer_df: the answers in a structured format
    # - risk_list_df: the risk list in a structured format
    # - key_indicator_numbers_df: the key indicator numbers in a structured format
    answer_df = pd.DataFrame([answer["answer"] for answer in answers])
    risk_list_df = analysis_df[["task", "pillar", "subpillar", "sector"]].copy()
    key_indicator_numbers_df = analysis_df[
        ["task", "pillar", "subpillar", "sector"]
    ].copy()
    priority_needs_df = analysis_df[["task", "pillar", "subpillar", "sector"]].copy()
    priority_interventions_df = analysis_df[
        ["task", "pillar", "subpillar", "sector"]
    ].copy()

    risk_list_df["risk_list"] = [answer["risk_list"] for answer in answers]
    key_indicator_numbers_df["key_indicator_numbers"] = [
        answer["key_indicator_numbers"] for answer in answers
    ]
    priority_needs_df["priority_needs"] = [
        answer["priority_needs"] for answer in answers
    ]
    priority_interventions_df["priority_interventions"] = [
        answer["priority_interventions"] for answer in answers
    ]

    risk_list_df = risk_list_df.explode("risk_list")
    risk_list_df = risk_list_df[
        risk_list_df["risk_list"].apply(lambda x: len(str(x)) > 5)
    ]
    key_indicator_numbers_df = key_indicator_numbers_df.explode("key_indicator_numbers")
    key_indicator_numbers_df = key_indicator_numbers_df[
        key_indicator_numbers_df["key_indicator_numbers"].apply(
            lambda x: len(str(x)) > 5
        )
    ]
    priority_needs_df = priority_needs_df.explode("priority_needs")
    priority_needs_df = priority_needs_df[
        priority_needs_df["priority_needs"].apply(lambda x: len(str(x)) > 5)
    ]
    priority_interventions_df = priority_interventions_df.explode(
        "priority_interventions"
    )
    priority_interventions_df = priority_interventions_df[
        priority_interventions_df["priority_interventions"].apply(
            lambda x: len(str(x)) > 5
        )
    ]

    for risk_cols in ["risk", "risk_score", "ID"]:
        risk_list_df[risk_cols] = risk_list_df["risk_list"].apply(
            lambda x: x[risk_cols]
        )

    for key_indicator_cols in [
        "key_indicator",
        "number",
        "unit",
        "location",
        "specific_population",
        "date",
        "risk_score",
        "ID",
    ]:
        key_indicator_numbers_df[key_indicator_cols] = key_indicator_numbers_df[
            "key_indicator_numbers"
        ].apply(lambda x: x[key_indicator_cols])

    risk_list_df = risk_list_df.drop(columns=["risk_list"])
    key_indicator_numbers_df = key_indicator_numbers_df.drop(
        columns=["key_indicator_numbers"]
    )

    for priority_cols in ["priority_need", "priority_need_score"]:
        priority_needs_df[priority_cols] = priority_needs_df["priority_needs"].apply(
            lambda x: x[priority_cols]
        )
    for priority_cols in ["priority_intervention", "priority_intervention_score"]:
        priority_interventions_df[priority_cols] = priority_interventions_df["priority_interventions"].apply(
            lambda x: x[priority_cols]
        )
    priority_needs_df = priority_needs_df.drop(columns=["priority_needs"])
    priority_interventions_df = priority_interventions_df.drop(
        columns=["priority_interventions"]
    )

    for col in ["task", "pillar", "subpillar", "sector"]:
        answer_df[col] = analysis_df[col].values

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
    priority_needs_df.to_json(
        os.path.join(save_folder, priority_needs_save_path),
        orient="records",
        indent=4,
    )
    priority_interventions_df.to_json(
        os.path.join(save_folder, priority_interventions_save_path),
        orient="records",
        indent=4,
    )
    # return answer_df, risk_list_df, key_indicator_numbers_df
