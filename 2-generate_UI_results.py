"""
Generate final dashboard visualisation data from CLEAR analysis outputs.

Usage:
    python generate_final_results.py [--data-folder DATA_FOLDER] [--viz-folder VIZ_FOLDER]

Defaults:
    --data-folder  data/WestAsia2026/analysis/Lebanon/
    --viz-folder   src/viz/
"""

import argparse
import json
import os
from ast import literal_eval
from collections import defaultdict

import pandas as pd

from src.analysis.merge_numbers import merge_entries_by_number


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_SCALE = {
    0: "UNKNOWN",
    1: "MINOR",
    2: "MINOR",
    3: "MODERATE",
    4: "MODERATE",
    5: "SERIOUS",
    6: "SERIOUS",
    7: "SEVERE",
    8: "SEVERE",
    9: "CRITICAL",
    10: "CRITICAL",
}

PILLARS_2D = ["Impact", "Humanitarian Conditions", "At Risk"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_folder: str) -> dict:
    """Load all required input files and return them as a dict of DataFrames."""
    with open(os.path.join(data_folder, "answers.json")) as f:
        answers_df = pd.DataFrame(json.load(f))

    with open(os.path.join(data_folder, "key_indicator_numbers.json")) as f:
        key_indicator_numbers_df = pd.DataFrame(json.load(f))

    with open(os.path.join(data_folder, "risk_list.json")) as f:
        risks_df = pd.DataFrame(json.load(f))

    with open(os.path.join(data_folder, "priority_needs.json")) as f:
        priority_needs_df = pd.DataFrame(json.load(f))

    with open(os.path.join(data_folder, "priority_interventions.json")) as f:
        priority_interventions_df = pd.DataFrame(json.load(f))

    with open(os.path.join(data_folder, "context_figures.json")) as f:
        context_figures = json.load(f)

    numbers_df = pd.read_csv(os.path.join(data_folder, "numbers_extraction.csv"))

    classification_csv = os.path.join(data_folder, "../..", "classification_dataset.csv")
    classification_df = pd.read_csv(classification_csv)
    classification_df["Document Source"] = classification_df["Document Source"].apply(literal_eval)
    classification_df = classification_df[["Entry ID", "Document Source"]]
    classification_df["Entry ID"] = classification_df["Entry ID"].astype(int)

    return {
        "answers_df": answers_df,
        "key_indicator_numbers_df": key_indicator_numbers_df,
        "risks_df": risks_df,
        "priority_needs_df": priority_needs_df,
        "priority_interventions_df": priority_interventions_df,
        "context_figures": context_figures,
        "numbers_df": numbers_df,
        "classification_df": classification_df,
    }


# ---------------------------------------------------------------------------
# JS helpers
# ---------------------------------------------------------------------------

def _write_js(viz_folder: str, filename: str, variable_name: str, data) -> None:
    """Serialise *data* to a JS file that sets a named window variable."""
    os.makedirs(viz_folder, exist_ok=True)
    js_content = (
        f"window.{variable_name} = "
        + json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True)
        + ";\n"
    )
    path = os.path.join(viz_folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Saved {filename}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_shown_risks(
    risks_df: pd.DataFrame,
    data_folder: str,
    viz_folder: str,
) -> dict:
    """
    For each (pillar_2d, sector) pair compute the highest risk score and the
    top-3 risks, then persist results as JSON and JS.
    """
    sectors_risks = risks_df[risks_df.task == "situation_analysis_2d"]
    sectors_list = sectors_risks.sector.unique()

    shown_risks: dict = defaultdict(lambda: defaultdict(dict))
    for pillar_2d in PILLARS_2D:
        for sector in sectors_list:
            subset = sectors_risks[
                (sectors_risks.pillar == pillar_2d) & (sectors_risks.sector == sector)
            ]
            if subset.empty:
                continue
            highest_score = int(subset.risk_score.max())
            shown_risks[pillar_2d][sector]["highest_score"] = highest_score
            shown_risks[pillar_2d][sector]["severity_scale"] = SEVERITY_SCALE.get(
                highest_score, "UNKNOWN"
            )
            top3 = subset.sort_values(by="risk_score", ascending=False).head(3)
            shown_risks[pillar_2d][sector]["top3_risks"] = [
                row["risk"] for _, row in top3.iterrows()
            ]

    # Persist as JSON
    json_path = os.path.join(data_folder, "shown_risks.json")
    with open(json_path, "w") as f:
        json.dump(shown_risks, f, allow_nan=True)
    print(f"Saved shown_risks.json")

    _write_js(viz_folder, "shown_risks_data.js", "SHOWN_RISKS_DATA", shown_risks)
    return dict(shown_risks)


def generate_humanitarian_access_data(
    risks_df: pd.DataFrame,
    key_indicator_numbers_df: pd.DataFrame,
    viz_folder: str,
) -> dict:
    """
    Build a dict of high-scoring humanitarian-access risks and key numbers,
    keyed by subpillar / key_indicator label.
    """
    ha_risks_df = risks_df[risks_df.pillar == "Humanitarian Access"]
    result: dict = defaultdict()

    for subpillar in ha_risks_df.subpillar.unique():
        high_risks = ha_risks_df[
            (ha_risks_df.subpillar == subpillar) & (ha_risks_df.risk_score >= 8)
        ].risk.unique().tolist()

        high_numbers_df = key_indicator_numbers_df[
            (key_indicator_numbers_df.pillar == "Humanitarian Access")
            & (key_indicator_numbers_df.subpillar == subpillar)
            & (key_indicator_numbers_df.risk_score >= 8)
        ]

        if high_risks:
            result[subpillar] = ", ".join(high_risks)

        for _, row in high_numbers_df.iterrows():
            result[row["key_indicator"]] = f"{row['risk_score']} {row['unit']}"

    _write_js(viz_folder, "humanitarian_access_data.js", "HUMANITARIAN_ACCESS_DATA", dict(result))
    return dict(result)


def generate_key_sector_numbers(
    key_indicator_numbers_df: pd.DataFrame,
    viz_folder: str,
) -> dict:
    """
    Pick the largest reported number per (sector, unit) pair for high-scoring
    2d indicators, then group them all under "Sectoral Needs".
    """
    filtered = key_indicator_numbers_df[
        (key_indicator_numbers_df.task == "situation_analysis_2d")
        & (key_indicator_numbers_df.risk_score >= 8)
        & (key_indicator_numbers_df.number.apply(lambda x: isinstance(x, int) and x > 100))
        & ~(
            (key_indicator_numbers_df.unit == "people")
            & (key_indicator_numbers_df.number.apply(lambda x: isinstance(x, int) and x < 1_000))
        )
    ]

    rows = []
    for sector in filtered.sector.unique():
        sector_df = filtered[filtered.sector == sector]
        for unit in sector_df.unit.unique():
            max_val = sector_df[sector_df.unit == unit].number.max()
            best_row = (
                sector_df[(sector_df.unit == unit) & (sector_df.number == max_val)][
                    ["sector", "key_indicator", "number", "unit"]
                ]
                .head(1)
            )
            rows.append(best_row)

    table = (
        pd.concat(rows)
        .sort_values(by="sector")
        .drop_duplicates(subset=["sector", "key_indicator", "number", "unit"])
    )

    grouped = {"Sectoral Needs": []}
    for _, row in table.iterrows():
        grouped["Sectoral Needs"].append(
            {
                "key_indicator": row["key_indicator"],
                "number": None if pd.isna(row["number"]) else row["number"],
                "unit": row["unit"],
            }
        )

    _write_js(viz_folder, "key_sector_numbers_data.js", "KEY_NUMBERS_DATA", grouped)
    print(f"  ({len(table)} rows)")
    return grouped


def generate_current_hazards_and_threats(risks_df: pd.DataFrame, viz_folder: str) -> list:
    """Extract unique risks from the 'Hazard & Threats' subpillar."""
    data = risks_df[risks_df.subpillar == "Hazard & Threats"].risk.unique().tolist()
    _write_js(
        viz_folder,
        "current_hazards_and_threats_data.js",
        "CURRENT_HAZARDS_AND_THREATS_DATA",
        data,
    )
    print(f"  ({len(data)} items)")
    return data


def generate_precrisis_vulnerabilities(risks_df: pd.DataFrame, viz_folder: str) -> list:
    """Extract unique risks from the 'Underlying-Aggravating Factors' subpillar."""
    data = (
        risks_df[risks_df.subpillar == "Underlying-Aggravating Factors"]
        .risk.unique()
        .tolist()
    )
    _write_js(
        viz_folder,
        "precrisis_vulnerabilities_data.js",
        "PRECRISIS_VULNERABILITIES_DATA",
        data,
    )
    print(f"  ({len(data)} items)")
    return data


def generate_displacement_risks(risks_df: pd.DataFrame, viz_folder: str) -> dict:
    """
    For 'Push Factors' and 'Intentions' subpillars under Displacement, collect
    the top-4 risks by score.
    """
    disp_df = risks_df[risks_df.pillar == "Displacement"]
    data: dict = defaultdict(list)

    for subpillar in ["Push Factors", "Intentions"]:
        top4 = (
            disp_df[disp_df.subpillar == subpillar]
            .sort_values(by="risk_score", ascending=False)
            .head(4)
            .risk.unique()
            .tolist()
        )
        data[subpillar] = top4

    _write_js(viz_folder, "displacement_risks_data.js", "DISPLACEMENT_RISKS_DATA", dict(data))
    print(f"  ({len(data)} subpillars)")
    return dict(data)


def generate_top_sectoral_needs(
    priority_needs_df: pd.DataFrame, viz_folder: str
) -> dict:
    """Group critical (score >= 9) sectoral priority needs by sector."""
    top = priority_needs_df[
        (priority_needs_df.task == "situation_analysis_2d")
        & (priority_needs_df.priority_need_score >= 9)
    ].sort_values(by="priority_need_score", ascending=False)

    data: dict = defaultdict(list)
    for _, row in top.iterrows():
        data[row["sector"]].append(row["priority_need"])

    _write_js(
        viz_folder, "top_sectoral_needs_data.js", "TOP_SECTORAL_NEEDS_DATA", dict(data)
    )
    print(f"  ({len(data)} sectors)")
    return dict(data)


def generate_top_priority_interventions(
    priority_interventions_df: pd.DataFrame, viz_folder: str
) -> dict:
    """Group critical (score >= 9) priority interventions by sector (or pillar if no sector)."""
    top = priority_interventions_df[
        priority_interventions_df.priority_intervention_score >= 9
    ].sort_values(by="priority_intervention_score", ascending=False)

    data: dict = defaultdict(list)
    for _, row in top.iterrows():
        field = row["pillar"] if row["sector"] == "-" else row["sector"]
        data[field].append(row["priority_intervention"])

    _write_js(
        viz_folder,
        "top_priority_interventions_data.js",
        "TOP_PRIORITY_INTERVENTIONS_DATA",
        dict(data),
    )
    print(f"  ({len(data)} groups)")
    return dict(data)


def generate_displacement_numbers(numbers_df: pd.DataFrame, viz_folder: str) -> list:
    """
    Find high-confidence displacement counts for Lebanon and persist as JS.
    """
    filtered = numbers_df[
        (numbers_df["what_happened"] == "displaced")
        & (numbers_df["risk_score"] >= 9)
        & (numbers_df["number"] > 100)
        & (numbers_df["start_location"] == "Lebanon")
    ]
    merged = merge_entries_by_number(filtered)

    data = [{"number": row["number"], "unit": row["unit"]} for _, row in merged.iterrows()]
    _write_js(viz_folder, "displacement_data.js", "DISPLACEMENT_DATA", data)
    print(f"  ({len(data)} items)")
    return data


def generate_final_numbers(numbers_df: pd.DataFrame, viz_folder: str) -> list:
    """
    For each event type (displaced / killed / injured …), keep only the largest
    reported figure for Lebanon with risk score >= 8.
    """
    filtered = numbers_df[
        (numbers_df["risk_score"] >= 8)
        & (numbers_df["number"] > 100)
        & (numbers_df["start_location"] == "Lebanon")
    ]
    merged = merge_entries_by_number(filtered)

    data = []
    for what_happened in merged["what_happened"].unique():
        subset = merged[merged["what_happened"] == what_happened]
        max_num = subset["number"].max()
        best = subset[subset["number"] == max_num].iloc[0]
        data.append(
            {
                "what_happened": what_happened,
                "number": int(best["number"]),
                "unit": best["unit"],
            }
        )

    _write_js(viz_folder, "final_numbers_data.js", "FINAL_NUMBERS_DATA", data)
    print(f"  ({len(data)} event types)")
    return data


def generate_top_5_sources(
    answers_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    viz_folder: str,
) -> list:
    """
    Count how many times each source document is cited across all answers,
    and return the top-5 most-cited sources.
    """
    id_counts = (
        answers_df[["ID"]]
        .explode("ID")["ID"]
        .value_counts()
        .to_frame()
    )
    id_counts["ID"] = id_counts.index.astype(int)
    id_counts.reset_index(drop=True, inplace=True)

    merged = pd.merge(id_counts, classification_df, left_on="ID", right_on="Entry ID", how="inner")

    top5 = (
        merged.explode("Document Source")
        .groupby("Document Source", as_index=False)
        .agg({"count": "sum"})
        .sort_values(by="count", ascending=False)
        .head(5)["Document Source"]
        .tolist()
    )

    _write_js(viz_folder, "top_5_sources_data.js", "TOP_5_SOURCES_DATA", top5)
    print(f"  ({len(top5)} sources)")
    return top5


def generate_output_context_risks(
    context_figures: list, viz_folder: str
) -> dict:
    """
    Flatten context_figures JSON, filter for critical risks (score >= 9),
    and group by context pillar.
    """
    rows = []
    for pillar_entry in context_figures:
        pillar_name = pillar_entry["context_pillar"]
        for risk_item in pillar_entry["figures"]["key_risks"]:
            rows.append(
                {
                    "context_pillar": pillar_name,
                    "risk": risk_item["risk"],
                    "risk_score": risk_item["risk_score"],
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["risk_score"] >= 9]

    data: dict = defaultdict(list)
    for _, row in df.iterrows():
        data[row["context_pillar"]].append(row["risk"])

    _write_js(
        viz_folder, "output_context_risks_data.js", "OUTPUT_CONTEXT_RISKS_DATA", dict(data)
    )
    print(f"  ({len(data)} context pillars)")
    return dict(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_folder: str, viz_folder: str) -> None:
    print(f"Loading data from: {data_folder}")
    data = load_data(data_folder)

    answers_df = data["answers_df"]
    key_indicator_numbers_df = data["key_indicator_numbers_df"]
    risks_df = data["risks_df"]
    priority_needs_df = data["priority_needs_df"]
    priority_interventions_df = data["priority_interventions_df"]
    context_figures = data["context_figures"]
    numbers_df = data["numbers_df"]
    classification_df = data["classification_df"]

    print("\n--- Generating shown_risks ---")
    generate_shown_risks(risks_df, data_folder, viz_folder)

    print("\n--- Generating humanitarian_access_data ---")
    generate_humanitarian_access_data(risks_df, key_indicator_numbers_df, viz_folder)

    print("\n--- Generating key_sector_numbers ---")
    generate_key_sector_numbers(key_indicator_numbers_df, viz_folder)

    print("\n--- Generating current_hazards_and_threats ---")
    generate_current_hazards_and_threats(risks_df, viz_folder)

    print("\n--- Generating precrisis_vulnerabilities ---")
    generate_precrisis_vulnerabilities(risks_df, viz_folder)

    print("\n--- Generating displacement_risks ---")
    generate_displacement_risks(risks_df, viz_folder)

    print("\n--- Generating top_sectoral_needs ---")
    generate_top_sectoral_needs(priority_needs_df, viz_folder)

    print("\n--- Generating top_priority_interventions ---")
    generate_top_priority_interventions(priority_interventions_df, viz_folder)

    print("\n--- Generating displacement_numbers ---")
    generate_displacement_numbers(numbers_df, viz_folder)

    print("\n--- Generating final_numbers ---")
    generate_final_numbers(numbers_df, viz_folder)

    print("\n--- Generating top_5_sources ---")
    generate_top_5_sources(answers_df, classification_df, viz_folder)

    print("\n--- Generating output_context_risks ---")
    generate_output_context_risks(context_figures, viz_folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-folder",
        default="data/WestAsia2026/analysis/Lebanon/",
        help="Path to the analysis data folder",
    )
    parser.add_argument(
        "--viz-folder",
        default="src/viz/",
        help="Output folder for JS visualisation files",
    )
    args = parser.parse_args()
    run(data_folder=args.data_folder, viz_folder=args.viz_folder)


if __name__ == "__main__":
    main()
