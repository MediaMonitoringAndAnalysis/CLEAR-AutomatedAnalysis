from ast import literal_eval
from tqdm import tqdm
import json
import os
import re
from openai import OpenAI
from typing import Any, List


SYSTEM_PROMPT = """
You are a humanitarian analyst. You are given a list of context questions and a country name.
You need to answer the questions based on the context of the country over the past 1-2 years.
Answer as a complete, precise, and synthesised answer to the questions in markdown format (use paragraphs and line breaks for clarity). Only include information relevant to the questions.
If you don't know the answer, return "-".

return a JSON object with the following keys:
- "answer": the answer to the questions in markdown format
- "key_risks": a list of key risks identified in the country over the past 1-2 years with the following keys:
  - "risk": the name of the risk
  - "risk_score": a score from 0 to 10 indicating the severity of the risk and its impact on the current situation
"""

base_answer = {
    "answer": "-",
    "key_risks": [],
}

context_questions = {
    "Demographics": "How many people live in the area? What are their demographic characteristics (population pyramid)? What was the impact of previous displacements on demography? How many people currently live in the geographical area? ",
    "Political": "Who are the governing authorities (both formal and unformal), what are the governance modalities? Is the political system democratically elected and representative? What are the main opposition parties? Is there political freedom? How peaceful have been past elections? Who are the main gatekeepers? ",
    "Economy": "What is the economy mostly based on? What are the main products imported and exported? Have there been recent significant volatilities in the market (e.g. significant rises in cost of food basket, drop in main exports (e.g. garment industry due to COVID)) ",
    "Socio culture": "What is the social composition of the population (ethnicity, languages, minorities, etc.). What is the role of men and women in the society? What is the role of young and elders in the society? Are there specific cultural attitudes and practice that have implications for programming (prevalence of early marriage, child labour, attitude to education etc.)? ",
    "Security": "What is the security context (banditry, criminality, robbery, injuries, fatalities)? Who enforces security? What risks exists for nationals and international staff? How is the security infrastructure of the country (centralized, fragmented)? Are there any sanctioned groups operating in areas? ",
    "Legal and policy": "what are the regulatory systems in place (customary, informal, community based, religious, etc.)? Is there equal access? Are they functioning (efficiency, timeliness, reliability, impartiality), how is law enforced? Are there specific groups that have limited rights/access to the support of law? Are there any local CT measures criminalizing humanitarian aid ?  ",
    "Infrastructure & technology": "What is the state of public and private infrastructures? What internet and mobile phone coverage is available in affected areas? How easy/difficult is electronic money transfer? ",
    "Environment": "What are the climate conditions (temperature, precipitations, wind)? is air/Soil polluted? What environmental laws are in place? Will specific seasonal factors have a significant impact and when? (e.g. rainy season, lean season). ",
    "Humanitarian Coordination": "Is the Cluster system activated?  Who is leading IDPs/refugees coordinating mechanisms? What are lines of leadership, coordination, and accountability for IDPs/Refugees? What government disaster management/refugee agencies exist? What role and relationship does your organisation have with these structures and bodies? ",
}


def _extract_and_evaluate_first(string, default_response):
    start_char = str(default_response)[0]
    end_char = str(default_response)[-1]
    escaped_start = re.escape(start_char)
    escaped_end = re.escape(end_char)

    # Build the regex pattern using the specified start and end characters
    pattern = rf"{escaped_start}.*{escaped_end}"

    # Extract the substring using the regex pattern
    result = re.search(pattern, string)

    # Check if the result was found and return it
    if result:
        return result.group(0)
    else:
        return str(default_response)


def _remove_commas_between_numbers(text):
    # This pattern matches commas between two digits
    pattern = r"(?<=\d),(?=\d)"
    # Replace matched commas with an empty string
    return re.sub(pattern, "", text)


def replace_unneeded_characters(s):
    return (
        s.replace("```", "")
        .replace("json\n", "")
        .replace("json", "")
        .replace("\n{", "{")
        .replace("}\n", "}")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\\xa0", "\\u00A0")
        # .strip()
    )


def _posprocess_gpt_output(s, default_response):
    # Remove trailing commas from objects and arrays
    s = re.sub(r",(\s*[}\]])", r"\1", s).strip()
    s = replace_unneeded_characters(s)
    s = _remove_commas_between_numbers(s)

    s = _extract_and_evaluate_first(s, default_response)

    return s


def postprocess_structured_output(output_text: str, default_response):
    output_text = _posprocess_gpt_output(output_text, default_response)
    try:
        gpt_extracted_infos = literal_eval(output_text)
    except:
        # print("literal_eval failed", output_text)
        try:
            gpt_extracted_infos = json.loads(output_text)
        except Exception as e:
            print("literal eval and json.loads failed", output_text)
            # st.markdown("Error in extracting structured data", e)
            # st.markdown(output_text)
            gpt_extracted_infos = default_response

    return gpt_extracted_infos

class OpenAIPipeline:
    name = "openai"
    provider = "openai"
    required_env_vars = ("openai_api_key",)

    def __init__(
        self,
        model: str,
        timeout: int = 60,
    ):
        self.model = model
        self.timeout = timeout

    def invoke(self, user_prompt: str, system_prompt: str) -> Any:
        api_key = os.getenv("openai_api_key")
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

        # Accept system/user prompt split
        # Build the request payload for Responses API
        args = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            # "temperature": 0,
            "tools": [{"type": "web_search"}],
            "reasoning": {"effort": "low"},
        }

        # if supports_structured:
        #     args["response_format"] = {
        #         "type": "json_schema",
        #         "json_schema": {
        #             "name": "DisplacementFiguresResponse",
        #             "schema": DisasterFiguresResponse.model_json_schema()
        #         }
        #     }

        response = client.responses.create(**args)

        # price = _calculate_price(response, self.model)

        content = response.output[-1].content[0]

        citations = content.annotations
        text = content.text
        cleaned_text = postprocess_structured_output(text, base_answer)

        # Convert AnnotationURLCitation objects to JSON-serializable format
        serializable_citations = []
        if citations:
            for citation in citations:
                final_citation = {
                    "title": citation.title,
                    "url": citation.url,
                    "start_index": citation.start_index,
                    "end_index": citation.end_index,
                    "type": citation.type,
                }
                serializable_citations.append(final_citation)

        final_output = {
            "figures": cleaned_text,
            "_citations": serializable_citations,
            # "price": price,
        }

        return final_output


def _generate_context_figures(country: str) -> List[dict]:
    qa_pipeline = OpenAIPipeline(model="gpt-5-mini")
    outputs = []

    for context_pillar, one_pillar_questions in tqdm(context_questions.items(), desc=f"Generating context for {country}"):
        user_prompt = f"Answer the following questions for {country}: {one_pillar_questions}"
        system_prompt = SYSTEM_PROMPT
        context = qa_pipeline.invoke(user_prompt, system_prompt)

        outputs.append({
            "context_pillar": context_pillar,
            "figures": context["figures"],
            "citations": context["_citations"],
        })

    return outputs
        

def generate_context(country: str, output_path: os.PathLike):
    if os.path.exists(output_path):
        return

    context_figures = _generate_context_figures(country)
    with open(output_path, "w") as f:
        json.dump(context_figures, f, indent=4)