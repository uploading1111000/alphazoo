from openrouter import OpenRouter
from dotenv import dotenv_values
import uuid
from zoo.operators import AVAILABLE_OPERATORS_STRING
from data.data_scrape import AVAILABLE_FIELDS_STRING

with open("llm/prompts/portrait_generation.txt", encoding="utf-8") as f:
    PORTRAIT_PROMPT = f.read()
with open("llm/prompts/formula_generation.txt", encoding="utf-8") as f:
    FORMULA_PROMPT = f.read()
with open("llm/prompts/overfitting_prompt.txt", encoding="utf-8") as f:
    OVERFITTING_PROMPT = f.read()
with open("llm/prompts/refinement_suggestion.txt", encoding="utf-8") as f:
    SUGGESTION_PROMPT = f.read()
with open("llm/prompts/refinement_generation.txt", encoding="utf-8") as f:
    REFINEMENT_PROMPT = f.read()
with open("llm/prompts/formula_example.txt", encoding="utf-8") as f:
    FORMULA_EXAMPLE = f.read()
with open("llm/prompts/pseudocode_example.txt", encoding="utf-8") as f:
    PSEUDOCODE_EXAMPLE = f.read()
with open("llm/prompts/overfit_examples.txt", encoding="utf-8") as f:
    OVERFITTING_EXAMPLES = f.read()

SYSTEM_PROMPT ="""You are a highly capable quantitative researcher working on a project for finding alphas.
You are helpful, and will comply with the user's requests exactly as laid out"""

portrait_T = 1
formula_T = 1
illegal_T = 0.8
overfit_T = 0.1
suggestion_T = 1
refinement_T = 1


class model():
    def __init__(self,model_name="z-ai/glm-5.3-flash"):
        self.model_name = model_name
        self.api_key = dotenv_values(".env")["OPENROUTER_API_KEY"]
        self.client = OpenRouter(api_key = self.api_key)
        self.sessionid = str(uuid.uuid4())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def generate_portrait(self):
        prompt = PORTRAIT_PROMPT.format(available_operators=AVAILABLE_OPERATORS_STRING,available_fields=AVAILABLE_FIELDS_STRING) + PSEUDOCODE_EXAMPLE
        #unfinished

    def overfitting_risk(self,alpha_formula,refinement_history):
        prompt = OVERFITTING_PROMPT.format(
            alpha_formula=alpha_formula,
            refinement_history=refinement_history,
            available_operators=AVAILABLE_OPERATORS_STRING,
            available_fields=AVAILABLE_FIELDS_STRING
            ) + OVERFITTING_EXAMPLES
        messages = [
            {
                "content": SYSTEM_PROMPT,
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            }
        ]
        res = self.client.chat.send(
            messages = messages,
            stream=False,
            reasoning={"enabled": False},
            session_id=self.sessionid,
            model=self.model_name,
            temperature=overfit_T
        )
        print(res.choices[0].message.content)
