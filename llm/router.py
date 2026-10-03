from openrouter import OpenRouter
from dotenv import dotenv_values
import uuid
from zoo.operators import AVAILABLE_OPERATORS_STRING
from data.data_scrape import AVAILABLE_FIELDS_STRING
import json


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
    def __init__(self,model_name="z-ai/glm-5.3-flash:floor"):
        self.model_name = model_name
        self.api_key = dotenv_values(".env")["OPENROUTER_API_KEY"]
        self.client = OpenRouter(api_key = self.api_key)
        self.sessionid = str(uuid.uuid4())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def generate_portrait(self): #rewrite like overfitting_risk
        prompt = PORTRAIT_PROMPT.format(
            available_operators=AVAILABLE_OPERATORS_STRING,
            available_fields=AVAILABLE_FIELDS_STRING
            ) + PSEUDOCODE_EXAMPLE
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
        for x in range(10):
            res = self.client.chat.send(
                messages = messages,
                stream=False,
                reasoning={"enabled": False},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=portrait_T
            )
            content = res.choices[0].message.content
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                print(f"Portrait prompt failed {x+1} time(s).")
                continue
            if type(returnable) is dict:
                if "name" in returnable and "description" in returnable and "pseudocode" in returnable:
                    return returnable
            print(f"Portrait prompt failed {x+1} time(s).")
            
        else:
            raise RuntimeError("Failed to generate valid portrait prompt after 10 tries")

    def generate_code(self, portrait_prompt, window=90):
        self.retries = 0
        prompt = PORTRAIT_PROMPT.format(
            available_operators=AVAILABLE_OPERATORS_STRING,
            available_fields=AVAILABLE_FIELDS_STRING,
            alpha_portrait_prompt=portrait_prompt,
            window_size = str(window)
        ) + FORMULA_EXAMPLE
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
        for x in range(10):
            res = self.client.chat.send(
                messages = messages,
                stream=False,
                reasoning={"enabled": False},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=formula_T
            )
            content = res.choices[0].message.content
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                print(f"Code prompt failed {x+1} time(s).")
                continue
            if type(returnable) is dict:
                if "formula" in returnable and "arguments" in returnable:
                    self._last_code_messages = messages + [{
                        "content": content,
                        "role": "assistant"
                    }]
                    return returnable
            print(f"Code prompt failed {x+1} time(s).")
        else:
            raise RuntimeError("Failed to generate valid code prompt after 10 tries")

    def generate_code_retry(self, failiure_reason):
        self.retries += 1
        print(f"Code retry! ({self.retries + 1})")
        prompt = f"Code was valid JSON but failed for following reason:\n{failiure_reason}\nTry again:"
        messages = self._last_code_messages + [{
            "content": prompt,
            "role": "user"
        }]
        for x in range(10):
            res = self.client.chat.send(
                messages = messages,
                stream=False,
                reasoning={"enabled": False},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=illegal_T
            )
            content = res.choices[0].message.content
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                print(f"Code prompt failed {x+1} time(s).")
                continue
            if type(returnable) is dict:
                if "formula" in returnable and "arguments" in returnable:
                    self._last_code_messages = messages + [{
                        "content": content,
                        "role": "assistant"
                    }]
                    return returnable
            print(f"Code prompt failed {x+1} time(s).")
        else:
            raise RuntimeError("Failed to generate valid code prompt after 10 tries")
        
    def overfitting_risk(self,alpha_formula,refinement_history):
        prompt = OVERFITTING_PROMPT.format(
            alpha_formula=alpha_formula,
            refinement_history=refinement_history,
            available_operators=AVAILABLE_OPERATORS_STRING,
            available_fields=AVAILABLE_FIELDS_STRING
            ) + OVERFITTING_EXAMPLES
        valid = False
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
        for x in range(10):
            res = self.client.chat.send(
                messages = messages,
                stream=False,
                reasoning={"enabled": False},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=overfit_T
            )
            content = res.choices[0].message.content
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                continue
            if type(returnable) is dict:
                if "reason" in returnable and "score" in returnable:
                    score = returnable["score"]
                    if type(score) is int:
                        if score <= 10 and score >= 0:
                            return returnable
        else:
            raise RuntimeError("Failed to generate valid overfitting score after 10 tries")
        
        
