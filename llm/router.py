from openrouter import OpenRouter
from openrouter.errors.toomanyrequestsresponse_error import TooManyRequestsResponseError
from dotenv import dotenv_values
import uuid
import time
from alpha_zoo.operators import AVAILABLE_OPERATORS_STRING
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

portrait_T = 1.2
formula_T = 1
illegal_T = 0.8
overfit_T = 0.1
suggestion_T = 1
refinement_T = 1


class Model():
    def __init__(self,model_name="z-ai/glm-5.3-flash"):
        self.model_name = model_name
        self.api_key = dotenv_values(".env")["OPENROUTER_API_KEY"]
        self.client = OpenRouter(api_key = self.api_key)
        self.sessionid = str(uuid.uuid4())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def _send_chat(self, **kwargs):
        for attempt in range(5):
            try:
                return self.client.chat.send(**kwargs)
            except TooManyRequestsResponseError as exc:
                if attempt == 4:
                    raise
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                print(f"OpenRouter rate limit reached; retrying in {delay:g}s.")
                time.sleep(delay)

    def generate_suggestion(self,alpha_formula,few_shot,dimension,alpha_score):
        prompt = SUGGESTION_PROMPT.format(
            alpha_formula = alpha_formula,
            dimension_of_interest=dimension,
            few_shot_examples=few_shot,
            alpha_score=alpha_score
        )
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
        res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "high"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=suggestion_T
        )
        return res.choices[0].message.content

    def generate_refinement(self,alpha_formula,suggestions,max_parameters):
        prompt = REFINEMENT_PROMPT.format(
            available_fields=AVAILABLE_FIELDS_STRING,
            available_operators=AVAILABLE_OPERATORS_STRING,
            origin_alpha_formula=alpha_formula,
            refinement_suggestions=suggestions,
            max_parameters=max_parameters
        )
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
            res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "high"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=refinement_T
            )
            content = res.choices[0].message.content
            if content is None:
                continue
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                print(f"Refinement prompt failed {x+1} time(s).")
                continue
            if type(returnable) is dict:
                if "name" in returnable and "description" in returnable and "pseudo_code" in returnable:
                    return returnable
            print(f"Refinement prompt failed {x+1} time(s).")
            
        else:
            raise RuntimeError("Failed to generate valid refinement prompt after 10 tries")
    

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
            res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "high"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=portrait_T
            )
            content = res.choices[0].message.content
            if content is None:
                continue
            try:
                returnable = json.loads(content)
            except json.decoder.JSONDecodeError:
                print(f"Portrait prompt failed {x+1} time(s).")
                continue
            if type(returnable) is dict:
                if "name" in returnable and "description" in returnable and "pseudo_code" in returnable:
                    return returnable
            print(f"Portrait prompt failed {x+1} time(s).")
            
        else:
            raise RuntimeError("Failed to generate valid portrait prompt after 10 tries")

    def generate_code(self, portrait_prompt, window=90):
        self.retries = 0
        prompt = FORMULA_PROMPT.format(
            available_operators=AVAILABLE_OPERATORS_STRING,
            available_fields=AVAILABLE_FIELDS_STRING,
            alpha_portrait_prompt=portrait_prompt,
            window_range = str(window)
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
            res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "low"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=formula_T
            )
            content = res.choices[0].message.content
            if content is None:
                continue
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
        prompt = f"Code was valid JSON but failed for following reason:\n{failiure_reason}\nTry again:"
        messages = self._last_code_messages + [{
            "content": prompt,
            "role": "user"
        }]
        for x in range(10):
            res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "low"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=illegal_T
            )
            content = res.choices[0].message.content
            if content is None:
                continue
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
            res = self._send_chat(
                messages = messages,
                stream=False,
                reasoning={"effort": "low"},
                session_id=self.sessionid,
                model=self.model_name,
                temperature=overfit_T
            )
            content = res.choices[0].message.content
            if content is None:
                continue
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
        
        
