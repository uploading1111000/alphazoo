from openrouter import OpenRouter
from dotenv import dotenv_values

class model():
    def __init__(self,model_name="z-ai/glm-5.3-flash"):
        self.model_name = model_name
        self.api_key = dotenv_values(".env")["OPENROUTER_API_KEY"]
        self.client = OpenRouter(api_key = self.api_key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(self, exc_type, exc_val, exc_tb)