import llm
from llm.router import FORMULA_EXAMPLE, PSEUDOCODE_EXAMPLE

with llm.model() as model:
    model.generate_portrait()
    model.overfitting_risk(f"Portrait: {PSEUDOCODE_EXAMPLE} \nImplemented formula: {FORMULA_EXAMPLE}", "This formula has no refinement history. (It has been generated from scratch)")

