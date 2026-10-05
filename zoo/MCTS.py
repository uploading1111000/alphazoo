from .formula import Formula
import numpy as np
from scipy.special import softmax
import random


class MCT_searcher:
    def __init__(self,backtester,model,zoo,B=3,b=1,prob_T=1,c=1,alph=0.5,k=1):
        self.backtester = backtester
        self.model = model
        self.zoo = zoo
        self.root = Formula(None)
        self.B = B
        self.b = b
        self.prob_T = prob_T
        self.c = c
        self.alph = alph
        self.k = k

    def start_new_tree(self,B=None,b=None,prob_T=None,c=None,alph=None,k=None):
        if B is not None:
            self.B = B
        if b is not None:
            self.b = b
        if prob_T is not None:
            self.prob_T = prob_T
        if c is not None:
            self.c = c
        if alph is not None:
            self.alph = alph
        if k is not None:
            self.k = k
        
        self.searches = 0
        self.budget = self.B

        self.root = Formula(None)
        portrait = self.model.generate_portrait
        self.root.add_pseudocode(portrait)

        code = self.model.generate_code(self.root.portrait)
        valid = self.root.add_code(code)
        for x in range(10):
            if valid != "valid":
                code = self.model.generate_code_retry(valid)
                valid = self.root.add_pseudocode(portrait)

        self.max_S = self.backtester.alpha_score(self.root)

    def explore_tree(self):
        expanded_node = self.UCT_traverse_recurse(self.root)
        probabilities = softmax(-expanded_node.ranks/self.prob_T)
        dimension_choice = random.choices(range(5),weights=probabilities,k=1)[0]
        examples = self.zoo.few_shot(expanded_node,dimension_choice)
        alpha_prompt = expanded_node.alpha()
        suggestion = self.model.generate_suggestion(
            alpha_prompt,
            examples,
            DIMENSIONS[dimension_choice]
        )
        refinement_portrait = self.model.generate_refinement(self,alpha_prompt,suggestion)
        new_formula = Formula(expanded_node)
        expanded_node.children.append(new_formula)
        new_formula.add_pseudocode(refinement_portrait)

        code = self.model.generate_code(new_formula)
        valid = self.root.add_code(code)
        for x in range(10):
            if valid != "valid":
                code = self.model.generate_code_retry(valid)
                valid = new_formula.add_pseudocode(refinement_portrait)

        new_S = self.backtester.alpha_score(new_formula)
        if new_S > self.max_S:
            self.max_S = new_S
            self.budget += self.b
        parent = expanded_node
        while parent is not None:
            parent.Q = max(parent.Q, new_S)
            parent = parent.parent
        
        self.searches += 1
        return self.searches == self.budget

    def UCT_traverse_recurse(self,formula):
        C = len(formula.children)
        N = formula.visits
        formula.visits += 1
        if C < self.k * (N ** self.alph):
            return formula
        else:
            ln = np.log(N)
            UCTS = [child.Q + self.c * np.sqrt(ln/child.visits) for child in formula.children]
            return self.UCT_traverse_recurse(formula.children[np.argmax(UCTS)])

DIMENSIONS = ["Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting Risk"]
