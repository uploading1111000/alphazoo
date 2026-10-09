from .formula import Formula
from .zoo import correlations
import numpy as np
from scipy.special import softmax
import random
from constants import CUTOFF

def merge(lists):
    return_list = []
    pointers = [0 for l in lists]
    while np.any([pointers[i] < len(lists[i]) for i in range(len(lists))]):
        i = np.argmin([lists[i][pointers[i]][0] if pointers[i] < len(lists[i]) else np.inf for i in range(len(lists))])
        return_list.append(lists[i][pointers[i]])
        pointers[i] += 1
    return return_list

class MCT_searcher:
    def __init__(self,backtester,model,zoo,B=3,b=1,prob_T=1,c=1):
        self.backtester = backtester
        self.model = model
        self.zoo = zoo
        self.root = Formula(None)
        self.B = B
        self.b = b
        self.prob_T = prob_T
        self.c = c

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
        portrait = self.model.generate_portrait()
        self.root.add_pseudocode(portrait)

        code = self.model.generate_code(self.root.portrait)
        valid = self.root.add_code(code)
        for x in range(10):
            if valid != "valid":
                code = self.model.generate_code_retry(valid)
                valid = self.root.add_code(code)
            else:
                break
        else:
            if valid != "valid":
                raise RuntimeError("Failed to create code after 10 retries")

        self.max_S = self.backtester.alpha_score(self.root)

    def explore_tree(self):
        expanded_node = self.UCT_traverse_recurse(self.root)
        scores = np.asarray(expanded_node.scores,dtype=float)
        modified_ranks = np.where(scores < CUTOFF, 0.0, expanded_node.ranks)
        probabilities = softmax(-modified_ranks/self.prob_T)
        dimension_choice = random.choices(range(5),weights=probabilities,k=1)[0]
        examples = self.zoo.few_shot(expanded_node,dimension_choice)
        alpha_prompt = expanded_node.alpha()
        suggestion = self.model.generate_suggestion(
            alpha_prompt,
            examples,
            DIMENSIONS[dimension_choice],
            score_prompt(scores[dimension_choice],dimension_choice)
        )
        depth = 0
        ancestor = expanded_node
        while ancestor.parent is not None:
            depth += 1
            ancestor = ancestor.parent
        refinement_portrait = self.model.generate_refinement(
            alpha_prompt, suggestion, max_parameters=4 + depth
        )
        new_formula = Formula(expanded_node)
        expanded_node.children.append(new_formula)
        new_formula.add_pseudocode(refinement_portrait)

        code = self.model.generate_code(new_formula.portrait)
        valid = new_formula.add_code(code)
        for x in range(10):
            if valid != "valid":
                code = self.model.generate_code_retry(valid)
                valid = new_formula.add_code(code)
            else:
                break
        else:
            if valid != "valid":
                raise RuntimeError("Failed to create code after 10 retries")

        new_S = self.backtester.alpha_score(new_formula)
        if new_S > self.max_S:
            self.max_S = new_S
            self.budget += self.b
        parent = expanded_node
        while parent is not None:
            parent.Q = max(new_S,parent.Q)
            parent = parent.parent
        
        self.searches += 1
        return self.searches == self.budget

    def UCT_traverse_recurse(self,formula):
        C = len(formula.children)
        N = formula.visits
        formula.visits += 1
        ln = np.log(N)
        UCTS = [formula.a_score + self.c * np.sqrt(ln/(1+C))] + [child.Q + self.c * np.sqrt(ln/child.visits) for child in formula.children]
        index = np.argmax(UCTS)
        if index == 0:
            return formula
        else:
            return self.UCT_traverse_recurse(formula.children[index-1])

    def a_score_traverse_recurse(self,formula):
        recurse = [self.a_score_traverse_recurse(child) for child in formula.children] + [[(formula.a_score,formula)]]
        return merge(recurse)

    def add_effective(self):
        ordered = self.a_score_traverse_recurse(self.root)[::-1]
        added_predictions = None
        num = 0
        for pair in ordered:
            alpha = pair[1]
            if alpha.scores[0] < CUTOFF[0] or alpha.ranks[0] < 0.05:
                print(f"Failed effectiveness. Score: {alpha.scores[0]}, Ranks: {alpha.ranks[0]}")
                continue
            if alpha.scores[1] < CUTOFF[1] or alpha.ranks[1] < 0.05:
                print(f"Failed stability. Score: {alpha.scores[1]}, Ranks: {alpha.ranks[1]}")
                continue
            if alpha.scores[2] < CUTOFF[2]:
                print(f"Failed turnover. Score: {-alpha.scores[2]}")
                continue
            if added_predictions is not None:
                new_corrs = np.abs(correlations(alpha.predictions,added_predictions))
                max_corr = max(-alpha.scores[3],np.max(new_corrs))
                if -max_corr > CUTOFF[3]:
                    num += 1
                    self.zoo.add(alpha)
                    reshaped_predictions = np.reshape(alpha.predictions,(1,alpha.predictions.shape[0],alpha.predictions.shape[1]))
                    added_predictions = np.concatenate((added_predictions,reshaped_predictions),axis=0)
                else:
                    print(f"Failed diversity. Score: {max_corr}")
            else:
                max_corr = alpha.scores[3]
                if max_corr > CUTOFF[3]:
                    num += 1
                    self.zoo.add(alpha)
                    added_predictions = np.reshape(alpha.predictions,(1,alpha.predictions.shape[0],alpha.predictions.shape[1]))
                else:
                    print(f"Failed diversity. Score: {-max_corr}")
        return num
            
def score_prompt(score,dim):
    if dim in [0,1]:
        return f"{score} (Cutoff for acceptance is >{CUTOFF[dim]})"
    if dim in [2,3]:
        return f"{-score} (Cutoff for acceptance is <{-CUTOFF[dim]})"
    return str(score)



DIMENSIONS = ["Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting Risk"]
