from .formula import Formula
from data import Alpha_Backtester
import numpy as np
from heapq import nsmallest, nlargest
from operator import itemgetter
import joblib
import copy
from constants import CUTOFF
ZOO_FILE_PATH = "alpha_zoo/zoo.pkl"
N_starters = 10

class Alpha_Zoo():
    def create_starters(self,forward,lookback,begin,end):
        starter_backtester = Alpha_Backtester(
            self.model,self,forward,lookback,begin,end)
        print("Loaded starter backtester")
        self.prediction_matrix = np.ndarray(
            (0,starter_backtester.universe_n,starter_backtester.targets_n))
        for i in range(N_starters):
            formula = Formula(None)
            portrait = self.model.generate_portrait()
            formula.add_pseudocode(portrait)

            code = self.model.generate_code(formula.portrait)
            valid = formula.add_code(code)
            for x in range(10):
                if valid != "valid":
                    code = self.model.generate_code_retry(valid)
                    valid = formula.add_code(code)
                else:
                    break
            else:
                if valid != "valid":
                    raise RuntimeError("Failed to generate code after 10 retries")

            scores = starter_backtester.score_all(formula)
            predictions = starter_backtester._most_recent_predictions
            formula.set_atts(scores,None,None,predictions)
            self.add(formula.zoo_member())
            print(f"Starter {i+1} generated!")

    def add(self,formula):
        self.formulas.append(formula)
        reshaped_scores = np.reshape(formula.scores,(5,1))
        reshaped_predictions = np.reshape(formula.predictions,(1,formula.predictions.shape[0],formula.predictions.shape[1]))
        
        self.scores = np.concatenate((self.scores,reshaped_scores),axis=1)
        self.prediction_matrix = np.concatenate((self.prediction_matrix,reshaped_predictions),axis=0)
        self.n += 1

    def remove_starters(self):
        self.formulas = self.formulas[N_starters:]
        self.scores = self.scores[:,N_starters:]
        self.prediction_matrix = self.prediction_matrix[N_starters:]
        self.n = self.n - N_starters

    def save(self,path=None):
        if path is not None:
            _path = path
        else:
            _path = ZOO_FILE_PATH
        no_model = copy.copy(self)
        no_model.model = None
        joblib.dump(no_model,_path)

    def load(path=None):
        if path is not None:
            _path = path
        else:
            _path = ZOO_FILE_PATH
        return joblib.load(_path)

    def __init__(self,model):
        self.model = model
        self.formulas = []
        self.prediction_matrix = np.ndarray((0,0,0))
        self.scores = np.ndarray((5,0))
        self.n = 0

    def few_shot(self, formula, dimension_index):
        K = 3

        if dimension_index == 2:
            return "No examples (This is a zero-shot task.)"
        if dimension_index == 4:
            return f"No examples (This is a zero-shot task.) \n Overfitting Feedback: \n {formula.ofr_reviews[0]}"

        abs_correlations = np.abs(self.prediction_correlations(formula.predictions))

        if dimension_index == 3:
            lowest_correlations = nsmallest(K,enumerate(abs_correlations),key=itemgetter(1))
            return "\n".join([member.alpha() for member in [self.formulas[i] for i,_ in lowest_correlations]])

        median = np.median(abs_correlations)
        bottom_percent_indices = set([i for i,corr in enumerate(abs_correlations) if corr < median])
        scores = [score if i in bottom_percent_indices else -np.inf for i,score in enumerate(self.scores[dimension_index])]
        highest_ranks = nlargest(K,enumerate(scores),key=itemgetter(1))
        return "\n".join([member.alpha() + f"\nScore:{member.scores[dimension_index]}" for member in [self.formulas[i] for i,_ in highest_ranks]])

    def prediction_correlations(self,prediction_matrix):
        return correlations(prediction_matrix,self.prediction_matrix)

def correlations(predictions,other_predictions):
    universe_n = other_predictions.shape[1]
    target_n = other_predictions.shape[2]
    reshaped_other = other_predictions.reshape((other_predictions.shape[0],universe_n*target_n))
    reshaped_predictions = predictions.reshape((universe_n*target_n))
            
    centered_predictions = reshaped_predictions - np.mean(reshaped_predictions)
    centered_other = reshaped_other - np.mean(reshaped_other,axis=1,keepdims=True)
            
    std_predictions = np.linalg.norm(centered_predictions)
    std_other = np.linalg.norm(centered_other,axis=1)
            
    return np.dot(centered_other,centered_predictions) / (std_predictions * std_other)
