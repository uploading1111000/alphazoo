#TODO: add names for each comment below when used in backtest.py
#Needs formula for each member
#Needs past scores in each dimension for each member self.scores(dimension, formula)
#Needs past ranks in each dimension for each member self.ranks(dimension, formula)
#Needs predictions in each dimension for each member (diversity metric) - self.prediction_matrix(zoo_n,universe_n,target_n)
#Needs .add() taking in above 3 as arguments 
#Needs number of members - self.n
import numpy as np
from heapq import nsmallest, nlargest
from operator import itemgetter

class Alpha_Zoo():
    def __init__(self):
        pass

    def few_shot(self, formula, dimension_index):
        K = 3

        if dimension_index in [2,4]:
            return "No examples (This is a zero-shot task.)"

        abs_correlations = np.abs(self.prediction_correlations(formula.predictions))

        if dimension_index == 3:
            lowest_correlations = nsmallest(K,enumerate(abs_correlations),key=itemgetter(1))
            return "\n".join([member.alpha() for member in [self.formulas[i] for i in lowest_correlations]])

        median = np.median(abs_correlations)
        bottom_percent_indices = [i for i,corr in enumerate(abs_correlations) if corr < median]
        ranks = [rank if i in bottom_percent_indices else 0 for i,rank in enumerate(self.ranks[dimension_index])]
        highest_ranks = nlargest(K,enumerate(ranks),key=itemgetter(1))
        return "\n".join([member.alpha for member in [self.formulas[i] for i in highest_ranks]])
    
            

    def prediction_correlations(self,prediction_matrix):
        zoo_predictions = self.prediction_matrix
        
        reshaped_zoo = zoo_predictions.reshape((zoo_predictions.shape[0],self.universe_n*self.targets_n))
        reshaped_predictions = prediction_matrix.reshape((self.universe_n*self.targets_n))
        
        centered_predictions = reshaped_predictions - np.mean(reshaped_predictions)
        centered_zoo = reshaped_zoo - np.mean(reshaped_zoo,axis=1,keepdims=True)
        
        std_predictions = np.linalg.norm(centered_predictions)
        std_zoo = np.linalg.norm(centered_zoo,axis=1)
        
        return np.dot(centered_zoo,centered_predictions) / (std_predictions * std_zoo)
