import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import initialize_prediction_data, get_date_range_lookback

class Predictor:
    def __init__(self,zoo,forward,lookback,begin,end):
        self.zoo = zoo
        universe = get_date_range_lookback(begin,end,lookback,forward)
        initialize_prediction_data(self, universe, forward, lookback)

    def train(self,k=None):
        for formula in self.zoo.formulas:
            predictions = formula.apply(self.histories)

        regressor = RidgeCV()


                