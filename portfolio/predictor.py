import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from data import initialize_prediction_data, get_date_range_lookback
import joblib
PREDICTOR_FILE_PATH = "portfolio/predictor.pkl"

class Predictor:
    def __init__(self,zoo):
        self.formulas = zoo.formulas

    def train(self,forward,lookback,begin,end,k=None,n_alphas=50,cv=None):
        universe = get_date_range_lookback(begin,end,lookback,forward)
        initialize_prediction_data(self, universe, forward, lookback)
        targets = self.targets.flatten()
        prediction_matrix = []
        for formula in self.formulas:
            predictions = formula.apply(self.histories)[:,:self.targets_n]
            prediction_matrix.append(predictions.flatten())
        prediction_matrix = np.array(prediction_matrix).T
        del self.histories #cleanup for save
        del self.targets
        del universe
        print("Train set built")

        self.scaler = StandardScaler()
        scaled_predictions = self.scaler.fit_transform(prediction_matrix)
        del prediction_matrix
        print("Features scaled")

        regressor = RidgeCV()
        with joblib.parallel_config(
            backend="loky",
            max_nbytes=None,
            n_jobs=5
        ):
            self.SFS = SequentialFeatureSelector(
                regressor,
                n_features_to_select=n_alphas,
                cv=cv
            )
            selected_train = self.SFS.fit_transform(scaled_predictions, targets)
        del scaled_predictions
        print("Features selected")

        self.regressor = regressor.fit(selected_train,targets)
        print(f"Regressor fit. Score: {self.regressor.best_score_}")

        joblib.dump(self,PREDICTOR_FILE_PATH)

    def load():
        return joblib.load(PREDICTOR_FILE_PATH)

    def predict(self,histories):
        prediction_matrix = []
        for i, formula in enumerate(self.formulas):
            if self.SFS.support_[i]:
                prediction_matrix.append(formula.apply(histories)[:,:])
            else:
                prediction_matrix.append()
                
