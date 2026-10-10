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
        selected = self.SFS.get_support()
        selected_predictions = []
        for formula in (f for f, keep in zip(self.formulas, selected) if keep):
            selected_predictions.append(formula.apply(histories))

        if not selected_predictions:
            raise ValueError("The trained predictor has no selected formulas")

        output_shape = tuple(
            min(prediction.shape[axis] for prediction in selected_predictions)
            for axis in range(selected_predictions[0].ndim)
        )
        slices = tuple(slice(0, size) for size in output_shape)
        selected_features = [
            prediction[slices].reshape(-1)
            for prediction in selected_predictions
        ]

        # Training scaled all formula columns before selecting a subset. Apply the
        # matching saved scaler parameters to the selected columns at inference.
        prediction_matrix = np.column_stack(selected_features)
        selected_scaled = (
            prediction_matrix - self.scaler.mean_[selected]
        ) / self.scaler.scale_[selected]
        predictions = self.regressor.predict(selected_scaled)
        return predictions.reshape(output_shape)
                
