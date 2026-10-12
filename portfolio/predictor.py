import numpy as np
from scipy.stats import rankdata
from lightgbm import LGBMRegressor
from data import initialize_prediction_data, get_date_range_lookback
import joblib
import pandas as pd
PREDICTOR_FILE_PATH = "portfolio/predictor.pkl"

class Predictor:
    def __init__(self,zoo):
        self.formulas = zoo.formulas

    def train(self,forward,lookback,begin,end,n_alphas=50):
        """Fit on all dates in [begin, end), using the top n_alphas by RankIR.

        Use begin=2020-01-01 and end=2025-01-01 for the 2020–2024
        training period. No validation split or early stopping is used.
        """
        if isinstance(n_alphas, bool) or not isinstance(n_alphas, (int, np.integer)) or n_alphas <= 0:
            raise ValueError("n_alphas must be a positive integer")
        eligible = [formula for formula in self.formulas if np.isfinite(formula.scores[1])]
        if not eligible:
            raise ValueError("No formulas have a finite RankIR score")
        self.selected_formulas = sorted(
            eligible, key=lambda formula: formula.scores[1], reverse=True
        )[:n_alphas]
        print(f"Selected {len(self.selected_formulas)} features by RankIR")

        universe = get_date_range_lookback(begin,end,lookback,forward)
        initialize_prediction_data(self, universe, forward, lookback)
        targets = self.rank_normalize(self.targets).flatten()
        prediction_matrix = []
        for formula in self.selected_formulas:
            predictions = formula.apply(self.histories)[:,:self.targets_n]
            prediction_matrix.append(self.rank_normalize(predictions).flatten())
        prediction_matrix = np.array(prediction_matrix).T
        del self.histories #cleanup for save
        del self.targets
        del universe
        print("Train set built")

        self.regressor = LGBMRegressor(
            num_leaves=32,
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
        )

        # Fixed boosting budget on the full training set; no eval_set or callbacks.
        self.regressor.fit(prediction_matrix,targets)

        joblib.dump(self,PREDICTOR_FILE_PATH)
        print("Regressor fit.")

    @staticmethod
    def load():
        return joblib.load(PREDICTOR_FILE_PATH)

    @staticmethod
    def rank_normalize(values):
        """Center per-date percentile ranks across tickers: (rank - 0.5) / n - 0.5.

        Ties receive average ranks; nonfinite values remain missing (NaN).
        A constant cross-section or a single valid ticker maps to zero.
        """
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        ranks = rankdata(np.where(finite, values, np.nan), axis=0, method="average", nan_policy="omit")
        counts = finite.sum(axis=0, keepdims=True)
        return (ranks - 0.5) / np.maximum(counts, 1) - 0.5

    def predict(self,histories):
        if not getattr(self, "selected_formulas", None) or not hasattr(self, "regressor"):
            raise ValueError("Retrain the predictor before predicting")
        selected_predictions = [formula.apply(histories) for formula in self.selected_formulas]

        output_shape = (
            selected_predictions[0].shape[0],
            min(prediction.shape[1] for prediction in selected_predictions),
        )
        slices = (slice(None), slice(0, output_shape[1]))
        selected_features = [
            self.rank_normalize(prediction[slices]).reshape(-1)
            for prediction in selected_predictions
        ]

        prediction_matrix = np.column_stack(selected_features)
        predictions = self.regressor.predict(prediction_matrix)
        return predictions.reshape(output_shape)

    def test_correlate(self):
        universe = get_date_range_lookback(
            pd.to_datetime("2025-01-01"),
            pd.to_datetime("2026-07-01"),
            90,
            20)
        initialize_prediction_data(self, universe, 20, 90)
        predictions = self.predict(self.histories)[:,:self.targets_n]
        R_i = np.zeros((self.targets_n))
        for day in range(self.targets_n):
            R_i[day] = np.corrcoef(self.rank_normalize(self.targets[:,day]),self.rank_normalize(predictions[:,day]))[0,1]
        print(np.mean(R_i))
