import numpy as np
from scipy.stats import rankdata
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import initialize_prediction_data, get_date_range_lookback
import joblib
import pandas as pd
PREDICTOR_FILE_PATH = "portfolio/predictor.pkl"

class Predictor:
    def __init__(self,zoo):
        self.formulas = zoo.formulas

    def train(self, forward, lookback, begin="2020-01-01", end="2025-01-01",
              n_alphas=50, *, validation_begin="2024-07-01", max_epochs=100,
              random_state=42):
        """Fit on [begin, validation_begin), validate on [validation_begin, end).

        Dates refer to forward-return outcomes, as in the shared data loader.
        Features and targets are cross-sectional centered percentile ranks.
        Stored zoo RankIR scores determine selection; they are not recomputed.
        """
        for name, value in (("n_alphas", n_alphas), ("max_epochs", max_epochs)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        eligible = [formula for formula in self.formulas if np.isfinite(formula.scores[1])]
        if not eligible:
            raise ValueError("No formulas have a finite RankIR score")
        self.selected_formulas = sorted(
            eligible, key=lambda formula: formula.scores[1], reverse=True
        )[:n_alphas]
        print(f"Selected {len(self.selected_formulas)} features by RankIR")

        begin, validation_begin, end = map(pd.Timestamp, (begin, validation_begin, end))
        if not begin < validation_begin < end:
            raise ValueError("Require begin < validation_begin < end (end is exclusive)")
        train_x, train_y = self._build_dataset(forward, lookback, begin, validation_begin)
        valid_x, valid_y = self._build_dataset(forward, lookback, validation_begin, end)
        # Keep model initialization and shuffled batches reproducible without
        # changing the caller's CPU random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_state)
            self._fit(train_x, train_y, valid_x, valid_y, max_epochs)
        self.training_period_ = (begin, validation_begin)
        self.validation_period_ = (validation_begin, end)
        joblib.dump(self, PREDICTOR_FILE_PATH)
        print(f"MLP fit; restored epoch {self.best_epoch_} (validation MSE {self.best_validation_loss_:.6f}).")

    def _build_dataset(self, forward, lookback, begin, end):
        universe = get_date_range_lookback(begin, end, lookback, forward)
        data = SimpleNamespace()
        initialize_prediction_data(data, universe, forward, lookback)
        features, _ = self._features(data.histories, n_dates=data.targets_n)
        targets = self.rank_normalize(data.targets).reshape(-1).astype(np.float32)
        valid = np.isfinite(targets)
        if not valid.any():
            raise ValueError(f"No finite targets in [{begin}, {end})")
        return features[valid], targets[valid]

    def _fit(self, train_x, train_y, valid_x, valid_y, max_epochs):
        self.regressor = nn.Sequential(
            nn.Linear(train_x.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        training = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                              batch_size=1024, shuffle=True)
        validation = DataLoader(TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y)),
                                batch_size=1024)
        self.history_ = []
        self.best_validation_loss_ = float("inf")
        stale_epochs = 0
        for epoch in range(1, max_epochs + 1):
            self.regressor.train()
            total_loss = 0.0
            for features, targets in training:
                optimizer.zero_grad()
                loss = loss_fn(self.regressor(features).squeeze(-1), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
            validation_loss = self._validation_loss(validation)
            if not np.isfinite(validation_loss):
                raise ValueError("MLP produced a nonfinite validation loss")
            self.history_.append({"train_loss": total_loss / len(train_y),
                                  "validation_loss": validation_loss})
            print(f"Epoch {epoch}: train MSE {self.history_[-1]['train_loss']:.6f}, "
                  f"validation MSE {validation_loss:.6f}")
            if validation_loss < self.best_validation_loss_:
                self.best_validation_loss_ = validation_loss
                self.best_epoch_ = epoch
                best_state = deepcopy(self.regressor.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 5:
                    break
        self.n_epochs_ = len(self.history_)
        self.regressor.load_state_dict(best_state)
        self.regressor.eval()

    def _validation_loss(self, validation):
        self.regressor.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, targets in validation:
                total_loss += nn.functional.mse_loss(
                    self.regressor(features).squeeze(-1), targets, reduction="sum"
                ).item()
        return total_loss / len(validation.dataset)

    @staticmethod
    def load():
        predictor = joblib.load(PREDICTOR_FILE_PATH)
        predictor._require_trained()
        return predictor

    def _require_trained(self):
        if not isinstance(getattr(self, "regressor", None), nn.Sequential) or not getattr(self, "selected_formulas", None):
            raise ValueError("Retrain Predictor to create an MLP model before prediction")

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

    def _features(self, histories, n_dates=None):
        selected_predictions = [formula.apply(histories) for formula in self.selected_formulas]
        available = min(prediction.shape[1] for prediction in selected_predictions)
        if n_dates is not None and available < n_dates:
            raise ValueError("Alpha history is too short; increase lookback")
        output_shape = (
            selected_predictions[0].shape[0],
            available if n_dates is None else n_dates,
        )
        slices = (slice(None), slice(0, output_shape[1]))
        selected_features = [
            self.rank_normalize(prediction[slices]).reshape(-1)
            for prediction in selected_predictions
        ]

        # A missing alpha is assigned the neutral centered rank, consistently
        # during training, validation and inference.
        prediction_matrix = np.nan_to_num(np.column_stack(selected_features), nan=0.0).astype(np.float32)
        return prediction_matrix, output_shape

    def predict(self, histories):
        self._require_trained()
        prediction_matrix, output_shape = self._features(histories)
        predictions = np.empty(len(prediction_matrix), dtype=np.float32)
        self.regressor.eval()
        with torch.no_grad():
            for start in range(0, len(prediction_matrix), 1024):
                batch = torch.from_numpy(prediction_matrix[start:start + 1024])
                predictions[start:start + 1024] = self.regressor(batch).squeeze(-1).numpy()
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
