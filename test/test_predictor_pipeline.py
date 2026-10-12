import tempfile
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch
from torch import nn
import pandas as pd

from alpha_zoo.formula import Formula
from portfolio.predictor import Predictor


def make_formula(score, operator="Neg", params=None):
    formula = Formula(None)
    formula.add_code({
        "formula": [{"name": operator, "input": ["close"],
                     "param": list(params or {}), "output": "alpha"}],
        "arguments": [params or {}],
    })
    formula.scores = [0, score, 0, 0, 0]
    return formula


class FormulaParallelismTests(unittest.TestCase):
    def test_parallel_results_preserve_order_arguments_and_inputs(self):
        formula = make_formula(1, "Diff", {"lag": 2})
        formula.arg_list.append({"lag": 1})
        formula.apply_args(1)
        histories = [{"close": np.array([i + 9., i + 4., i + 2.])} for i in range(8)]
        originals = [history["close"].copy() for history in histories]
        expected = np.array([values[:-1] - values[1:] for values in originals])
        assert_array_equal(formula.apply(histories, max_workers=4), expected)
        assert_array_equal(formula.apply(histories, max_workers=1), expected)
        for history, original in zip(histories, originals):
            self.assertEqual(set(history), {"close"})
            assert_array_equal(history["close"], original)

    def test_tickers_execute_in_multiple_processes_in_input_order(self):
        formula = make_formula(1)
        with tempfile.TemporaryDirectory() as directory:
            def operator(variables, parameters):
                # Rendezvous makes this fail for a serial backend or threads,
                # rather than relying on task duration or scheduler timing.
                Path(directory, str(os.getpid())).touch()
                deadline = time.monotonic() + 15
                while len(list(Path(directory).iterdir())) < 2:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Ticker tasks did not run in separate processes")
                    time.sleep(0.01)
                return np.array([-variables[0][0], os.getpid()])

            with patch.dict("alpha_zoo.formula.OPERATORS", {"Neg": operator}):
                histories = [{"close": np.array([value])} for value in range(8)]
                result = formula.apply(histories, max_workers=2)
                repeated = formula.apply(histories, max_workers=2)
        assert_array_equal(result[:, 0], -np.arange(8))
        workers = set(result[:, 1])
        self.assertEqual(len(workers), 2)
        self.assertNotIn(os.getpid(), workers)
        self.assertEqual(set(repeated[:, 1]), workers)

    def test_worker_errors_propagate(self):
        formula = make_formula(1, "Delay", {"lag": 10})
        with self.assertRaises(ValueError):
            formula.apply([{"close": np.array([1., 2.])}], max_workers=2)


class PredictorPipelineTests(unittest.TestCase):
    def test_ranks_are_cross_sectional_and_handle_ties_and_missing_values(self):
        values = np.array([
            [10, 400, 7, np.nan, np.nan],
            [20, 100, 7, 5, np.inf],
            [20, 300, 7, np.inf, -np.inf],
            [40, 200, 7, -np.inf, np.nan],
        ])
        expected = np.array([
            [-0.375, 0.375, 0, np.nan, np.nan],
            [0, -0.375, 0, 0, np.nan],
            [0, 0.125, 0, np.nan, np.nan],
            [0.375, -0.125, 0, np.nan, np.nan],
        ])
        assert_allclose(Predictor.rank_normalize(values), expected, equal_nan=True)
        assert_allclose(Predictor.rank_normalize(values[:, :3] * 10 + 50), expected[:, :3])

    def test_selection_training_prediction_and_saved_model(self):
        formulas = [make_formula(0.2), make_formula(0.8, "Delay", {"lag": 1}),
                    make_formula(0.8), make_formula(np.nan), make_formula(np.inf)]
        predictor = Predictor(SimpleNamespace(formulas=formulas))
        universe = [{"close": np.array(values, dtype=float)} for values in (
            [10, 11, 12, 13, 14, 15],
            [25, 24, 23, 22, 21, 20],
            [18, 17, 19, 16, 20, 21],
        )]
        histories = [{"close": history["close"][1:]} for history in universe]
        expected_features = np.column_stack([
            Predictor.rank_normalize(formula.apply(histories)[:, :3]).flatten()
            for formula in formulas[1:3]
        ])
        expected_targets = Predictor.rank_normalize(np.array([
            history["close"][:3] / history["close"][1:4] - 1 for history in universe
        ])).flatten()
        with tempfile.TemporaryDirectory() as directory:
            model_path = str(Path(directory) / "predictor.pkl")
            with patch("portfolio.predictor.get_date_range_lookback", return_value=universe) as load_data, \
                 patch("portfolio.predictor.PREDICTOR_FILE_PATH", model_path), \
                 patch.object(formulas[0], "apply", side_effect=AssertionError("Unselected alpha evaluated")), \
                 patch.object(predictor, "_fit", wraps=predictor._fit) as fit, \
                 patch("portfolio.predictor.joblib.dump"):
                predictor.train(1, 2, n_alphas=2, max_epochs=2)
                assert_allclose(fit.call_args.args[0], expected_features)
                assert_allclose(fit.call_args.args[1], expected_targets)
                assert_allclose(fit.call_args.args[2], expected_features)
                assert_allclose(fit.call_args.args[3], expected_targets)
                self.assertEqual(load_data.call_args_list[0].args,
                                 (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"), 2, 1))
                self.assertEqual(load_data.call_args_list[1].args,
                                 (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), 2, 1))
            self.assertEqual(predictor.selected_formulas, formulas[1:3])
            result = predictor.predict(histories)
            self.assertEqual(result.shape, (3, 4))
            self.assertTrue(np.isfinite(result).all())
            assert_array_equal(result, predictor.predict(histories))
            used_features, _ = predictor._features(histories)
            assert_allclose(used_features.reshape(3, 4, 2)[:, :3, :].reshape(-1, 2), expected_features)
            with torch.no_grad():
                expected = predictor.regressor(torch.from_numpy(used_features)).numpy().reshape(3, 4)
            assert_allclose(result, expected)
            layers = [layer for layer in predictor.regressor if isinstance(layer, nn.Linear)]
            self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                             [(2, 256), (256, 128), (128, 64), (64, 1)])
            self.assertEqual([layer.p for layer in predictor.regressor if isinstance(layer, nn.Dropout)],
                             [0.3, 0.3, 0.3])
            joblib.dump(predictor, model_path)
            with patch("portfolio.predictor.PREDICTOR_FILE_PATH", model_path):
                restored = Predictor.load()
            assert_allclose(restored.predict(histories), result)
            self.assertFalse(hasattr(restored, "histories"))
            self.assertFalse(hasattr(restored, "targets"))

    def test_early_stopping_restores_best_weights_after_five_stale_epochs(self):
        predictor = Predictor(SimpleNamespace(formulas=[]))
        x = np.array([[-0.3], [0.1], [0.4]], dtype=np.float32)
        y = x[:, 0].copy()
        checkpoints = []
        losses = iter([3., 2., 2., 2.1, 2.2, 2.3, 2.4])

        def validation_loss(loader):
            checkpoints.append({key: value.clone() for key, value in predictor.regressor.state_dict().items()})
            return next(losses)

        with patch.object(predictor, "_validation_loss", side_effect=validation_loss):
            predictor._fit(x, y, x, y, max_epochs=20)
        self.assertEqual(predictor.n_epochs_, 7)
        self.assertEqual(predictor.best_epoch_, 2)
        self.assertEqual(predictor.best_validation_loss_, 2.)
        for key, value in predictor.regressor.state_dict().items():
            torch.testing.assert_close(value, checkpoints[1][key], rtol=0, atol=0)
        self.assertFalse(predictor.regressor.training)

    def test_missing_features_are_neutral_and_missing_targets_excluded(self):
        predictor = Predictor(SimpleNamespace(formulas=[]))
        predictor.selected_formulas = [SimpleNamespace(apply=lambda histories: np.array(
            [[np.nan, 1.], [2., np.inf]]))]

        def initialize(data, universe, forward, lookback):
            data.histories = []
            data.targets_n = 2
            data.targets = np.array([[np.nan, 3.], [2., 4.]])

        with patch("portfolio.predictor.get_date_range_lookback", return_value=[]), \
             patch("portfolio.predictor.initialize_prediction_data", side_effect=initialize):
            features, targets = predictor._build_dataset(1, 2, "2020-01-01", "2024-01-01")
        assert_array_equal(features, np.zeros((3, 1)))
        assert_allclose(targets, [-0.25, 0., 0.25])

    def test_invalid_dates_fail_before_loading_data(self):
        predictor = Predictor(SimpleNamespace(formulas=[make_formula(1)]))
        with patch("portfolio.predictor.get_date_range_lookback") as load_data:
            for begin, split, end in (("2024-01-01", "2024-01-01", "2025-01-01"),
                                      ("2020-01-01", "2025-01-01", "2025-01-01")):
                with self.assertRaisesRegex(ValueError, "begin < validation_begin < end"):
                    predictor.train(1, 2, begin, end, validation_begin=split)
            load_data.assert_not_called()

    def test_selection_defaults_to_n_alphas_and_caps_to_available(self):
        formulas = [make_formula(0.2), make_formula(0.8)]
        predictor = Predictor(SimpleNamespace(formulas=formulas))
        # Stop at data loading: selection must already be complete at this point.
        with patch("portfolio.predictor.get_date_range_lookback", side_effect=RuntimeError("stop")):
            for count, expected in ((1, [formulas[1]]), (50, formulas[::-1])):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    predictor.train(1, 2, n_alphas=count)
                self.assertEqual(predictor.selected_formulas, expected)

    def test_invalid_selection_fails_before_loading_data(self):
        predictor = Predictor(SimpleNamespace(formulas=[make_formula(1)]))
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                predictor.train(1, 2, None, None, n_alphas=count)
        for formulas in ([], [make_formula(np.nan)], [make_formula(np.inf)]):
            with self.assertRaisesRegex(ValueError, "finite RankIR"):
                Predictor(SimpleNamespace(formulas=formulas)).train(1, 2, None, None)

    def test_old_models_require_retraining(self):
        with self.assertRaisesRegex(ValueError, "Retrain"):
            Predictor(SimpleNamespace(formulas=[])).predict([])


if __name__ == "__main__":
    unittest.main()
