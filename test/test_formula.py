from alpha_zoo import Formula
from data import Alpha_Backtester
import pandas as pd

formula = Formula(None)

form_dict = {
  "formula": [
    {
      "name": "Pct",
      "param": [
        "daily_lag"
      ],
      "input": [
        "close"
      ],
      "output": "daily_return"
    },
    {
      "name": "Std",
      "param": [
        "volatility_window"
      ],
      "input": [
        "daily_return"
      ],
      "output": "return_volatility"
    },
    {
      "name": "Pct",
      "param": [
        "reversal_window"
      ],
      "input": [
        "close"
      ],
      "output": "recent_return"
    },
    {
      "name": "DivA",
      "param": [],
      "input": [
        "recent_return",
        "return_volatility"
      ],
      "output": "vol_scaled_move"
    },
    {
      "name": "Tanh",
      "param": [],
      "input": [
        "vol_scaled_move"
      ],
      "output": "bounded_move"
    },
    {
      "name": "Neg",
      "param": [],
      "input": [
        "bounded_move"
      ],
      "output": "alpha"
    }
  ],
  "arguments": [
    {
      "daily_lag": 1,
      "volatility_window": 20,
      "reversal_window": 5
    },
    {
      "daily_lag": 1,
      "volatility_window": 40,
      "reversal_window": 10
    },
    {
      "daily_lag": 1,
      "volatility_window": 60,
      "reversal_window": 15
    }
  ]
}

formula.add_code(form_dict)
backtester = Alpha_Backtester(None,None,20,90,pd.to_datetime("2025-01-01"),pd.to_datetime("2026-01-01"))
formula.apply(backtester.histories)