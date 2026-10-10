import data
import alpha_zoo
import llm
import portfolio
import pandas as pd

#zoo builder
""" print("Opening Openrouter session")
with llm.Model() as model:
    print("Openrouter session opened, generating starters...")
    zoo = alpha_zoo.Alpha_Zoo(model)
    zoo.create_starters(20,90,pd.to_datetime("2020-01-01"),pd.to_datetime("2025-01-01"))
    print("Starters generated")
    zoo.save()
    zoo = alpha_zoo.Alpha_Zoo.load()
    print(f"Zoo loaded with {zoo.n} members")
    zoo.model = model
    backtester = data.Alpha_Backtester(model,zoo,20,90,pd.to_datetime("2020-01-01"),pd.to_datetime("2025-01-01"))
    print("Loaded backtester")
    mcts = alpha_zoo.MCT_searcher(backtester,model,zoo)
    print("Beginning zoo building")
    i = 0
    while zoo.n < 100:
        i += 1
        if i >= 300:
            zoo.save()
            raise RuntimeError("Too many tree searches completed with no alphas, exiting now!")
        print(f"Starting {i}th tree search...")
        mcts.start_new_tree()
        while not mcts.explore_tree():
            print(f"{mcts.searches} nodes searched, budget: {mcts.budget}")
        n_new = mcts.add_effective()
        print(f"{i}th tree search complete. {n_new} new alphas in zoo (now size {zoo.n})")
        if n_new > 0:
            zoo.save()
    print("Sufficient zoo members")
    zoo.save() """

#portfolio
"""zoo = alpha_zoo.Alpha_Zoo.load()
print(f"Zoo loaded with {zoo.n} members")
predictor = portfolio.Predictor(zoo)
predictor.train(20,90,pd.to_datetime("2025-01-01"),pd.to_datetime("2025-07-01"),n_alphas=100)"""
predictor = portfolio.Predictor.load()
predictor.test_correlate()