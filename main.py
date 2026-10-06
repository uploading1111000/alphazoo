import data
import alpha_zoo
import llm
import pandas as pd

print("Opening Openrouter session")
with llm.Model() as model:
    print("Openrouter session opened, generating starters...")
    zoo = alpha_zoo.Alpha_Zoo(model)
    zoo.create_starters(20,90,pd.to_datetime("2020-01-01"),pd.to_datetime("2025-01-01"))
    print("Starters generated")
    backtester = data.Alpha_Backtester(model,zoo,20,90,pd.to_datetime("2020-01-01"),pd.to_datetime("2025-01-01"))
    mcts = alpha_zoo.MCT_searcher(backtester,model,zoo)
    print("Beginning zoo building")
    i = 0
    while zoo.n < 20:
        i += 1
        print(f"Starting {i}th tree search...")
        mcts.start_new_tree()
        while not mcts.explore_tree():
            print(f"{mcts.searches} nodes searched, budget: {mcts.budget}")
        n_new = mcts.add_effective()
        print(f"{i}th tree search complete. {n_new} new alphas in zoo (now size {zoo.n})")
    print("Sufficient zoo members to remove starters.")
    zoo.remove_starters()
    zoo.save()

#portfolio?