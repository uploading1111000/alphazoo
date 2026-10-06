import data
import alpha_zoo
import llm
import pandas as pd

#model
with llm.Model() as model:
    zoo = alpha_zoo.Alpha_Zoo(model)
    zoo.create_starters(20,90,pd.to_datetime("2020-01-01"),pd.to_datetime("2026-01-01"))
    
#zoo
#backtester
#mcts

#portfolio?