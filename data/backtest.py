from .data_scrape import get_date_range_lookback
import scipy.stats as ss
import numpy as np

class train_backtester():
    def __init__(self, zoo, forward, lookback,begin,end,turnover_skip=5,turnover_top_p=0.1):
        self.forward = forward
        self.lookback = lookback
        self.zoo = zoo
        self.universe = get_date_range_lookback(begin,end,lookback,forward)
        self.universe_n = len(get_date_range_lookback)
        self.turnover_skip = turnover_skip
        self.turnover_top_p = turnover_top_p

        #targets
        self.day_length = len(self.universe[0]["CLOSE"])
        self.targets_n = self.day_length - forward - lookback
        self.lookback_n = self.day_length - forward
        self.targets = np.zeros((self.universe_n,self.targets_n))
        self.targets[0] = self.universe[0]["CLOSE"][:self.targets_n]
        for i, history in enumerate(self.universe[1:]):
            assert len(history["CLOSE"]) == self.day_length
            self.targets[i+1] = history["CLOSE"][:self.targets_n]

        #universe_histories
        self.histories = []
        for history in self.universe:
            trunc_history = {}
            for key in history:
                trunc_history[key] = history[key][forward:]
            self.histories.append(trunc_history)

    #score formula by 5 metrics
    def score_all(self,formula):
        self._most_recent_predictions = formula.apply(self.histories)[:self.targets_n]
        self.score_effectiveness_stability()
        self.score_turnover()
        self.score_diversity()
        self.score_OFR()

    #score formula by 5 metrics then rank each position in zoo to produce e_i
    def rank_all(self, formula):
        self.score_all(formula)

    #score formula, rank each in zoo, produce score
    def alpha_score(self, formula):
        self.rank_all(formula)

    def add_zoo(self, formula=None):
        if formula != None:
            self.score_all(formula)
        self.zoo.add(formula)

    def score_effectiveness_stability(self):
        self._pred_ranks = np.zeros((self.targets_n,self.universe_n))
        RIC_t = np.zeros((self.targets_n))
        for i in range(self.targets_n):
            self._pred_ranks[i] = ss.rankdata(self._most_recent_predictions[:,i])
            target_ranks = ss.rankdata(self.targets[:, i])
            RIC_t[i] = ss.pearsonr(self._pred_ranks,target_ranks)
        self._most_recent_effectiveness = np.mean(RIC_t)
        self._most_recent_stability = self._most_recent_effectiveness / np.std(RIC_t)

    #run AFTER effectiveness_stability (reuses prediction ranking)
    def score_turnover(self):
        selected = [set([j for j,rank in enumerate(self._pred_ranks[i]) if rank > self.universe_n * (1-self.turnover_top_p)]) for i in range(self.targets_n)] #it's beautiful
        Ti = np.array((self.targets_n-self.turnover_skip))
        for i in range(self.targets_n-self.turnover_skip):
            Ti[i] = len(selected[i].difference(i+self.turnover_skip))
        self._most_recent_turnover = np.mean(Ti)

    def score_diversity(self):
        if self.zoo.n == 0:
            self._most_recent_diversity = 0
            return
        
        zoo_predictions = self.zoo.prediction_matrix
        centered_predictions = self._most_recent_predictions - np.mean(self._most_recent_predictions)
        centered_zoo = zoo_predictions - np.mean(zoo_predictions,axis=1,keepdims=True)

        std_predictions = np.linalg.norm(centered_predictions)
        std_zoo = np.linalg.norm(centered_zoo,axis=1)

        m_correlations = np.abs(np.dot(centered_predictions,centered_zoo)) / (std_predictions * std_zoo)
        self._most_recent_diversity = -np.max(m_correlations)

#TODO: Overfitting Risk (requires thinking through formula class)


#effectiveness - rankIC. Uses target/predictions
#stability - rankIR. Uses target/predictions
#turnover - skip metric. Uses predictions and turnover params, as well as _pred_ranks from effectiveness/stability
#diversity - max corr. Uses predictions and other zoo members
#OFR - LLM based. Only uses formula and LLM




class full_backtester():
    def __init__(self):
        raise NotImplementedError