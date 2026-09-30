from .data_scrape import get_date_range_lookback
import scipy.stats as ss
import numpy as np

def rank_in(score, past_scores):
    N = len(past_scores)
    assert N >= 5 #deal with small zoo case later

    if not np.isfinite(score):
        return 1.0  # Final reward becomes zero.

    return sum(
        np.isfinite(past_score) and past_score >= score
        for past_score in past_scores
    ) / N

class alpha_backtester():
    def __init__(self, model, zoo, forward, lookback,begin,end,turnover_skip=5,turnover_top_p=0.1):
        self.model = model
        self.forward = forward
        self.lookback = lookback
        self.zoo = zoo
        self.universe = get_date_range_lookback(begin,end,lookback,forward)
        self.universe_n = len(self.universe)
        self.turnover_skip = turnover_skip
        self.turnover_top_p = turnover_top_p

        #targets
        self.day_length = len(self.universe[0]["close"])
        self.targets_n = self.day_length - forward - lookback
        self.lookback_n = self.day_length - forward
        self.targets = np.zeros((self.universe_n,self.targets_n))
        self.targets[0] = self.universe[0]["close"][:self.targets_n] / self.universe[0]["close"][forward:forward+self.targets_n] - 1
        for i, history in enumerate(self.universe[1:]):
            assert len(history["close"]) == self.day_length
            self.targets[i+1] = history["close"][:self.targets_n] / history["close"][forward:forward+self.targets_n] - 1

        #universe_histories
        self.histories = []
        for history in self.universe:
            trunc_history = {}
            for key in history:
                trunc_history[key] = history[key][forward:]
            self.histories.append(trunc_history)

    #score formula by 5 metrics
    def score_all(self,formula):
        self._most_recent_predictions = formula.apply(self.histories)[:,:self.targets_n] #(universe_n,targets_n)
        scores = self.score_effectiveness_stability()
        scores.append(self.score_turnover())
        scores.append(self.score_diversity())
        scores.append(self.score_OFR(formula))
        ranks = [1 - rank_in(scores[i],self.zoo.alpha_scores[i]) for i in range(5)]
        a_score = sum(ranks) / 5
        return scores, ranks, a_score


    #score formula, rank each in zoo, produce score
    def alpha_score(self, formula):
        scores_more = []
        ranks_more = []
        a_scores = []
        predictions = []
        for i in range(len(formula.arg_list)):
            formula.apply_args(i)
            scores, ranks, a_score = self.score_all(formula)
            scores_more.append(scores)
            ranks_more.append(ranks)
            a_scores.append(a_score)
            predictions.append(self._most_recent_predictions)
        best_i = np.argmax(a_scores)
        formula.best_args(best_i)
        formula.scores = scores_more[best_i]
        formula.ranks = ranks_more[best_i]
        formula.a_score = a_scores[best_i]
        formula.predictions = predictions[best_i]
        return a_scores[best_i]

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
            RIC_t[i] = ss.pearsonr(self._pred_ranks[i],target_ranks).statistic
        effectiveness = np.mean(RIC_t)
        stability = effectiveness / np.std(RIC_t)
        return [effectiveness, stability]

    #run AFTER effectiveness_stability (reuses prediction ranking)
    def score_turnover(self):
        selected = [set([j for j,rank in enumerate(self._pred_ranks[i]) if rank > self.universe_n * (1-self.turnover_top_p)]) for i in range(self.targets_n)] #it's beautiful
        Ti = np.empty((self.targets_n-self.turnover_skip))
        for i in range(self.targets_n-self.turnover_skip):
            Ti[i] = len(selected[i].difference(selected[i+self.turnover_skip]))
        return -np.mean(Ti)

    def score_diversity(self):
        if self.zoo.n == 0:
            return 0
        
        zoo_predictions = self.zoo.prediction_matrix

        reshaped_zoo = zoo_predictions.reshape((zoo_predictions.shape[0],self.universe_n*self.targets_n))
        reshaped_predictions = self._most_recent_predictions.reshape((self.universe_n*self.targets_n))

        centered_predictions = reshaped_predictions - np.mean(reshaped_predictions)
        centered_zoo = reshaped_zoo - np.mean(reshaped_zoo,axis=1,keepdims=True)

        std_predictions = np.linalg.norm(centered_predictions)
        std_zoo = np.linalg.norm(centered_zoo,axis=1)

        m_correlations = np.abs(np.dot(centered_zoo,centered_predictions)) / (std_predictions * std_zoo)
        return -np.max(m_correlations)

    def score_OFR(self, formula):
        overfit_assessment = self.model.overfitting_risk(formula.alpha(),formula.refinement_history())
        formula.add_ofr_review(overfit_assessment["reason"])
        return overfit_assessment["score"]

#effectiveness - rankIC. Uses target/predictions
#stability - rankIR. Uses target/predictions
#turnover - skip metric. Uses predictions and turnover params, as well as _pred_ranks from effectiveness/stability
#diversity - max corr. Uses predictions and other zoo members
#OFR - LLM based. Only uses formula and LLM

class full_backtester():
    def __init__(self):
        raise NotImplementedError