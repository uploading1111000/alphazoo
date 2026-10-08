from .data_scrape import get_date_range_lookback
import scipy.stats as ss
import numpy as np
from constants import CUTOFF

def rank_in(score, past_scores):
    N = len(past_scores)
    assert N >= 5 #deal with small zoo case later

    if not np.isfinite(score):
        return 1.0  # Final reward becomes zero.

    return sum(
        np.isfinite(past_score) and past_score >= score
        for past_score in past_scores
    ) / N

class Alpha_Backtester():
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

        print(f"{self.targets_n} days loaded against {self.universe_n} tickers!")

    #score formula by 5 metrics
    def score_all(self,formula):
        self._most_recent_predictions = formula.apply(self.histories)[:,:self.targets_n] #(universe_n,targets_n)
        scores = self.score_effectiveness_stability()
        scores.append(self.score_turnover())
        scores.append(self.score_diversity())
        scores.append(self.score_OFR(formula))
        return scores

    #score formula, rank each in zoo, produce score
    def alpha_score(self, formula):
        scores_more = []
        ranks_more = []
        a_scores = []
        predictions = []
        for i in range(len(formula.arg_list)):
            formula.apply_args(i)
            scores = self.score_all(formula)
            #ranks = np.array([1 - rank_in(scores[i],self.zoo.scores[i]) for i in range(4)] + [scores[4]/10])
            #STARTER
            ranks = np.array([
                1 if scores[0] > CUTOFF[0] else scores[0] / CUTOFF[0],
                1 if scores[1] > CUTOFF[1] else scores[1] / CUTOFF[1],
                1 if scores[2] > CUTOFF[2] else min(1.0, 2.0 - scores[2]/CUTOFF[2]),
                1 if scores[3] > CUTOFF[3] else min(1.0, 2.0 - scores[3]/CUTOFF[3]),
                scores[4] / 10
            ])
            a_score = sum(ranks) / 5
            scores_more.append(scores)
            ranks_more.append(ranks)
            a_scores.append(a_score)
            predictions.append(self._most_recent_predictions)
        best_i = np.argmax(a_scores)
        formula.best_args(best_i)
        formula.set_atts(
            scores_more[best_i],
            ranks_more[best_i],
            a_scores[best_i],
            predictions[best_i]
        )
        return a_scores[best_i]

    def score_effectiveness_stability(self):
        self._pred_ranks = np.zeros((self.targets_n,self.universe_n))
        RIC_t = np.zeros((self.targets_n))
        for i in range(self.targets_n):
            self._pred_ranks[i] = ss.rankdata(self._most_recent_predictions[:,i],nan_policy="omit")
            target_ranks = ss.rankdata(self.targets[:, i],nan_policy="omit")
            pred_ranks = self._pred_ranks[i]
            if np.ptp(pred_ranks) == 0 or np.any(np.isnan(pred_ranks)):
                RIC_t[i] = 0
            else:
                RIC_t[i] = ss.pearsonr(pred_ranks,target_ranks).statistic
        effectiveness = np.mean(RIC_t)
        stability = effectiveness / np.std(RIC_t)
        return [effectiveness, stability]

    #run AFTER effectiveness_stability (reuses prediction ranking)
    def score_turnover(self):
        selected = [set([j for j,rank in enumerate(self._pred_ranks[i]) if rank > self.universe_n * (1-self.turnover_top_p)]) for i in range(self.targets_n)] #it's beautiful
        Ti = np.empty((self.targets_n-self.turnover_skip))
        for i in range(self.targets_n-self.turnover_skip):
            Ti[i] = len(selected[i].difference(selected[i+self.turnover_skip]))
        return -np.mean(Ti)/(self.turnover_top_p*self.universe_n)

    def score_diversity(self):
        if self.zoo.n == 0:
            return 0
        
        m_correlations = self.zoo.prediction_correlations(self._most_recent_predictions)

        return -np.max(np.abs(m_correlations))

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
