from data import get_date_range_lookback
import numpy as np
from heapq import nsmallest, nlargest
from operator import itemgetter

class Portfolio:
    def __init__(self,predictor,top_k,forward,skip=5):
        self.predictor = predictor
        self.top_k = top_k
        self.forward = forward
        self.skip=skip
        self.top_n = (top_k * skip) // forward

    def backtest(self,begin,end,lookback):
        universe = get_date_range_lookback(begin,end,lookback,self.forward)
        universe_n = len(universe)
        print("Data loaded")
        predictions = self.predictor.predict(universe)[:,::-1]
        print("Predictions complete")
        pred_length = predictions.shape[1]
        prices = np.array([history["close"] for history in universe])[:,:pred_length][:,::-1]
        held_stocks = np.array([1/(prices[i,0]*universe_n) for i in range(universe_n)])

        self.first_day(predictions[:,0],prices[:,0])
        for i in range(self.skip,pred_length,self.skip):
            self.single_day(predictions[:,i],prices[:,i])

        print(self.holdings_value(prices[:,pred_length-1]))
        value = sum(held_stocks * prices[:,pred_length-1])
        print(value)




    def first_day(self,predictions,prices,total=1):
        self.holdings = np.zeros_like(predictions)
        top_stocks = nlargest(self.top_k,enumerate(predictions),key=itemgetter(1))
        for stock_i,_ in top_stocks:
            self.holdings[stock_i] = total / (self.top_k * prices[stock_i])
        return ([i for i,_ in top_stocks], [])

    def single_day(self,predictions,prices):
        owned = [(i,predictions[i]) for i,number in enumerate(self.holdings) if number > 0]
        unowned = [(i,predictions[i]) for i,number in enumerate(self.holdings) if number == 0]

        top_stocks_i = set([index for index, _ in nlargest(self.top_k,enumerate(predictions),key=itemgetter(1))])
        owned_i = set([index for index, _ in owned])
        suggested_turnover = len(top_stocks_i.difference(owned_i))
        turnover = min(suggested_turnover,self.top_n)

        sellables = nsmallest(turnover,owned,key=itemgetter(1))
        purchaseables = nlargest(turnover,unowned,key=itemgetter(1))
        sell_value = 0
        for i, _ in sellables:
            sell_value += prices[i] * self.holdings[i]
            self.holdings[i] = 0
        
        sell_value = sell_value * 0.997 #0.15% on buy + 0.15% on sell < 0.3% overall cost

        for i, _ in purchaseables:
            self.holdings[i] = sell_value / (turnover * prices[i])

    def holdings_value(self,prices):
        return sum([holding * prices[i] for i, holding in enumerate(self.holdings)]) * 0.9985

        




        