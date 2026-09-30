import numpy as np
import warnings
from scipy.stats import skew, kurtosis

class Operator():
    n_variables = 1
    n_parameters = 0
    name = "Noop"
    example = "Noop(x)"
    descriptor = "Does nothing"

    def __call__(self, variables, parameters):
        return self.calculate(variables, parameters)

    def calculate(self, variables, parameters):
        variables, parameters = self.arg_check(variables, parameters)
        return self._calc(variables, parameters)

    def arg_check(self, variables, parameters):
        if len(variables) != self.n_variables:
            raise TypeError(
                f"{self.name} takes {self.n_variables} variables but {len(variables)} given"
            )
        if len(parameters) != self.n_parameters:
            raise TypeError(
                f"{self.name} takes {self.n_parameters} parameters but {len(parameters)} given"
            )
        variables = [np.asarray(variable) for variable in variables]
        if len(variables) > 1:
            window = min(len(variable) for variable in variables)
            variables = [variable[:window] for variable in variables]
        return variables, parameters

    def _check_lookback(self, lookback):
        if isinstance(lookback, (bool, np.bool_)) or not isinstance(lookback, (int, np.integer)):
            raise TypeError(f"{self.name} lookback must be a positive integer")
        if lookback <= 0:
            raise ValueError(f"{self.name} lookback must be a positive integer")
        return lookback
    
    def _calc(self, variables, params):
        warnings.warn("Passthrough operator (noop) ran")
        return variables[0]

    def lookback_window(self,variable_lookbacks,params):
        return max(variable_lookbacks)

    def _rolling_lookback(self, variable_lookbacks, params):
        return max(variable_lookbacks) + self._check_lookback(params[0]) - 1

class NegO(Operator):
    n_variables = 1
    n_parameters = 0
    name = 'Neg'
    example = 'Neg(x)'
    descriptor = 'The opposite value of x'

    def _calc(self, variables, params):
        return np.negative(variables[0])

class AbsO(Operator):
    n_variables = 1
    n_parameters = 0
    name = 'Abs'
    example = 'Abs(x)'
    descriptor = 'The absolute value of x'

    def _calc(self, variables, params):
        return np.abs(variables[0])

class PowerO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Pow"
    example = "Pow(x,n)"
    descriptor = 'The value of x^n'

    def _calc(self, variables, params):
        return np.power(variables[0],params[0])

class SignO(Operator):
    n_variables = 1
    n_parameters = 0
    name = "Sign"
    example = "Sign(x)"
    descriptor = "Returns -1 for x < 0, 1 for x > 0, 0 for x==0"

    def _calc(self, variables, params):
        return np.sign(variables[0])

class SinO(Operator):
    n_variables = 1
    n_parameters = 0
    name = "Sin"
    example = "Sin(x)"
    descriptor = "The sine of x in radians"

    def _calc(self, variables, params):
        return np.sin(variables[0])

class CosO(Operator):
    n_variables = 1
    n_parameters = 0
    name = "Cos"
    example = "Cos(x)"
    descriptor = "The cosine of x in radians"
    
    def _calc(self, variables, params):
        return np.cos(variables[0])

class TanhO(Operator):
    n_variables = 1
    n_parameters = 0
    name = "Tanh"
    example = "Tanh(x)"
    descriptor = "The hyperbolic tangent of x"

    def _calc(self, variables, params):
        return np.tanh(variables[0])

class LogO(Operator):
    n_variables = 1
    n_parameters = 0
    name = "Log"
    example = "Log(x)"
    descriptor = "The natural logarithm of x"

    def _calc(self, variables, params):
        return np.log(variables[0])

class DelayO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Delay"
    example = "Delay(x,t)"
    descriptor = "The value of x at t trading days prior"

    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        if len(x) < t+1:
            raise ValueError(
                f"{self.name} Lookback time greater than number of days in window"
                f"X length: {len(x)}, Lookback Time: {t}"
            )
        return x[t:]

    def lookback_window(self, variable_lookbacks, params):
        return variable_lookbacks[0] + self._check_lookback(params[0])

class DiffO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Diff"
    example = "Diff(x,t)"
    descriptor = "The difference between x and the value t trading days prior (equiv to SubA(x,Delay(x,t)))"
    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        delay = DelayO._calc(self,[x],[t])

        return x[:len(delay)] - delay

    def lookback_window(self, variable_lookbacks, params):
        return variable_lookbacks[0] + self._check_lookback(params[0])

class PctO(Operator):
    n_variables = 1
    n_parameters = 1
    n_variables = 1
    name = "Pct"
    example = "Pct(x,t)"
    descriptor = "The relative change between x and the value t trading days prior (equiv to DivA(x,Delay(x,t)) - 1)"
        
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        delay = DelayO._calc(self,[x],[t])
    
        return (x[:len(delay)]/delay) - np.ones(len(delay),dtype=np.float32)

    def lookback_window(self, variable_lookbacks, params):
        return variable_lookbacks[0] + self._check_lookback(params[0])

class MaO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Ma"
    example = "Ma(x,t)"
    descriptor = "The mean of x over the last t trading days"
            
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        means = []
        for i in range(len(x)-t+1):
            means.append(np.mean(x[i,i+t]))
        return np.array(means)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class MedO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Med"
    example = "Med(x,t)"
    descriptor = "The median of x over the last t trading days"
                
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        medians = []
        for i in range(len(x)-t+1):
            medians.append(np.median(x[i,i+t]))
        return np.array(medians)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class SumO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Sum"
    example = "Sum(x,t)"
    descriptor = "The sum of x over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        sums = []
        for i in range(len(x)-t+1):
            sums.append(np.sum(x[i,i+t]))
        return np.array(sums)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class StdO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Std"
    example = "Std(x,t)"
    descriptor = "The standard deviation of x over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        stds = []
        for i in range(len(x)-t+1):
            stds.append(np.std(x[i,i+t]))
        return np.array(stds)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class MaxO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Max"
    example = "Max(x,t)"
    descriptor = "The maximum value of x over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        maxs = []
        for i in range(len(x)-t+1):
            maxs.append(np.max(x[i,i+t]))
        return np.array(maxs)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class MinO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Min"
    example = "Min(x,t)"
    descriptor = "The minimum value of x over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        mins = []
        for i in range(len(x)-t+1):
            mins.append(np.max(x[i,i+t]))
        return np.array(mins)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class RankO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Rank"
    example = "Rank(x,t)"
    descriptor = "The ranking of x relative to its values over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        ranks = []
        for i in range(len(x)-t+1):
            rank = 1
            val = x[i]
            for other in x[i+1,i+t]:
                if other < val:
                    rank += 1
            ranks.append(rank)
        return np.array(ranks)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class SkewO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Skew"
    example = "Skew(x,t)"
    descriptor = "The skewness value of x over the last t trading days"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        skews = []
        for i in range(len(x)-t+1):
            skews.append(skew(x[i,i+t]))
        return np.array(skews)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)
    
class KurtO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Kurt"
    example = "Kurt(x,t)"
    descriptor = "The kurtosis value of x over the last t trading days (Fisher)"
                    
    def _calc(self, variables, params):
        x,t = variables[0], self._check_lookback(params[0])
        kurts = []
        for i in range(len(x)-t+1):
            kurts.append(kurtosis(x[i,i+t]))
        return np.array(kurts)

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class VariO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Vari"
    example = "Vari(x,t)"
    descriptor = "The variation value of x over the last t trading days. Equivalent to DivA(Std(x,t),Ma(x,t))"

    def _calc(self, variables, params):
        std = StdO._calc(self,variables, params)
        ma = MaO._calc(self,variables, params)
        return std/ma

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class AutocorrO(Operator):
    n_variables = 1
    n_parameters = 2
    name = "Autocorr"
    example = "Autocorr(x,t,n)"
    descriptor = "The autocorrelation coefficient of x with a lag of n over the past t trading days. Equiv to Corr(x,Delay(x,n),t)"

    def _calc(self, variables, params):
        delay = DelayO._calc(self,[variables[0]],[params[1]])
        return CorrO._calc(self,[variables[0],delay],[params[0]])

    def lookback_window(self, variable_lookbacks, params):
        t = self._check_lookback(params[0])
        n = self._check_lookback(params[1])
        return variable_lookbacks[0] + n + t - 1

class ZscoreO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "Zscore"
    example = "Zscore(x,t)"
    descriptor = "The z-score normalization of x based on the mean and standard deviation of its values over the past trading days. Equiv to DivA(SubA(x,Ma(x,t)),Std(x,t))"

    def _calc(self, variables, params):
        ma = MaO._calc(self,variables,params)
        std = StdO._calc(self,variables,params)
        return (variables[0] - ma) / std

    def lookback_window(self, variable_lookbacks, params):
        return self._rolling_lookback(variable_lookbacks, params)

class AddAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "AddA"
    example = "AddA(x,y)"
    descriptor = "Add two variables x + y"

    def _calc(self, variables, params):
        return variables[0] + variables[1]

class SubAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "SubA"
    example = "SubA(x,y)"
    descriptor = "Subtract two variables x - y"

    def _calc(self, variables, params):
        return variables[0] - variables[1]

class MulAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "MulA"
    example = "MulA(x,y)"
    descriptor = "Multiply two variables x * y"

    def _calc(self, variables, params):
        return variables[0] * variables[1]

class DivAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "DivA"
    example = "DivA(x,y)"
    descriptor = "Divide two variables x / y"

    def _calc(self, variables, params):
        return variables[0] / variables[1]

class AddPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "AddP"
    example = "AddP(x,n)"
    descriptor = "Add variable x to parameter n"

    def _calc(self, variables, params):
        return variables[0] + params[0]

class SubPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "SubP"
    example = "SubP(x,n)"
    descriptor = "Subtract parameter n from variable x"

    def _calc(self, variables, params):
        return variables[0] - params[0]

class MulPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "MulP"
    example = "MulP(x,n)"
    descriptor = "Multiply variable x by parameter n"

    def _calc(self, variables, params):
        return variables[0] * params[0]

class DivPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "DivP"
    example = "DivP(x,n)"
    descriptor = "Divide variable x by parameter n"

    def _calc(self, variables, params):
        return variables[0] / params[0]

class GreaterAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "GreaterA"
    example = "GreaterA(x,y)"
    descriptor = "1 if x>y, 0 otherwise"

    def _calc(self, variables, params):
        x, y = variables[0], variables[1]
        window = min(len(x),len(y))
        return np.array([1 if x[i] > y[i] else 0 for i in range(window)])
    
class LessAO(Operator):
    n_variables = 2
    n_parameters = 0
    name = "LessA"
    example = "LessA(x,y)"
    descriptor = "1 if x<y, 0 otherwise"

    def _calc(self, variables, params):
        x, y = variables[0], variables[1]
        window = min(len(x),len(y))
        return np.array([1 if x[i] < y[i] else 0 for i in range(window)])

class GreaterPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "GreaterP"
    example = "GreaterP(x,n)"
    descriptor = "1 if x>n, 0 otherwise"

    def _calc(self, variables, params):
        x, n = variables[0], params[0]
        return np.array([1 if xval > n else 0 for xval in x])

class LessPO(Operator):
    n_variables = 1
    n_parameters = 1
    name = "LessP"
    example = "LessP(x,n)"
    descriptor = "1 if x<n, 0 otherwise"

    def _calc(self, variables, params):
        x, n = variables[0], params[0]
        return np.array([1 if xval < n else 0 for xval in x])
    
class CovO(Operator):
    n_variables = 2
    n_parameters = 1
    name = "Cov"
    example = "Cov(x,y,t)"
    descriptor = "Covariance of x and y over the last t trading days"

    def _calc(self, variables, params):
        x,y,t = variables[0], variables[1], self._check_lookback(params[0])
        window = min(len(x),len(y))
        if t > window:
            raise ValueError(
                f"{self.name} Lookback time greater than number of days in window"
                f"X length: {len(x)}, Y length: {len(y)}, Lookback Time: {t}"
            )
        covariance = []
        for i in range(window - t + 1):
            m = np.vstack([x[i:i+t],y[i:i+t]])
            covariance.append(np.cov(m[0,1]))
        return np.array(covariance,dtype=np.float32)

    def lookback_window(self, variable_lookbacks, params):
        return max(variable_lookbacks) + self._check_lookback(params[0]) - 1

class CorrO(Operator):
    n_variables = 2
    n_parameters = 1
    name = "Corr"
    example = "Corr(x,y,t)"
    descriptor = "Correlation of x and y over the last t trading days. Equivalent to DivA(Cov(x,y,t),MulA(Std(x,t),Std(y,t)))"

    def _calc(self, variables, params):
        window = min(len(variables[0]), len(variables[1]))
        variables = [variable[:window] for variable in variables]
        cov = CovO._calc(self,variables, params)
        stdx = StdO._calc(self,[variables[0]],[params[0]])
        stdy = StdO._calc(self,[variables[1]],[params[0]])
        return cov / (stdx * stdy)

    def lookback_window(self, variable_lookbacks, params):
        return max(variable_lookbacks) + self._check_lookback(params[0]) - 1

_OPERATORS = [NegO(),AbsO(),PowerO(),SignO(),SinO(),CosO(),TanhO(),LogO(),DelayO(),DiffO(),PctO(),
              MaO(),MedO,SumO(),StdO(),MaxO(),MinO(),RankO(),SkewO(),KurtO(),VariO(),AutocorrO(),ZscoreO(),
              AddAO(),AddPO(),SubAO(),SubPO(),MulAO(),MulPO(),DivAO(),DivPO(),GreaterAO(),GreaterPO(), LessAO(), LessPO(),
              CovO(), CorrO()]       

OPERATORS = {} 
AVAILABLE_OPERATORS_STRING = ""
for Op in _OPERATORS:
    OPERATORS[Op.name] = Op
    AVAILABLE_OPERATORS_STRING = AVAILABLE_OPERATORS_STRING + "\n" + Op.example + " - " + Op.descriptor

