import copy
from .operators import OPERATORS
from data import AVAILABLE_FIELDS
import json
import numpy as np

def validate_code(code,arguments,window=90): #not sure how to handle the window so far, seems silly to keep window around as a member of every class
    #this will probably have to update as we see whatever nonsensical things the LLM comes up with
    parameters = arguments[0].keys()
    for i, params in enumerate(arguments):
        if params.keys() != parameters:
            return f"Invalid Arguments. Argument {i} doesnt have the same parameters as argument {0}"

    available_variables = set(AVAILABLE_FIELDS)
    for i,line in enumerate(code):
        try:
            name = line["name"]
            params = line["param"]
            variables = line["input"]
            output = line["output"]
        except KeyError:
            return f"Invalid code. Line {i} is missing a key."
        if not name in OPERATORS:
            return f"Invalid code. Line {i} has an invalid operator."
        operator = OPERATORS[name]
        if len(variables) != operator.n_variables:
            return f"Invalid code. Line {i} has invalid number of inputs (expected {operator.n_variables}, received {len(variables)})"
        if len(params) != operator.n_parameters:
            return f"Invalid code. Line {i} has invalid number of parameters (expected {operator.n_parameters}, received {len(param)})"
        for variable in variables:
            if not variable in available_variables:
                return f"Invalid code. Line {i} expects variable {variable} but does not yet exist."
        for param in params:
            if not param in parameters:
                return f"Invalid code. Line {i} expects parameter {param} but this does not exist in arguments."
        if output in available_variables:
            return f"Invalid code. Line {i} outputs to variable {output} but this already exists."
        available_variables.add(output)
    if not "alpha" in available_variables:
        return "Invalid code. No alpha variable available"
    return validate_lookback(code,arguments,window)
        
def validate_lookback(code,arguments,window):
    for j, parameters in enumerate(arguments):
        available_variables = {}
        for field in AVAILABLE_FIELDS:
            available_variables[field] = 1
        for i,line in enumerate(code):
            name = line["name"]
            params = line["param"]
            variables = line["input"]
            output = line["output"]
            operator = OPERATORS[name]
            variable_lookbacks = [available_variables[variable] for variable in variables]
            param_values = [parameters[param_name] for param_name in params]
            lookback_window = operator.lookback_window(variable_lookbacks,param_values)
            available_variables[output] = lookback_window
        lookback = np.max(available_variables.values())
        if lookback > window:
            return f"Argument {j} produces lookback window larger than allowed (Argument: {lookback}, permitted window: {window})"
    return "valid"
        
    
    
class Formula():
    def __init__(self,parent): #dict-form pseudocode and code, might make it easier to errorcheck
        self.parent = parent
        self.children = []
        self.arg_i = 0

    def add_code(self,formula):
        self.arg_list = formula["arguments"]
        self.code = formula["formula"]
        self.formula_string = json.dumps(formula,indent=5)
        return validate_code(self.code,self.arg_list)

    def add_pseudocode(self,pseudocode):
        self.name = pseudocode["name"]
        self.description = pseudocode["description"]
        self.pseudocode = pseudocode["pseudo_code"]
        self.portrait = json.dumps(pseudocode,indent=5)

    def zoo_member(self):
        z_member = copy.copy(self)
        z_member.parent = None
        z_member.children = None
        return z_member

    def apply(self, histories): #this is going to be a big function!!!!
        #apply to each history in histories
        #return np array of shape (universe_n,>targets_n)
        parameters = self.arg_list[self.arg_i]
        for history in histories:
            variables = copy.copy(history)
            for line in self.code:
                name = line["name"]
                params = line["param"]
                variables = line["input"]
                output = line["output"]
                operator = OPERATORS[name]
                


    def apply_args(self,i):
        self.arg_i = i

    def best_args(self,i):
        self.arg_list = [self.arg_list[i]]
        self.arg_i = 0

    def alpha(self):
        #return pseudocode or formula or whatever with best params
        pass

    def refinement_history(self):
        #return refinement history from parents (recursively?)
        pass