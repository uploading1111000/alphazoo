import copy
from .operators import OPERATORS
#tree = reference to MCTS tree

class formula():
    def __init__(self,parent,pseudocode,code):
        self.parent = parent
        self.children = []
        self.arg_i = 0
        raise NotImplementedError

    def handle_code(self,code):
        self.arg_list = []
        pass

    def handle_pseudocode(self,pseudocode):
        pass

    def zoo_member(self):
        z_member = copy.copy(self)
        z_member.tree = None
        z_member.parent = None
        return z_member

    def apply(self, histories):
        #apply to each history in histories
        #return np array of shape (universe_n,>targets_n)
        pass

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