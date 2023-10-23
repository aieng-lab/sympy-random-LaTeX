import uuid


class PermutationStats(dict):

    def __init__(self):
        super().__init__()
        self['substitution_var'] = {}
        self['substitution_fnc'] = {}
        self['strategy_equality'] = {}
        self['strategy_inequality'] = {}
        self['strategy_swap'] = {}
        self['strategy_variables'] = {}
        self['strategy_random_formula'] = {}
        self['strategy_constants'] = {}
        self['strategy_distribute'] = {}
        self['formula_name_id'] = None
        self['id'] = str(uuid.uuid4())
        self['tex'] = {}
        self['original'] = None
        self['original_id'] = None

    def union(self, other):
        tmp = self.copy()
        for k, v in other.items():
            if k in tmp:
                if isinstance(tmp[k], dict) and isinstance(v, dict):
                    tmp[k].update(v)
                elif isinstance(tmp[k], list) and isinstance(v, list):
                    tmp[k] += v
                elif isinstance(tmp[k], str) and isinstance(v, str):
                    if k != 'id':
                        tmp[k] = v
                elif isinstance(tmp[k], bool) and isinstance(v, bool):
                    tmp[k] |= v
                else:
                    if v is not None:
                        tmp[k] = v
            else:
                tmp[k] = v
        return tmp


class PermutationStatsEntry:
    amount = 0

