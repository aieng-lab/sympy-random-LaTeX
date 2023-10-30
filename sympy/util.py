import random


class RandomTruthValue:

    def __init__(self, probability_true=0.5):
        self.probability_true = probability_true

    def __bool__(self):
        return self._random_state()

    def _random_state(self):
        return random.random() < self.probability_true

    def __hash__(self):
        return hash(self._random_state())

class RandomDecidedTruthValue(RandomTruthValue):

    def __init__(self, probability_true=0.5):
        super().__init__(probability_true)
        self.decide()

    def __bool__(self):
        return self.state

    def __hash__(self):
        return hash(self.state)

    def _random_state(self):
        return self.state

    def decide(self):
        self.state = super()._random_state()

class RandomChoice:

    def __init__(self, choices, weights=None):
        assert(len(choices) > 0)
        if weights is not None:
            assert(len(choices) == len(weights))
        else:
            weights = len(choices) * [1]
        self.choices = choices
        self.weights = weights

    def _random_state(self):
        return random.choices(self.choices, weights=self.weights)[0]

    def __hash__(self):
        return hash(self._random_state())

    def __eq__(self, other):
        return self._random_state() == other

    def __mul__(self, other):
        return self._random_state() * other

    def strip(self):
        state = self._random_state()
        if isinstance(state, str):
            return state.strip()
        elif state is None:
            return state
        else:
            raise ValueError("Can not strip value %s of class %s" % (state, type(state)))

    def __str__(self):
        return str(self._random_state())

class RandomDecidedChoice(RandomChoice):

    def __init__(self, choices, weigths=None):
        super().__init__(choices, weights=weigths)
        self.decide()

    def __eq__(self, other):
        return self.state == other

    def __hash__(self):
        return hash(self.state)

    def _random_state(self):
        return self.state

    def decide(self):
        self.state = super()._random_state()

    def __mul__(self, other):
        return self.state * other