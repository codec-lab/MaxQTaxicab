import random
from collections import defaultdict
import math
from typing import Sequence, Tuple, Hashable, Set, List, Optional, Union, Dict, TypeVar, Generic

import matplotlib.pyplot as plt

Element = TypeVar('Element', bound=Hashable)

class Distribution(Generic[Element]):
    def sample(self, rng: random.Random = random) -> Element:
        raise NotImplementedError
    def prob(self, x: Element) -> float:
        raise NotImplementedError
    
class DiscreteDistribution(Distribution,Generic[Element]):
    support : List[Element]
    def __init__(self, support : List[Element], probs : List[float]):
        assert len(support) == len(probs), "Support and probabilities must be the same length"
        assert abs(sum(probs) - 1) < 1e-6, "Probabilities must sum to 1"
        items = defaultdict(float)
        for x, p in zip(support, probs):
            items[x] += p
        support, probs = zip(*items.items())
        self.support = support
        self.probs = probs
    def sample(self, rng : random.Random = random) -> Element:
        return rng.choices(self.support, weights=self.probs)[0]
    def prob(self, x) -> float:
        if x not in self.support:
            return 0
        return self.probs[self.support.index(x)]
    def items(self) -> List[Tuple[Element, float]]:
        return zip(self.support, self.probs)
    def __repr__(self) -> str:
        return f"Discrete(support={self.support}, probs={self.probs})"
    def asdict(self) -> Dict[Element, float]:
        return dict(self.items())
    def fromdict(d : Dict) -> 'DiscreteDistribution[Element]':
        return DiscreteDistribution(*zip(*d.items()))

class Uniform(DiscreteDistribution):
    def __init__(self, support : List[Element]):
        super().__init__(support, [1/len(support)] * len(support))
    def __repr__(self) -> str:
        return f"Uniform(support={self.support})"

class Gaussian(Distribution[float]):
    def __init__(self, mean : float, std : float):
        self.mean = mean
        self.std = std
    def sample(self, rng : random.Random = random) -> float:
        return rng.gauss(mu=self.mean, sigma=self.std)
    def prob(self, x) -> float:
        return math.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (self.std * (2 * math.pi)**.5)
    def plot(self, ax : Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        xs = [self.mean + self.std * x / 100 for x in range(-500, 500)]
        ax.plot(xs, [self.prob(x) for x in xs], **kwargs)
        return ax

