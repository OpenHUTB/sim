import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass


class NegativeExpDistanceWithHitBonus(BaseFunction):

  def __init__(self, k):
    self.k = k

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 8
    elif info["inside_target"]:
      return 0
    else:
      if callable(self.k):
        k = self.k()
      else:
        k = self.k
      return (np.exp(-dist * k) - 1) / 10

  def __repr__(self):
    return "NegativeExpDistanceWithHitBonus"
