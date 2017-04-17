import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class GridEnv(gym.Env):

    def __init__(self, nS, nA, P, isd):
        """
        Has the following members
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)

        (*) dictionary dict of dicts of lists, where
          P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS


        """
        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd # initial state distribution:
        self.lastaction = None # for rendering.

        # Define observation_space and action_space
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        transitions = self.P[self.s][action]
        i = categorical_sample([tup[0] for tup in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = action
        return (s, r, d, {"prob": p})

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def _render(self, mode='human', close=False):
        ...
