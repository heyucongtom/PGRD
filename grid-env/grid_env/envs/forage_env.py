import numpy as np
import sys
from six import StringIO, b

from gym import utils
import grid_env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Bar(object):

    def __init__(self, grid1, grid2):
        self.g1 = grid1
        self.g2 = grid2
        self.lst = (self.g1, self.g2)

    def __getitem__(self, key):
        return self.lst[key]

    def __str__(self):
        s = "BarObject{0}-{1}"
        return s.format(self.g1, self.g2)

    def __repr__(self):
        return self.__str__()



MAPS = {
    '3x3': {
        'layout':[
            'SEE',
            'EEE',
            'EEG'
        ], 'bars': [
            Bar((0, 1), (1, 1)),
            Bar((0, 2), (1, 2)),
            Bar((1, 1), (2, 1)),
            Bar((1, 2), (2, 2))
        ]}
}



class ForageEnv(grid_env.GridEnv):

    """
    The environment is make of empty spaces (E) and goal (G).
    In order to replicate the paper settings, we provide layout and bar objectes, too.
    """

    def __init__(self, map_name=None, is_slippery=False):

        if map_name == None:
            raise ValueError("Must provide map_name")
        desc = MAPS[map_name]['layout']

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nA = nA = 4
        self.nS = nS = nrow * ncol # Not applicable in Atari scene;

        self.timestep = 0

        # c(l, i) is the number of time steps since the agent has visited location
        self.internal = {
            "lastVisitAt": {},
            "lastLocActionAt": {}
        }

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col


        def inc(row, col, a):
            if a == 0: # left
                col = max(col - 1, 0)
            if a == 1: # down
                row = min(row+1, nrow-1)
            if a == 2: # right
                col = min(col+1, ncol-1)
            if a == 3: # up
                row = max(row-1, 0)
            return row, col

        def isBar(state, newstate):
            for bar in MAPS[map_name]['bars']:
                s0 = to_s(*bar[0])
                s1 = to_s(*bar[1])
                # print(s0, s1, state, newstate)
                if (s0, s1) == (state, newstate) or (s0, s1) == (newstate, state):
                    return True
            return False

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(self.nA):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'G':
                        li.append((1.0, s, 1, True)) # Transition model if done.
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, a)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                if isBar(s, newstate):
                                    newstate = s
                                    newletter = desc[row, col]

                                reward = float(newletter == b'G')
                                isdone = bytes(newletter in b'G')
                                li.append((0.8 if a == b else 0.1, newstate, reward, isdone))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            if isBar(s, newstate):
                                newstate = s
                                newletter = desc[row, col]

                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(ForageEnv, self).__init__(nS, nA, P, isd)

    def objReward(self, ob, action):

        """
        This function computes the true reward for (observation, action) pair.
        """
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        li = self.P[ob][action]
        sumReward = 0
        for tup in li:
            prob, _1, rew, _2 = tup
            sumReward += prob * rew
        return sumReward

    def getCountLocation(self, s):
        # Compute c(location, i)
        c_li = 0.8
        if s in self.internal["lastVisitAt"]:

            c_li = self.timestep - self.internal["lastVisitAt"][s]
            print("Called!", s, self.timestep, self.internal)
            c_li = 1 - 1.0 / c_li
        return c_li

    def getCountLocAct(self, s, a):

        # compute c(location, action, i)
        c_la = 1
        if (s, a) in self.internal["lastLocActionAt"]:
            c_la = self.timestep - self.internal["lastLocActionAt"][(s, a)]
            c_la = 1 - 1.0 / c_la
        return c_la

    def _step(self, a):

        """
        # To collect a trajectory, we need to reset the environment every time.
        # Thus we store the internal feature vectors in the model.
        # In the later training, we collect a trajectory of those vectors (internal_t0, internal_t1, ...),
        # and feed them into the network to train the network.
        """

        self.internal["lastVisitAt"][self.s] = self.timestep
        self.internal["lastLocActionAt"][(self.s, a)] = self.timestep
        s, r, d, model = super()._step(a)
        self.timestep += 1

        return (s, r, d, model)
