import numpy as np
from copy import deepcopy
from forage_env import ForageEnv
import itertools
env = ForageEnv('3x3')


def featurizeForage(env, ob, action):

    """
    We use the recency parameterization which uses features based on the history of
    observations. The feature vector is:

    [obj_reward(o, a), 1, count(location, internal), last_action_time_at_location(location, action, internal)]
    where internal is the internal state vector that we keep track of.
    """
    obj_reward = env.objReward(ob, action)
    c_li = env.getCountLocation(ob)
    c_la = env.getCountLocAct(ob, action)
    return np.array([obj_reward, 1, c_li, c_la])


def generate_expl_seq(env, num_actions = 4, depth=3):
    """
    Starting from a env state, do mcts on each subsequent spaces
    for each exploration, store it's feature vec.
    """
    # First element is dummy.
    expl_seq = []

    def do_generate(env_lst):
        """
        One-step generation for envs in env_lst
        """
        results = []
        next_envs = []
        for env in env_lst:
            for action in range(num_actions):
                branch = deepcopy(env)
                new_state, obj_rew, done, model = branch.step(action)
                # Use the old env to evaluate the "explored" state.
                encoded_state = featurizeForage(env, new_state, action)

                results.append(encoded_state)
                next_envs.append(branch)
        return results, next_envs

    envs = [env]
    while depth > 0:
        results, envs = do_generate(envs)
        expl_seq.append(results)
        depth -= 1

    retr = list(itertools.chain(*expl_seq))
    # Dummy element for the first.
    retr.insert(0, np.array([0, 1, 0, 0], dtype=np.float32))
    return retr
