import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from forage_env import ForageEnv

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

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

def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def plan(sy_feature_vec, sy_feature_theta, d, gamma, env):
    """
    Running MCTS to calculate the score function for each action, at certain observation.
    Return an array with length #action_num, corresponding to the Q-value of each action.

    We will use this Q-value later to sample an action.
    """

    from copy import copy
    num_actions = 4
    Q_array = []
    prob_array = []
    T = []
    for action in range(num_actions):
        branch = copy(env)
        new_state, obj_rew, done, model = branch.step(action)
        encoded_state = featurizeForage(branch, new_state, action)

        feature_vec = sess.run(sy_feature_vec, feed_dict={sy_feature_vec: encoded_state})

        designed_rew = sy_feature_theta * feature_vec

        Q_array.append(designed_rew)
        T.append(model["prob"])

    if d == 0:
        return Q_array
    else:
        prob_array = np.asarray(prob_array, dtype=np.float32)
        return tf.add_n(Q_array, T * gamma * tf.reduce_max(plan(sy_feature_vec, sy_feature_theta, d-1, gamma, env)))



def mainForagePGRD(feature_size=4, sess):

    env = ForageEnv('3x3')

    ob_dim = 1
    num_actions = 4

    sy_feature_theta = tf.get_variable("feature", [feature_size], dtype=tf.float32, initializer=tf.ones_initializer()) # Target feature vec.
    sy_feature_vec = tf.placeholder(shape=[None, feature_size], name="fp", dtype=tf.float32) # Feature placeholder.
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation

    # sy_h1 = tf.nn.relu(dense(sy_feature_vec, 64, "h1", weight_init=normc_initializer(1.0)))
    # sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05))

    sy_Q_na = plan(sy_feature_vec, sy_feature_theta, d, gamma, env) # [batch_size * 4]
    sy_logp_na = tf.nn.log_softmax(sy_Q_na)
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    sy_sampled_ac = categorical_sample_logits(sy_Q_na)[0]



    # Option 1: raw reward.
    sy_reward = sy_adv_n * sy_logprob_n
