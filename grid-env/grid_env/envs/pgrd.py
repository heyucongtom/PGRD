import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import itertools
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

def pathlength(path):
    return len(path["reward"])

def generate_expl_seq(env, num_actions = 4, depth=3):
    """
    Starting from a env state, do mcts on each subsequent spaces
    for each exploration, store it's feature vec.
    """
    # First element is dummy.
    expl_seq = [None]

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
    return list(itertools.chain(*expl_seq))

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def plan(sy_feature_no, sy_feature_theta, d, gamma):
    """
    Feed the feature and observation for d levels of exploration. (Feed on fly).
    Considering batching.
    We use a "tensorArray" to define tree node and leaf node.
    For leaf, we multiply feature_no with the feature_theta
    For node, we use tf.reduce_max(tf.gather(idx of leafs, tensorArray)) ...
    We return the max tensor

    sy_explore_no is the array with one starting point. [1, 4, 16 ...]
    """

    def isLeaf(idx, depth, branch_num=4):
        """
        Return given idx is a leaf node.
        assuming full 4-th tree.

        E.g: if depth = 2,
        then idx >= 5 is leaf. (5 -> 15)
        """
        if depth <= 1:
            return True
        else:
            prev_num = (1 - branch_num ** depth) / (1 - branch_num)
            # return tf.greater_equal(idx, prev_num)
            return idx >= prev_num

    def getLeaf(idx, depth, branch_num=4):
        """
        Given an idx, return a list of its nextlevel node idx.
        Assume full 4-tree.
        """
        if isLeaf(idx, depth):
            return None
        else:
            # Calculate relative level:
            lev, prev = 0, 0
            while lev < depth:
                prev += branch_num ** lev
                if prev > idx:
                    break
                lev += 1

            prev_num = (1 - branch_num ** lev) / (1 - branch_num)
            skip_elms = (idx - prev_num) * branch_num

            idx_start = (1 - branch_num ** (lev + 1)) / (1 - branch_num)
            leaves_idx = np.arange(idx_start+skip_elms, idx_start+skip_elms+branch_num, dtype=np.int32)
            return leaves_idx

    def calculate_node_tensor(sy_feature_theta, sy_feature_no, idx, depth):
        """
        Consider batch_size = 1, we have explore_array[1, 4, 16, 64]....
        The Q value of single leaf node is: ob * theta

        return the tensor calculating the Q at given idx.
        tensorArray.write()...
        """
        if idx == 0:
            R = sy_feature_no[idx] * 0
        else:
            R = sy_feature_no[idx] * sy_feature_theta

        if isLeaf(idx, depth):
            return R
        else:
            leaf_idx = getLeaf(idx, depth)
            return tf.add_n(R, gamma * tf.reduce_max(node_tensors.gather(leaf_idx)))

    def loop_body(node_tensors, i):

        # Loop from the leaf.
        node_tensor = calculate_node_tensor(sy_feature_theta, sy_feature_no, i, d)
        node_tensors = node_tensors.write(i, node_tensor)
        tf.subtract(i, 1)


    sy_n = tf.shape(sy_feature_no)[0] # Subject to revision.
    node_tensors = tf.TensorArray(tf.float32, size=sy_n, clear_after_read=False, infer_shape=False)

    loop_cond = lambda node_tensors, i: tf.greater_equal(i, 0)

    node_tensors, _ = tf.while_loop(loop_cond, loop_body, [node_tensors, sy_n-1], parallel_iterations=1)
    # The first elem in tensorArray stores the calculated result.
    return node_tensors.read(0)

class NnValueFunction(object):
    # Given ob, reward,
    # Fit a neural network
    coef = None
    def __init__(self, ob_dim=None):
        from sklearn.neural_network import MLPRegressor
        self.net = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', alpha=1e-4)
        self.fitted = False

    def fit(self, X, y, iter_num=1000, sess=None):
        self.net.fit(X, y)
        self.fitted = True

    def predict(self, X, sess=None):
        if not self.fitted:
            return np.zeros(X.shape[0])
        return self.net.predict(X)

    def preproc(self, X):
        # print(X.shape)
        return X

def mainForagePGRD(feature_size=4, depth=3, n_iter=100, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, animate=False, logdir=None):

    env = ForageEnv('3x3')

    ob_dim = 1
    num_actions = 4

    logz.configure_output_dir(logdir)
    vf = NnValueFunction()

    sy_feature_theta = tf.get_variable("feature", [feature_size], dtype=tf.float32, initializer=tf.ones_initializer()) # Target feature vec.
    sy_feature_no = tf.placeholder(shape=[None, int((1 - num_actions ** (depth + 1)) / (1 - num_actions)), feature_size], name="fp", dtype=tf.float32) # Feature placeholder.
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    # sy_explore_no = tf.placeholder(shape=[None, ob_dim], name="exploration", dtype=tf.float32)

    sy_logits_na = plan(sy_feature_no, sy_feature_theta, depth, gamma) # [batch_size * 4]
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32)


    sy_n = tf.shape(sy_feature_vec)[0]
    sy_logp_na = tf.nn.log_softmax(sy_logits_na)
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0]

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na)
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)


    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n)

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0


    for i in range(n_iter):
        with tf.Graph().as_default(), tf.Session() as sess:
            # Reset the default graph once in a while
            print("***** Iteration %i ******" % i)
            timesteps_this_batch = 0
            paths = []
            while True:
                ob = env.reset()
                terminated = False
                features, acs, rewards = [], [], []
                animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
                while True:
                    if animate_this_episode:
                        env.render()

                    feature_seq = generate_expl_seq(env)
                    features.append(feature_seq)

                    # Generate a observation sequence:
                    ac = sess.run(sy_sampled_ac, feed_dict={sy_feature_no: feature_seq})
                    acs.append(ac)
                    ob, rew, done, _ = env.step(ac)
                    rewards.append(rew)
                    if done:
                        break
                path = {"features" : np.array(features), "terminated" : terminated,
                        "reward" : np.array(rewards), "action" : np.array(acs)}
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch
            vtargs, vpreds, advs = [], [], []
            for path in paths:
                # rew_t already discounted.
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                vpred_t = vf.predict(path["observation"])
                adv_t = return_t - vpred_t
                advs.append(adv_t)
                vtargs.append(return_t)
                vpreds.append(vpred_t)

            feature_no = np.concatenate([path["observation"] for path in paths])
            ac_n = np.concatenate([path["action"] for path in paths])

            adv_n = np.concatenate(advs)
            standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
            vtarg_n = np.concatenate(vtargs)
            vpred_n = np.concatenate(vpreds)
            vf.fit(feature_no, vtarg_n)

            _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_feature_no:feature_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
            kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_feature_no:ob_no, sy_oldlogits_na:oldlogits_na})

            # Log diagnostics
            logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
            logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
            logz.log_tabular("KLOldNew", kl)
            logz.log_tabular("Entropy", ent)
            logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
            logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            # If you're overfitting, EVAfter will be way larger than EVBefore.
            # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
            logz.dump_tabular()

if __name__ == "__main__":
    if 1:
        # general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
        mainForagePGRD()
