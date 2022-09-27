import os
import random
from MEC_AC_env_two_games import MEC_network
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()  # 禁用默認的即時執行模式(tensorflow 1轉2 需要)
import gym
import time
import random
import math
import collections

# MAX_N_MEC_NET = mp.cpu_count()
#
# MAX_GLOBAL_EP = 3000
# UPDATE_GLOBAL_ITER = 1

GAMMA = 0.9
ALPHA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.01  # learning rate for actor
LR_C = 0.05  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
COST_TO_CLOUD = 15

# State and Action Space


n_mobile_user = 3
N_mec_edge = 2
N_A = N_mec_edge + 1

N_S = 5  # Latency State or Latency State + action phi


class Actor(object):
    def __init__(self, scope, sess, n_nodes, lr=0.001, q_size=10):
        self.sess = sess
        self.lr = lr
        self.t = 1
        self.n_nodes = n_nodes
        self.n_actions = 3
        self.n_features = 3  # pStates, qStates, and cStates
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")  # Try different dimensions
        self.epsilon = 0.9
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.q_size = q_size

        with tf.variable_scope(scope + 'Actor'):  # 命名用e0Actor
            l1 = tf.layers.dense(  # hidden layer
                inputs=self.s,
                units=10,  # number of hidden units #35
                # activation=tf.nn.relu,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(  # output layer
                inputs=l1,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope(scope + 'exp_v'):
            # log_prob = tf.log(self.acts_prob[0, self.a])
            log_prob = tf.log(self.acts_prob)  # 自然對數函數
            self.exp_v = tf.reduce_mean(
                tf.math.reduce_sum(tf.math.multiply(log_prob, self.td_error)))  # advantage (TD_error) guided loss

        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                # Adam optimization algorithm (for stochastic optimization)
                -self.exp_v * .5)  # -.2  # minimize(-exp_v) = maximize(exp_v) #10.5
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(-self.exp_v*0.00005)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(-self.exp_v*0.005)
            # self.train_op = tf.train.RMSPropOptimizer(lr).minimize(-self.exp_v*.05)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}  # dictionary
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        # self.lr = min(1, self.lr * math.pow(1.000001, self.t))
        self.t += 1
        return exp_v

    def choose_action(self, s, total_tasks):
        prob_weights = self.sess.run(self.acts_prob, feed_dict={self.s: s[np.newaxis, :]})
        # print(prob_weights)
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        action = np.zeros(self.n_actions)
        # print("fuck2")
        for i in range(int(total_tasks)):
            index = np.random.choice(np.arange(self.n_actions), p=prob_weights.ravel())
            action[index] += 1
        return action

        #
        # s = s[np.newaxis, :]
        # probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        # #print("action prob", probs)
        #
        # """
        # Here we are using method 2.
        # We should consider possibility decreasing
        # """
        # #print(probs.ravel())
        # action = np.zeros(probs.ravel().shape)
        # #print("Total work", total_work)
        # for i in range(total_tasks):
        #     #print(probs.shape[1])
        #     index = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        #     action[index] += 1
        #
        # return action

    def reset(self):
        tf.reset_default_graph()


class Critic(object):
    def __init__(self, scope, sess, n_nodes, lr=0.001):
        self.sess = sess
        self.lr = lr
        self.t = 1
        n_features = 3
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope(scope + 'Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units #50
                # activation=tf.nn.relu,  # None
                activation=tf.nn.tanh,
                # tf.nn.tanh
                # tf.nn.selu
                # tf.nn.softplus
                # tf.nn.elu
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope(scope + 'squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(-self.loss * .5)  # for under 10 nodes .01

            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss*.05) #.5
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v * 10.5)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss*0.001)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v = self.sess.run(self.v, {self.s: s})
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        # self.lr = min(1, self.lr * math.pow(1.0000005, self.t))
        self.t += 1
        return td_error, v_, loss, v

    def reset(self):
        tf.reset_default_graph()


class Predictor(object):
    def __init__(self, scope, sess, n_nodes, lr=0.001):
        self.sess = sess
        self.lr = lr
        self.t = 1
        n_features = n_nodes + 1
        self.s = tf.placeholder(tf.float32, [1, pow(n_nodes, 2)-1], "action1")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next1")
        self.r = tf.placeholder(tf.float32, None, 'r1')

        with tf.variable_scope(scope + 'Predictor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=10,  # number of hidden units #50
                # activation=tf.nn.relu,  # None
                activation=tf.nn.tanh,
                # tf.nn.tanh
                # tf.nn.selu
                # tf.nn.softplus
                # tf.nn.elu
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope(scope + 'squared_TD_error'):
            self.td_error = self.v - self.r
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss)  # for under 10 nodes .01
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss*.05) #.5
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v * 10.5)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss*0.001)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.loss)

    def choose_action(self, s):
        s = np.array(s)
        s = s[np.newaxis, :]

        v = self.sess.run(self.v, {self.s: s})
        return v

    def learn(self, s, r, r_):
        r, r_ = np.array(r).reshape(1, 1), np.array(r_).reshape(1, 1)
        # r,r_ = r[np.newaxis, :], r_[np.newaxis, :]
        s = np.array(s)
        s = s[np.newaxis, :]
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.v: r_, self.r: r, self.s: s})
        # print("loss, r,r_",loss, r,r_)
        # self.lr = min(1, self.lr * math.pow(1.0000005, self.t))
        self.t += 1
        return td_error, loss

    def reset(self):
        tf.reset_default_graph()


class Edge(object):  # contain a local actor, critic, global critic
    def __init__(self, scope, lar=0.001, lcr=0.001, q_size=10, sess=None):
        self.n_nueron_ac = 5
        self.sess = sess
        self.la_r = lar
        self.lc_r = lcr
        # self.la_s = tf.placeholder(tf.float32, 1, [None, N_S], 'la_s')
        self.epsilon = 0.8
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.q_size = q_size
        self.local_actor = Actor(scope, self.sess, 3, self.la_r, self.q_size)
        self.local_critic = Critic(scope, self.sess, 3, self.lc_r)
        self.local_predictor = Predictor(scope, self.sess, 3, self.lc_r)


# No global training
def run(tr):
    import time
    GAMMA = 0.9

    COST_TO_CLOUD = 15

    # State and Action Space

    N_mec_edge = 2
    N_A = N_mec_edge + 1

    total_delay = []
    total_jobs = []
    total_drop = []
    total_time = 2000
    q_len = 0
    total_r = 0
    u = 0
    total_q_delay = 0
    total_drop = 0
    total_utility = 0
    total_state_value = 0
    total_s_delay = 0
    shared_ations = {}

    SESS = tf.Session()
    # Three edge networks plus a cloud
    # create networks
    mec_0 = MEC_network(task_arrival_rate=tr, num_nodes=1, Q_SIZE=50, node_num=0)
    s_0, total_work_0 = mec_0.reset()
    edge_0 = Edge(scope='e' + str(0), lar=0.001, lcr=0.01, q_size=50, sess=SESS)

    mec_1 = MEC_network(task_arrival_rate=tr, num_nodes=2, Q_SIZE=50, node_num=1)
    s_1, total_work_1 = mec_1.reset()
    edge_1 = Edge(scope='e' + str(1), lar=0.001, lcr=0.01, q_size=50, sess=SESS)

    mec_2 = MEC_network(task_arrival_rate=tr, num_nodes=3, Q_SIZE=50, node_num=2)
    s_2, total_work_2 = mec_2.reset()
    edge_2 = Edge(scope='e' + str(2), lar=0.001, lcr=0.01, q_size=50, sess=SESS)

    # edges_g0 = [Edge(scope='e'+str(i), lar=0.001, lcr=0.01, q_size=50, sess=SESS) for i in range(N_mec_edge)]

    # mec_1 = MEC_network(task_arrival_rate=tr, num_nodes=2, Q_SIZE=50)
    # s_1, total_work_1 = mec_1.reset()
    # edges_g1 = [Edge(scope='e'+str(i+2), lar=0.001, lcr=0.01, q_size=50, sess=SESS) for i in range(N_mec_edge)]

    SESS.run(tf.global_variables_initializer())

    q_len += s_0[1] + s_1[1] + s_2[1]  # the total queue len of all edge
    # q_len += sum(s_0[N_mec_edge:(N_mec_edge + N_mec_edge)]) + sum(s_1[N_mec_edge:(N_mec_edge + N_mec_edge)]) + \
    #          sum(s_2[N_mec_edge:(N_mec_edge + N_mec_edge)]) + sum(s_3[N_mec_edge:(N_mec_edge + N_mec_edge)]) + \
    #          sum(s_4[N_mec_edge:(N_mec_edge + N_mec_edge)]) + sum(s_5[N_mec_edge:(N_mec_edge + N_mec_edge)])

    c_0 = 1
    c_1 = 1
    c_2 = 1
    shared_ations[0] = [s_0[1], 0, 0, 0]
    shared_ations[1] = [0, s_1[1], 0, 0]
    shared_ations[2] = [0, 0, s_2[1], 0]

    s_0 = np.hstack((s_0, c_0))
    s_1 = np.hstack((s_1, c_1))
    s_2 = np.hstack((s_2, c_2))

    for i in range(total_time):
        # print("time", i, tr)

        total_work_0 = s_0[1]
        total_work_1 = s_1[1]
        total_work_2 = s_2[1]

        c_0 = edge_0.local_predictor.choose_action(np.hstack((shared_ations[1], shared_ations[2]))).flatten()[0]  # predict the other edge's price
        c_1 = edge_1.local_predictor.choose_action(np.hstack((shared_ations[0], shared_ations[2]))).flatten()[0]
        c_2 = edge_2.local_predictor.choose_action(np.hstack((shared_ations[0], shared_ations[1]))).flatten()[0]

        s_0 = np.hstack((s_0[:len(s_0) - 1], c_0))  # the other edge's price
        s_1 = np.hstack((s_1[:len(s_1) - 1], c_1))
        s_2 = np.hstack((s_2[:len(s_2) - 1], c_2))

        a0_pre = shared_ations[0]
        a1_pre = shared_ations[1]
        a2_pre = shared_ations[2]

        a0 = edge_0.local_actor.choose_action(s_0, total_work_0)
        a1 = edge_1.local_actor.choose_action(s_1, total_work_1)
        a2 = edge_2.local_actor.choose_action(s_2, total_work_2)

        # a2 = edges_g2[0].local_actor.choose_action(s_2, total_work_2)
        # a3 = edges_g3[0].local_actor.choose_action(s_3, total_work_3)
        # a4 = edges_g4[0].local_actor.choose_action(s_4, total_work_4)
        # a5 = edges_g5[0].local_actor.choose_action(s_5, total_work_5)

        shared_ations[0] = a0
        shared_ations[1] = a1
        shared_ations[2] = a2
        # shared_ations[3] = a3
        # shared_ations[4] = a4
        # shared_ations[5] = a5
        # First sharing - states (latency states)

        s_0_, total_work_0, r_0, d_0, q_d_0, new_task_0, avg_delay_0 = mec_0.step(shared_ations)  # s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
        s_1_, total_work_1, r_1, d_1, q_d_1, new_task_1, avg_delay_1 = mec_1.step(shared_ations)
        s_2_, total_work_2, r_2, d_2, q_d_2, new_task_2, avg_delay_2 = mec_2.step(shared_ations)

        if r_0 < 0 or r_1 < 0 or r_2 < 0:
            print(r_0, r_1)
            print("stop1")
            exit()
        r_0 = r_0 + shared_ations[0][1] * avg_delay_1 + shared_ations[0][2] * avg_delay_2
        r_1 = r_1 + shared_ations[1][0] * avg_delay_0 + shared_ations[1][2] * avg_delay_2
        r_2 = r_2 + shared_ations[2][0] * avg_delay_0 + shared_ations[2][1] * avg_delay_1

        if r_0 < 0 or r_1 < 0 or r_2 < 0:
            print("stop2")
            exit()

        c_0_ = avg_delay_0
        c_1_ = avg_delay_1
        c_2_ = avg_delay_2
        s_0_ = np.hstack((s_0_, c_0_))  # next state
        s_1_ = np.hstack((s_1_, c_1_))
        s_2_ = np.hstack((s_2_, c_2_))

        q_len += new_task_0 + new_task_1 + new_task_2   # +sum(new_task_1) +sum(new_task_2)+sum(new_task_3)+sum(new_task_4)+sum(new_task_5)

        td_error_0, v_0, _, v_0_ = edge_0.local_critic.learn(s_0, r_0, s_0_)
        td_error_1, v_1, _, v_1_ = edge_1.local_critic.learn(s_1, r_1, s_1_)
        td_error_2, v_2, _, v_2_ = edge_1.local_critic.learn(s_2, r_2, s_2_)

        edge_0.local_actor.learn(s_0, shared_ations[0], td_error_0)
        edge_1.local_actor.learn(s_1, shared_ations[1], td_error_1)
        edge_2.local_actor.learn(s_2, shared_ations[2], td_error_2)

        edge_0.local_predictor.learn(a0_pre, c_0_, c_0) # error
        edge_1.local_predictor.learn(a1_pre, c_1_, c_1)
        edge_2.local_predictor.learn(a2_pre, c_2_, c_2)

        s_0 = s_0_
        s_1 = s_1_
        s_2 = s_2_
        c_0 = c_0_
        c_1 = c_1_
        c_2 = c_2_


        ###########################
        edge_0.local_actor.lr = min(1, edge_0.local_actor.lr * math.pow(1.000001, i))  # learning rate
        edge_0.local_critic.lr = min(1, edge_0.local_critic.lr * math.pow(1.000001, i))

        edge_1.local_actor.lr = min(1, edge_1.local_actor.lr * math.pow(1.000001, i))
        edge_1.local_critic.lr = min(1, edge_1.local_critic.lr * math.pow(1.000001, i))

        edge_2.local_actor.lr = min(1, edge_2.local_actor.lr * math.pow(1.000001, i))
        edge_2.local_critic.lr = min(1, edge_2.local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g0)):
        #     edges_g0[i].local_actor.lr = min(1, edges_g0[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g0[i].local_critic.lr = min(1, edges_g0[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g1)):
        #     edges_g1[i].local_actor.lr = min(1, edges_g1[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g1[i].local_critic.lr = min(1, edges_g1[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g2)):
        #     edges_g2[i].local_actor.lr = min(1, edges_g2[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g2[i].local_critic.lr = min(1, edges_g2[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g3)):
        #     edges_g3[i].local_actor.lr = min(1, edges_g3[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g3[i].local_critic.lr = min(1, edges_g3[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g4)):
        #     edges_g4[i].local_actor.lr = min(1, edges_g4[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g4[i].local_critic.lr = min(1, edges_g4[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g3)):
        #     edges_g5[i].local_actor.lr = min(1, edges_g5[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g5[i].local_critic.lr = min(1, edges_g5[i].local_critic.lr * math.pow(1.000001, i))

        GAMMA = GAMMA / pow(1.00005, i)  # 1.00005
        # GAMMA = GAMMA / math.pow(1.000001, i)
        # ALPHA = ALPHA / math.pow(1.000001, i)

        # total_q_delay += q_delay_n0 #+q_d_1+q_d_2+q_d_3+q_d_4+q_d_5
        # total_drop += d_delay_n0 #d_0#+d_1+d_2+d_3+d_4+d_5
        # # total_s_delay += s_delay_n0#(r_0 - q_d_0 - 15 * d_0) #+(r_1 - q_d_1 - 15 * d_1)+(r_2 - q_d_2 - 15 * d_2)+(r_3 - q_d_3 - 15 * d_3)+\
        # #                  (r_4 - q_d_4 - 15 * d_4) + (r_5 - q_d_5 - 15 * d_5)
        # # total_utility += math.exp(-r)
        #
        # total_q_delay += q_delay_n0#/new_task_0[0] if new_task_0[0] > 0 else 0
        # total_s_delay += s_delay_n0#/new_task_0[0] if new_task_0[0] > 0 else 0
        # total_utility += math.exp(-((q_delay_n0+s_delay_n0+ d_delay_n0*15)/new_task_0[0])) if new_task_0[0] > 0 else 0 #+math.exp(-((r_1 - 15 * d_1) + 20 * (d_1 * 15)))\
        #                  # +math.exp(-((r_2 - 15 * d_2) + 20 * (d_2 * 15)))+math.exp(-((r_3 - 15 * d_3) + 20 * (d_3 * 15)))\
        #                  # +math.exp(-((r_4 - 15 * d_4) + 20 * (d_4 * 15)))+math.exp(-((r_4 - 15 * d_4) + 20 * (d_4 * 15)))
        #
        # # GAMMA = GAMMA / pow(1.0005, i_episode)
        total_r += r_0 + r_1 + r_2  # + r_1+r_2+r_3+r_4+r_5

    # print("GAMMA", GAMMA)
    print("task", q_len)
    # print(total_jobs)
    # print("drop", total_drop/total_time)
    # print(total_delay)
    # try:
    #     latency[tr] = sum(total_delay)/total_jobs
    # except:
    #     print(tr)
    tf.summary.FileWriter("logs/", SESS.graph)
    tf.reset_default_graph()
    return total_r / q_len, total_drop / total_time, total_q_delay, total_utility, total_s_delay
    # return sum(total_delay)/total_jobs, sum(total_drop)


if __name__ == "__main__":
    latency = []
    drop = []
    q_delay = []
    utility = []
    s_delay = []
    la = []
    dr = []

    for j in range(5):
        latency = []
        drop = []
        q_delay = []
        utility = []
        s_delay = []
        for i in range(1, 40):  # task arrival rate
            print(j, i)
            # i,r =pool.apply_async(func=run, args=(i,))
            # print((i,r))
            l, d, l_q, u, l_s = run(i)
            # ans.append(l)
            # drop.append(d)
            # p.start()
            # pros.append(p)
            latency.append(l)
            print(latency)
            drop.append(d)
            q_delay.append(l_q)
            utility.append(u)
            s_delay.append(l_s)
        la.append(latency)
        dr.append(drop)
    print(la)
    la = np.mean(np.array(la), axis=0)
    dr = np.mean(np.array(dr), axis=0)
    print(la)
    # la = np.mean(a, axis=0)

    ############### Measure performance only#############
    # import multiprocessing  as mp
    # import os
    # for rate in range(5,85,5):
    #     s = time.time()
    #     p = mp.Process(target=run, args=(rate,))
    #     p.start()
    #     print(p.pid)
    #     file = "ac_"+str(N_mec_edge)+"n_r"+str(rate)+"_activity_dis.txt"
    #     print(file)
    #     command = "psrecord "+str(p.pid)+ " --log /Users/hsiehli-tse/Desktop/Reseearch_2020_3/MEC_performance/"+ file+\
    #               " --duration 150 --interval 1 --include-children"
    #     print(command)
    #     os.system(command)
    #     os.system("kill -9 "+ str(p.pid))
    #     print(time.time()-s)

import pickle
#
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_l_v7.txt","wb") as fp:
#     pickle.dump(la, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_q_v7.txt","wb") as fp:
#     pickle.dump(q_delay, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_d_v7.txt","wb") as fp:
#     pickle.dump(dr, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_s_v7.txt","wb") as fp:
#     pickle.dump(s_delay, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_u_v7.txt","wb") as fp:
#     pickle.dump(utility, fp)

import matplotlib.pyplot as plt

x = range(1, 40)
plt.plot(x, la, color='#9D2EC5', marker='o', label='Distributed Actor Critic', linewidth=3.0)
# plt.plot(x, ac_12n_dis_l ,color= '#F5B14C',marker='o',label='Distributed Actor Critic (group = 1)', linewidth=3.0)
plt.title("add_ActorLearning")

# plt.xticks(range(35,60,5))
plt.xlabel('Average Task Arrivals per Slot', fontsize=13)
plt.ylabel('Average Service Delay', fontsize=13)
plt.tick_params(labelsize=11)
plt.legend(fontsize=9)
plt.show()
