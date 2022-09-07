import numpy as np
import time
import collections
from copy import deepcopy
import time
import collections

np.random.seed(int(time.time()))
COST_TO_CLOUD = 15
# self.Q_SIZE = 25


pT = collections.defaultdict()
pT[0] = [[0.54287417, 0.45712583], [0, 0, 0], [0, 0, 0], [0.4376822, 0.5623178]]
pT[1] = [[0.84451706, 0.15548294], [0, 0, 0], [0, 0, 0], [0.70195544, 0.29804456]]
pT[2] = [[0.33072517, 0.66927483], [0, 0, 0], [0, 0, 0], [0.19587869, 0.80412131]]
pT[3] = [[0.33072517, 0.66927483], [0, 0, 0], [0, 0, 0], [0.19587869, 0.80412131]]
pT[4] = [[0.84451706, 0.15548294], [0, 0, 0], [0, 0, 0], [0.70195544, 0.29804456]]
pT[5] = [[0.33072517, 0.66927483], [0, 0, 0], [0, 0, 0], [0.4376822, 0.5623178]]
pT[6] = [[0.68013973, 0.31986027], [0, 0, 0], [0, 0, 0], [0.58572196, 0.41427804]]
pT[7] = [[0.8769391, 0.1230609], [0, 0, 0], [0, 0, 0], [0.484112, 0.515888]]
pT[8] = [[0.7192134, 0.2807866], [0, 0, 0], [0, 0, 0], [0.50275623, 0.49724377]]
pT[9] = [[0.70085048, 0.29914952], [0, 0, 0], [0, 0, 0], [0.61134567, 0.38865433]]
pT[10] = [[0.86149522, 0.13850478], [0, 0, 0], [0, 0, 0], [0.4549394, 0.5450606]]
pT[11] = [[0.11580692, 0.88419308], [0, 0, 0], [0, 0, 0], [0.3942863, 0.6057137]]

# pT[0] = [[0.04055034, 0.92525759, 0.03419207], [0.0447594, 0.45980075, 0.49543985], [0,0,0], [0.29821074,0.59653676, 0.1052525]]
# pT[1] = [[0.09839428, 0.44264436, 0.45896136], [0.62303035, 0.27488022, 0.10208943],[0,0,0], [0.24017474, 0.09779999, 0.66202527]]
# pT[2] = [[0.3117113, 0.57842914, 0.10985956], [0.24115865, 0.48706442, 0.27177693], [0,0,0], [0.52267071, 0.37230537, 0.10502393]]
# pT[3] = [[0.08148805, 0.45368999, 0.46482195],[0.4832546, 0.29815309, 0.21859231], [0,0,0], [0.46305416, 0.37684479, 0.16010105]]
# pT[4] = [[0.42037853, 0.56566125, 0.01396022], [0.74386348, 0.21466623, 0.04147029], [0,0,0], [0.40730214, 0.07149202, 0.52120583]]
# pT[5] = [[0.04055034, 0.92525759, 0.03419207], [0.0447594, 0.45980075, 0.49543985], [0,0,0], [0.29821074,0.59653676, 0.1052525]]
# pT[6] = [[0.09839428, 0.44264436, 0.45896136], [0.62303035, 0.27488022, 0.10208943],[0,0,0], [0.24017474, 0.09779999, 0.66202527]]
# pT[7] = [[0.3117113, 0.57842914, 0.10985956], [0.24115865, 0.48706442, 0.27177693], [0,0,0], [0.52267071, 0.37230537, 0.10502393]]
# pT[8] = [[0.08148805, 0.45368999, 0.46482195],[0.4832546, 0.29815309, 0.21859231], [0,0,0], [0.46305416, 0.37684479, 0.16010105]]
# pT[9] = [[0.42037853, 0.56566125, 0.01396022], [0.74386348, 0.21466623, 0.04147029], [0,0,0], [0.40730214, 0.07149202, 0.52120583]]
# pT[10] = [[0.04055034, 0.92525759, 0.03419207], [0.0447594, 0.45980075, 0.49543985], [0,0,0], [0.29821074,0.59653676, 0.1052525]]
# pT[11] = [[0.09839428, 0.44264436, 0.45896136], [0.62303035, 0.27488022, 0.10208943],[0,0,0], [0.24017474, 0.09779999, 0.66202527]]
# pT[12] = [[0.3117113, 0.57842914, 0.10985956], [0.24115865, 0.48706442, 0.27177693], [0,0,0], [0.52267071, 0.37230537, 0.10502393]]
# pT[13] = [[0.08148805, 0.45368999, 0.46482195],[0.4832546, 0.29815309, 0.21859231], [0,0,0], [0.46305416, 0.37684479, 0.16010105]]
# pT[14] = [[0.42037853, 0.56566125, 0.01396022], [0.74386348, 0.21466623, 0.04147029], [0,0,0], [0.40730214, 0.07149202, 0.52120583]]
# pT[15] = [[0.04055034, 0.92525759, 0.03419207], [0.0447594, 0.45980075, 0.49543985], [0,0,0], [0.29821074,0.59653676, 0.1052525]]
# pT[16] = [[0.09839428, 0.44264436, 0.45896136], [0.62303035, 0.27488022, 0.10208943],[0,0,0], [0.24017474, 0.09779999, 0.66202527]]
# pT[17] = [[0.3117113, 0.57842914, 0.10985956], [0.24115865, 0.48706442, 0.27177693], [0,0,0], [0.52267071, 0.37230537, 0.10502393]]
# pT[18] = [[0.08148805, 0.45368999, 0.46482195],[0.4832546, 0.29815309, 0.21859231], [0,0,0], [0.46305416, 0.37684479, 0.16010105]]
# pT[19] = [[0.42037853, 0.56566125, 0.01396022], [0.74386348, 0.21466623, 0.04147029], [0,0,0], [0.40730214, 0.07149202, 0.52120583]]
# pT[20] = [[0.11401054, 0.29576703, .59022243], [0.06809479, 0.78398124, 0.14792397], [0, 0, 0],[0.09483194, 0.17024863, 0.73491943]]
# pT[21] = [[0.37504972, 0.4799627, .14498758], [0.72731018, 0.1939028, 0.07878702], [0, 0, 0],[0.52405994, 0.22625999, 0.24968007]]
# pT[22] = [[0.36744668, 0.03363974, .59891359], [0.24764197, 0.48411535, 0.26824267], [0, 0, 0],[0.27774519, 0.05746011, 0.6647947]]
# pT[23] = [[0.78454334, 0.00121103, .21424563], [0.13702824, 0.66701449, 0.19595727], [0, 0, 0],[0.59591727, 0.11253978, 0.29154295]]
# pT[24] = [[0.19793005, 0.77990464, .02216531], [0.66515888, 0.17167272, 0.1631684], [0, 0, 0],[0.35323075, 0.49613314, 0.15063612]]
# pT[25] = [[0.19154744, 0.43568636, .37276619], [0.26818315, 0.58910063, 0.14271622], [0, 0, 0],[0.04970944, 0.03015975, 0.92013081]]
# pT[26] = [[0.05064329, 0.42692284, .52243387], [0.04837133, 0.46575382, 0.48587485], [0, 0, 0], [0.18334541, 0.59247208, 0.22418251]]
# pT[27] = [[0.69611964, 0.23232221, .07155815], [0.35121869, 0.07025531, 0.578526, ], [0, 0, 0],[0.37768088, 0.43935213, 0.18296699]]
# pT[28] = [[0.22237911, 0.16609108, .6115298], [0.23664979, 0.54806393, 0.21528627], [0, 0, 0],[0.93711848, 0.04247276, 0.02040876]]
# pT[29] = [[0.53261102, 0.29433922, .17304976], [0.16235767, 0.14253586, 0.69510647], [0, 0, 0],[0.62652985, 0.11505098, 0.25841917]]
# pT[30] = [[0.22620664, 0.19207079, .58172257], [0.03886022, 0.84725822, 0.11388155], [0, 0, 0],[0.14209633, 0.83364489, 0.02425878]]
# pT[31] = [[0.14224647, 0.27159556, .58615798], [0.28462829, 0.70343145, 0.01194026], [0, 0, 0],[0.00915397, 0.40301683, 0.5878292]]
# pT[32] = [[0.49558362, 0.47340171, .03101467], [0.5845193, 0.25645912, 0.15902159], [0, 0, 0],[0.63841531, 0.2128526, 0.14873209]]
# pT[33] = [[0.15776987, 0.28904635, .55318377], [0.33842834, 0.51609551, 0.14547615], [0, 0, 0],[0.2144405, 0.71312556, 0.07243393]]
# pT[34] = [[0.29889499, 0.61651677, .08458824], [0.33267594, 0.37544003, 0.29188404], [0, 0, 0],[0.54330465, 0.1462003, 0.31049504]]
# pT[35] = [[0.01101036, 0.18962047, .79936917], [0.10339297, 0.28188314, 0.61472389], [0, 0, 0],[0.48803609, 0.40986509, 0.10209882]]
# pT[36] = [[0.7174602, 0.11821355, .16432626], [0.91179766, 0.04627241, 0.04192993], [0, 0, 0],[0.07374196, 0.3053656, 0.62089244]]
# pT[37] = [[0.01018571, 0.13345749, .8563568], [0.18980234, 0.75414805, 0.0560496], [0, 0, 0],[0.61908812, 0.08250629, 0.2984056]]
# pT[38] = [[0.32060135, 0.53012872, .14926992], [0.59782628, 0.3263191, 0.07585462], [0, 0, 0],[0.00385943, 0.57639844, 0.41974214]]
# pT[39]=[[ 0.34432096,0.55805301,.09762603],[ 0.82885594,0.04325355,0.12789051],[0,0,0],[ 0.03367989,0.38357356,0.58274655]]

# pT[0] = [[ 0.68013973,  0.31986027], [0, 0, 0],[0, 0, 0],[ 0.58572196,  0.41427804]]
# pT[1] = [[ 0.8769391 , 0.1230609], [0, 0, 0],[0, 0, 0],[ 0.484112,  0.515888]]
# pT[2] = [[ 0.7192134,  0.2807866], [0, 0, 0],[0, 0, 0],[ 0.50275623,  0.49724377]]
# pT[3] = [[ 0.01191055,  0.98808945],[0, 0, 0],[0, 0, 0],[ 0.88676471,  0.11323529]]
# pT[4] = [[ 0.25892978,  0.74107022], [0, 0, 0],[0, 0, 0],[ 0.11365471,  0.88634529]]
# pT[5] = [[ 0.41634011,  0.58365989], [0, 0, 0],[0, 0, 0],[ 0.04752128,  0.95247872]]
# pT[6] = [[ 0.70085048,  0.29914952], [0, 0, 0],[0, 0, 0],[ 0.61134567,  0.38865433]]
# pT[7] = [[ 0.86149522,  0.13850478], [0, 0, 0],[0, 0, 0],[ 0.4549394,  0.5450606]]
# pT[8] = [[ 0.11580692,  0.88419308],[0, 0, 0],[0, 0, 0],[ 0.3942863,  0.6057137]]
# pT[9] = [[ 0.99637448,  0.00362552], [0, 0, 0],[0, 0, 0],[ 0.84443942,  0.15556058]]


cT = collections.defaultdict()
cT[2] = [0.36, 0.30, 0.34]
cT[5] = [0.23, 0.21, 0.56]
cT[10] = [0.57, 0.06, 0.37]


class MEC_network:
    def __init__(self, task_arrival_rate, num_nodes, Q_SIZE, node_num):
        self.node_num = node_num
        self.num_nodes = num_nodes
        # self.num_group = num_group
        # self.p_state = [np.random.choice([4,4]) for i in range(self.num_nodes)]
        # self.q_state = [np.random.choice(5) for i in range(self.num_nodes)]
        # self.c_state = [[np.random.choice([0.2, 0.5, 1]) if j != num_nodes else COST_TO_CLOUD
        #                  for j in range(num_nodes + 1)] for i in range(num_nodes + 1)]
        self.p_state = np.random.choice([4, 4])
        self.q_state = np.random.choice(5)
        self.task_arrival_rate = task_arrival_rate
        self.weight_q = 1
        self.weight_d = COST_TO_CLOUD
        self.weight_s = 1
        self.Q_SIZE = Q_SIZE
        self.p_a = 0

    # def shipDelay(self, phi):
    #
    #     cost_state = self.c_state
    #     process_state = self.p_state
    #     queue_state = self.q_state
    #     num_nodes = self.num_nodes
    #     q_bar = self.Q_SIZE
    #
    #     # print("sac", phi)
    #     # print("sc", cost_state)
    #     # print("sp", process_state)
    #     # print("sq", queue_state)
    #
    #     # for t in phi:
    #     #     if t > q_bar:
    #     #         return float('inf')
    #     cost = 0
    #     avg = 0
    #     needed, queue = deepcopy(queue_state), deepcopy(queue_state)
    #     needed.append(0)
    #     queue.append(0)
    #     requester, server = [], []
    #     #print(needed)
    #     #print(phi)
    #     #print(queue)
    #     for i in range(len(process_state)+1):
    #         needed[i] = phi[i] - queue[i]  #if phi[i] > queue[i] else 0
    #         if needed[i] > 0:
    #             server.append([i, needed[i]])
    #         elif needed[i] < 0:
    #             requester.append([i, needed[i]])
    #     #print(server)
    #     #print(requester)
    #     # Calculate shipping
    #     shipDelay = 0
    #     while server and requester:
    #         if len(server) > len(requester):
    #             if len(requester) == 1:
    #                 for s in server:
    #                     shipDelay += (cost_state[requester[0][0]][s[0]]) * s[1]
    #                     requester[0][1] += s[1]
    #                     s[1] = 0
    #             else:
    #                 for r in requester:
    #                     x = r[0]
    #                     stack = []
    #                     for s in server:
    #                         if s[1] <= 0:
    #                             continue
    #                         y = s[0]
    #                         if len(stack) == 0:
    #                             stack.append([x, y, cost_state[x][y]])
    #                         else:
    #                             if cost_state[x][y] < stack[-1][2]:
    #                                 stack.append([x, y, cost_state[x][y]])
    #
    #                     cost = stack[-1][2]
    #                     ansServer = stack[-1][1]
    #
    #                     shipDelay += 1 * cost
    #                     r[1] += 1
    #                     for s in server:
    #
    #                         if s[0] == ansServer:
    #                             s[1] -= 1
    #
    #         elif len(server) == len(requester):
    #             for r in requester:
    #                 x = r[0]
    #                 stack = []
    #                 for s in server:
    #                     y = s[0]
    #                     if s[1] == 0:
    #                         continue
    #                     if len(stack) == 0:
    #                         stack.append([x, y, cost_state[x][y]])
    #                     else:
    #                         if cost_state[x][y] < stack[-1][2]:
    #                             stack.append([x, y, cost_state[x][y]])
    #
    #                 cost = stack[-1][2]
    #                 ansServer = stack[-1][1]
    #
    #                 shipDelay += 1 * cost
    #                 r[1] += 1
    #                 for s in server:
    #
    #                     if s[0] == ansServer:
    #                         s[1] -= 1
    #
    #         else:
    #             for s in server:
    #                 x = s[0]
    #                 stack = []
    #                 for r in requester:
    #                     y = r[0]
    #                     if r[1] == 0:
    #                         continue
    #                     if len(stack) == 0:
    #                         stack.append([x, y, cost_state[x][y]])
    #                     else:
    #                         if cost_state[x][y] < stack[-1][2]:
    #                             stack.append([x, y, cost_state[x][y]])
    #                 cost = stack[-1][2]
    #                 ansReques = stack[-1][1]
    #
    #                 shipDelay += 1 * cost
    #                 s[1] -= 1
    #                 for r in requester:
    #                     if r[0] == ansReques:
    #                         r[1] += 1
    #
    #         server.sort(key=lambda x: x[1], reverse=True)
    #         requester.sort(key=lambda x: x[1], reverse=True)
    #         for i in range(len(server)):
    #             if server[-1][1] == 0:
    #                 server.pop()
    #         for i in range(len(requester)):
    #             if requester[-1][1] == 0:
    #                 requester.pop()
    #
    #         # cost = cost * 1.2
    #         # cost += cost_state*phi[1]# + (float(phi[1])/4)
    #
    #     if len(requester) > 0:
    #         for r in requester:
    #             r_ID = r[0]
    #             r_tasks = r[1]
    #             shipDelay += (cost_state[r_ID][num_nodes] * r_tasks)
    #     #print(shipDelay)
    #     return shipDelay
    #
    # def shipDelay_n0(self, phi, cost_state, process_state, queue_state, q_bar):
    #     # print("sp", phi)
    #     # print("sc", cost_state)
    #     # print("sp", process_state)
    #     # print("sq", queue_state)
    #
    #     cost = 0
    #     avg = 0
    #     needed, queue = deepcopy(queue_state), deepcopy(queue_state)
    #     needed.append(0)
    #     queue.append(0)
    #     requester, server = [], []
    #     # print(needed)
    #     # print(phi)
    #     # print(queue)
    #     for i in range(len(process_state) + 1):
    #         needed[i] = phi[i] - queue[i]  #if phi[i] > queue[i] else 0
    #         if needed[i] > 0:
    #             server.append([i, needed[i]])
    #         elif needed[i] < 0:
    #             requester.append([i, needed[i]])
    #
    #     if 0 in server:
    #         return 0
    #     # print(server)
    #     # print(requester)
    #     # Calculate shipping
    #     ship_delay_for_n0 = 0
    #     while server and requester:
    #         if len(server) > len(requester):
    #             if len(requester) == 1:
    #                 if requester[0][0] == 0:
    #                     for s in server:
    #                         ship_delay_for_n0 += (cost_state[requester[0][0]][s[0]]) * s[1]
    #                         requester[0][1] += s[1]
    #                         s[1] = 0
    #                 else:
    #                     return ship_delay_for_n0
    #             else:
    #                 for r in requester:
    #                     x = r[0]
    #                     stack = []
    #                     for s in server:
    #                         if s[1] <= 0:
    #                             continue
    #                         y = s[0]
    #                         if len(stack) == 0:
    #                             stack.append([x, y, cost_state[x][y]])
    #                         else:
    #                             if cost_state[x][y] < stack[-1][2]:
    #                                 stack.append([x, y, cost_state[x][y]])
    #
    #                     cost = stack[-1][2]
    #                     ansServer = stack[-1][1]
    #
    #                     if r[0] == 0:
    #                         ship_delay_for_n0 += 1 * cost
    #
    #                     r[1] += 1
    #                     for s in server:
    #                         if s[0] == ansServer:
    #                             s[1] -= 1
    #
    #
    #         elif len(server) == len(requester):
    #             for r in requester:
    #                 x = r[0]
    #                 stack = []
    #                 for s in server:
    #                     y = s[0]
    #                     if s[1] == 0:
    #                         continue
    #                     if len(stack) == 0:
    #                         stack.append([x, y, cost_state[x][y]])
    #                     else:
    #                         if cost_state[x][y] < stack[-1][2]:
    #                             stack.append([x, y, cost_state[x][y]])
    #
    #                 cost = stack[-1][2]
    #                 ansServer = stack[-1][1]
    #
    #                 if r[0] == 0:
    #                     ship_delay_for_n0 += 1 * cost
    #
    #                 r[1] += 1
    #                 for s in server:
    #
    #                     if s[0] == ansServer:
    #                         s[1] -= 1
    #
    #         else:
    #             for s in server:
    #                 x = s[0]
    #                 stack = []
    #                 for r in requester:
    #                     y = r[0]
    #                     if r[1] == 0:
    #                         continue
    #                     if len(stack) == 0:
    #                         stack.append([x, y, cost_state[x][y]])
    #                     else:
    #                         if cost_state[x][y] < stack[-1][2]:
    #                             stack.append([x, y, cost_state[x][y]])
    #                 cost = stack[-1][2]
    #                 ansReques = stack[-1][1]
    #
    #                 if ansReques == 0:
    #                     ship_delay_for_n0 += 1 * cost
    #
    #
    #                 s[1] -= 1
    #                 for r in requester:
    #                     if r[0] == ansReques:
    #                         r[1] += 1
    #
    #         server.sort(key=lambda x: x[1], reverse=True)
    #         # requester.sort(key=lambda x: x[1], reverse=True)
    #         for i in range(len(server)):
    #             if server[-1][1] == 0:
    #                 server.pop()
    #         for r in requester:
    #             if r[1] == 0:
    #                 if r[0] == 0:
    #                     return ship_delay_for_n0
    #                 requester.remove(r)
    #         if len(requester) == 0:
    #             return ship_delay_for_n0
    #         # cost = cost * 1.2
    #         # cost += cost_state*phi[1]# + (float(phi[1])/4)
    #
    #
    #
    #
    #     if len(requester) > 0:
    #         for r in requester:
    #             r_ID = r[0]
    #             r_tasks = abs(r[1])
    #             if r[0] == 0:
    #                 ship_delay_for_n0 += (cost_state[r_ID][self.num_nodes] * r_tasks)
    #     # print(shipDelay)
    #
    #     return ship_delay_for_n0

    def reset(self):
        # observation = np.zeros((self.num_nodes, 2)) # observation includes processing and queue states
        # for l1 in range(self.num_nodes + 1):
        #     for l2 in range(self.num_nodes):
        #         if l1 == l2:
        #             self.c_state[l1][l2] = 0
        #             continue
        #         elif l1 < l2:
        #             cost = np.random.choice([0.2, 0.5, 1], p=cT[int(self.c_state[l1][l2]*10)])
        #             # cost = 0
        #             self.c_state[l1][l2] = np.round(cost,2)
        #         else:
        #             self.c_state[l1][l2] = np.round(self.c_state[l2][l1], 2)
        #             # self.c_state[l1][l2] = 0
        #     if l1 == self.num_nodes:
        #         continue
        #     self.c_state[l1][self.num_nodes] = COST_TO_CLOUD
        # print(self.c_state[0])

        # for node in range(self.num_nodes):
        #     self.p_state[node] = np.random.choice([1, 4], p=pT[node][self.p_state[node] - 1])

        self.p_state = np.random.choice([4, 4])
        # c_s = np.hstack((self.c_state[:self.num_nodes]))
        # return observation (states)
        s = np.hstack((self.p_state, self.q_state))
        total_work = (self.q_state)
        # print("p", self.p_state)
        # print("q", self.q_state)
        # print("c", self.c_state)
        return s, total_work

    def step(self, shared_action):
        task_arrival_rate = self.task_arrival_rate
        new_task = np.random.poisson(task_arrival_rate)  # poisson distribution

        # self.p_a += sum(self.p_state)
        q_delay = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        local_jobs = 0
        for k, v in shared_action.items():
            local_jobs += v[self.node_num]
        # print("local", local_jobs)
        self.q_state = local_jobs - self.p_state + new_task
        self.q_state = self.q_state if self.q_state > 0 else 0
        d_delay = self.q_state - self.Q_SIZE if self.q_state > self.Q_SIZE else 0
        self.q_state = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        reward = float(q_delay + self.weight_d * d_delay)

        self.p_state = np.random.choice([4, 4])
        # self.p_state = np.random.choice([4, 4], p=pT[node][self.p_state[node] - 1])

        s_ = np.hstack((self.p_state, self.q_state))
        total_work_ = self.q_state

        # if self.Q_SIZE-self.q_state==0:
        #     print("p_", self.p_state)
        #     print("q_", self.q_state)
        #     print("new jobs", new_task)
        #     print(shared_action)
        #     exit()

        avg_delay = (1 / (self.Q_SIZE - self.q_state)) if self.Q_SIZE - self.q_state != 0 else 15

        return s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
