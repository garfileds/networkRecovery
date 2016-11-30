# /usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import math
import re
import copy
import json
import random
import datetime
import logging
import logging.handlers as logHandler

import adoug.network_generator as ng
import adoug.config_raw as config_raw
import adoug.mapColor as mc
import adoug.config as config
import adoug.findSharedPath as fsp
import adoug.estimation as estimation
import adoug.clustering as clu
import adoug.predict as predict
import adoug.sharedPathBaseOnHop as shared_path_finder

LOG_FILE = './log/info_recovery.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024*1024, backupCount=1000)
fmt = '%(message)s'

formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger = logging.getLogger('info')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def find_measure_nodes(num, data):
    data_sort = sorted(data, reverse=True)
    measure_nodes = []

    for item in data_sort:
        for index in range(100):
            if item == data[index]:
                measure_nodes.append(index)

    logger.info("appear_count_sort_real is: \n%s\n" % (str(measure_nodes)))
    return measure_nodes[:num]


def which_net_type(ip):
    net_type = ''
    match = re.match(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', ip)
    first, sencond, third, fourth = match.group(1, 2, 3, 4)
    if first == '10' or first == '172' and int(sencond) >= 16 and int(sencond) <= 31 or first == '192' and sencond == '168':
        net_type = 'intranet'
    elif int(first) <= 127:
        net_type = 'A'
    elif int(first) >= 128 and int(first) <= 191:
        net_type = 'B'
    elif int(first) >= 192 and int(first) <= 223:
        net_type = 'C'
    elif int(first) >= 224 and int(first) <= 239:
        net_type = 'D'
    elif int(first) >= 240 and int(first) <= 255:
        net_type = 'E'

    return net_type


# 配置
Epsilon = 1

NUM_LeaNode = 1000
NUM_MeasureNode = 15
n_components_gmm = 10
# 配置 end

now = datetime.datetime.now()
timeStyle = now.strftime("%Y-%m-%d %H:%M:%S")
logger.info('****%s: recoveryTopo_direct.py start****\n' % (timeStyle))

config_raw = config_raw.config_raw()
nodelist_ip = list(config_raw['node_serial_map'].keys())
nodelist = list(config_raw['node_serial_map'].values())
node_serial_map = config_raw['node_serial_map']
hop_dict = config_raw['hopDict']
appear_count = config_raw['appearCount']
appear_countlist = config_raw['appearCountList']
print('nodelist: %d\n' % len(nodelist))

node_serial_map_reverse = {}
for (ip, number) in node_serial_map.items():
    node_serial_map_reverse[number] = ip

# appear_count_sort = sorted(appear_countlist, reverse=True)
# logger.info('appear_count_sort is: \n%s\n' % str(appear_count_sort))

measure_nodes = config_raw['appearCountSortReal'][:NUM_MeasureNode]

# 假设相关变量
leaf_nodes = copy.deepcopy(nodelist)
for item in measure_nodes:
    leaf_nodes.remove(item)
leaf_nodes = leaf_nodes[:NUM_LeaNode]

map_leaf_to_i = {}
for i in range(len(leaf_nodes)):
    map_leaf_to_i[leaf_nodes[i]] = i

hoplist = [[0 for i in range(len(measure_nodes))] for j in range(len(leaf_nodes))]
path = {}
count_miss_hop = 0
for j in range(len(measure_nodes)):
    dest = measure_nodes[j]
    for i in range(len(leaf_nodes)):
        source = leaf_nodes[i]
        hop = 10000
        if dest in hop_dict and source in hop_dict[dest]:
            hop = hop_dict[dest][source]
        elif source in hop_dict and dest in hop_dict[source]:
            hop = hop_dict[source][dest]

        if hop == 10000:
            count_miss_hop += 1

        hoplist[i][j] = hop

        if source not in path:
            path[source] = {}
            path[source][dest] = {}
        elif dest not in path[source]:
            path[source][dest] = {}
        path[source][dest] = hop
print('count_miss_hop is: %d\n' % count_miss_hop)
print('init 1: complete measure_nodes, leaf_nodes, path, hoplist\n')

hop_sum_list = [0 for i in range(NUM_MeasureNode)]
hop_miss_count = [0 for i in range(NUM_MeasureNode)]
hop_average_list = [0 for i in range(NUM_MeasureNode)]

# caculate hopCountAverage without miss hop
for i in range(NUM_MeasureNode):
    for j in range(NUM_LeaNode):
        if hoplist[j][i] != 10000:
            hop_sum_list[i] += hoplist[j][i]
            hop_miss_count[i] += 1

for i in range(len(hop_sum_list)):
    hop_average_list[i] = round(hop_sum_list[i] / (NUM_LeaNode - hop_miss_count[i]))

# replace miss hop with average of other hops
hoplist_replace = hoplist[:len(hoplist)]
hop_miss_index = []

for i in range(len(hoplist_replace)):
    for j in range(NUM_MeasureNode):
        if hoplist_replace[i][j] == 10000:
            hoplist_replace[i][j] = hop_average_list[j]
            hop_miss_index.append((i, j))

# caculate hopCountAverage after replacing miss hop
hop_sum_list_replace = [0 for i in range(NUM_MeasureNode)]
hop_average_list_replace = [0 for i in range(NUM_MeasureNode)]

for i in range(len(hoplist_replace)):
    for j in range(NUM_MeasureNode):
        hop_sum_list_replace[j] += hoplist_replace[i][j]

for i in range(NUM_MeasureNode):
    hop_average_list_replace[i] = round(hop_sum_list_replace[i] / NUM_LeaNode)

# hop-count contrast
hop_contrast = hoplist[:len(hoplist)]

for i in range(NUM_LeaNode):
    for j in range(NUM_MeasureNode):
        hop_contrast[i][j] = round(abs(hoplist_replace[i][j] - hop_average_list_replace[j]))

train_data = np.array(hop_contrast)
dpgmm = mixture.GaussianMixture(n_components_gmm,
                                        covariance_type='full').fit(train_data, miss_index_hop=hop_miss_index, hoplist_contrast_mean=hop_contrast)
logger.info('hoplist_contrast_mean after gmm.fit is:\n%s\n' % str(hop_contrast))

means = dpgmm.means_
logger.info('dpgmm\'s means is: %s\n' % (means))
print('dpgmm\'s means is: %s\n' % (means))

# predict label
predict_result = dpgmm.predict(train_data)
predict_result_list = predict_result.tolist()
logger.info('result of predicting leafnode: \n' + str(predict_result_list))

# replacing miss hop with gmm.mean_
hoplist_gmm = hoplist[:len(hoplist)]

for index in hop_miss_index:
    leaf = index[0]
    measure = index[1]
    hoplist_gmm[leaf][measure] = hop_average_list_replace[measure] + means[predict_result_list[leaf]][measure]

    path[leaf_nodes[leaf]][measure_nodes[measure]] = int(round(hoplist_gmm[leaf][measure]))

logger.info('hopList recovering by gmm is:\n%s\n' % (str(hoplist_gmm)))
print('5:predictResult\n')

# shared path estimation
likehood = predict.caculateLikehoodMap(hoplist_gmm, leaf_nodes, measure_nodes, Epsilon)
logger.info('likehood: \n%s\n' % (str(likehood)))
print('6:likehood\n')

# 假设某些共享路径已知
shared_path_known = {}
likehood_map_kown = {}
likehood_map_measure_c = likehood['map_measure']
likehood_map_leaf_leaf = likehood['map']

for k in range(len(measure_nodes)):
    measure = measure_nodes[k]

    for likehood in range(1, NUM_MeasureNode + 1):
        if likehood not in likehood_map_measure_c[measure]:
            pair_select = []
        elif len(likehood_map_measure_c[measure][likehood]) < 2:
            pair_select = likehood_map_measure_c[measure][likehood][:1]
        else:
            pair_select = likehood_map_measure_c[measure][likehood][:2]

        if measure not in likehood_map_kown:
            likehood_map_kown[measure] = {}
            likehood_map_kown[measure][likehood] = []
        elif likehood not in likehood_map_kown[measure]:
            likehood_map_kown[measure][likehood] = []

        for pair in pair_select:
            likehood_map_kown[measure][likehood].append(pair)

            source = pair[0]
            dest = pair[1]

            alpha = random.random()

            if source not in shared_path_known:
                shared_path_known[source] = {}
                shared_path_known[source][dest] = {}
            elif dest not in shared_path_known[source]:
                shared_path_known[source][dest] = {}

            shared_path_known[source][dest][measure] = int(alpha * min(hoplist_gmm[map_leaf_to_i[source]][k], hoplist_gmm[map_leaf_to_i[dest]][k]))

alpha_map_c_measure = predict.getAlpha(likehood_map_kown, shared_path_known, path, measure_nodes)
print('7:alphaMap\n')

shared_path_predict = shared_path_finder.RecoveryTopo().find_shared_path_based_on_hop(likehood_map_leaf_leaf, alpha_map_c_measure, path, leaf_nodes, measure_nodes)

network_gmm, label_figid_to_node_gmm = ng.NetworkGenerator().generate(shared_path_predict, hoplist_gmm, leaf_nodes, measure_nodes[:1])

map_node_to_color = {}
colordict = {
    'intranet': 'red',
    'A': 'blue',
    'B': 'yellow',
    'C': 'white',
    'D': 'c',
    'E': 'green'
}

for node in nodelist:
    ip = node_serial_map_reverse[node]
    ip_type = which_net_type(ip)
    map_node_to_color[node] = colordict[ip_type]

map_node_to_figid_gmm = {}
for (figid, node) in label_figid_to_node_gmm.items():
    map_node_to_figid_gmm[node] = figid

figid_list_known_gmm = list(label_figid_to_node_gmm.keys())
colormap_known_gmm = []
for figid in figid_list_known_gmm:
    node = label_figid_to_node_gmm[figid]

    colormap_known_gmm.append(map_node_to_color[node])
colormap_known_gmm[0] = '#a51391'

print('nodes\' number: %d\n' % len(network_gmm.nodes()))
print('edges\' number: %d\n' % len(network_gmm.edges()))

# 度分布
print('度分布\n')
degree_gmm = nx.degree_histogram(network_gmm)
logger.info('degree is:\n%s\n' % str(degree_gmm))
x = []
for i in range(len(degree_gmm)):
    for j in range(degree_gmm[i]):
        x.append(i)
n, bins, patches = plt.hist(x, 5, normed=1, facecolor='green', alpha=0.5)
plt.show()
plt.cla()

# 图的直径
diameter_gmm = nx.diameter(network_gmm)
print('diameter_gmm is: %s\n' % (str(diameter_gmm)))
logger.info('diameter_gmm is: %s\n' % (str(diameter_gmm)))

print('已知节点的拓扑图\n')
nx.draw(network_gmm, nodelist=figid_list_known_gmm, node_color=colormap_known_gmm, node_size=30, labels=label_figid_to_node_gmm, font_size=10)
plt.axis('off')
plt.show()
plt.cla()

print('所有节点的拓扑图\n')
nx.draw(network_gmm, node_size=30, labels=label_figid_to_node_gmm, font_size=10)
plt.axis('off')
plt.show()
plt.cla()
