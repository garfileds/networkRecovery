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

LOG_FILE = './log/info_raw.log'

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


def replaceMissHop(hopVector):
    for i in range(len(hopVector)):
        if hopVector[i] == 10000:
            hopVector[i] = hop_average_raw[i]
    return hopVector


now = datetime.datetime.now()
timeStyle = now.strftime("%Y-%m-%d %H:%M:%S")
logger.info('****%s: network_raw.py start****\n' % (timeStyle))

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

measure_nodes = config_raw['appearCountSortReal'][:8]

# 假设相关变量
leaf_nodes = copy.deepcopy(nodelist)
for item in measure_nodes:
    leaf_nodes.remove(item)

hoplist = [[0 for i in range(len(measure_nodes))] for j in range(len(leaf_nodes))]
path = {}
for j in range(len(measure_nodes)):
    dest = measure_nodes[j]
    for i in range(len(leaf_nodes)):
        source = leaf_nodes[i]
        hop = 0
        if dest in hop_dict and source in hop_dict[dest]:
            hop = hop_dict[dest][source]
        elif source in hop_dict and dest in hop_dict[source]:
            hop = hop_dict[source][dest]
        elif hop == 0:
            hop = random.randint(1, 88)
        else:
            hop = random.randint(1, 88)

        hoplist[i][j] = hop

        if source not in path:
            path[source] = {}
            path[source][dest] = {}
        elif dest not in path[source]:
            path[source][dest] = {}
        path[source][dest] = hop
print('1: complete measure_nodes, leaf_nodes, path, hoplist\n')

shared_path = {}
for i in range(len(leaf_nodes) - 1):
    source = leaf_nodes[i]
    for j in range(i + 1, len(leaf_nodes)):
        dest = leaf_nodes[j]
        for k in range(len(measure_nodes)):
            measure = measure_nodes[k]
            alpha = random.random()

            if source not in shared_path:
                shared_path[source] = {}
                shared_path[source][dest] = {}
            elif dest not in shared_path[source]:
                shared_path[source][dest] = {}

            shared_path[source][dest][measure] = int(alpha * min(hoplist[i][k], hoplist[j][k]))

print('2: complete shared_path\n')

# 节点标签和对应的颜色，画图所需
label = {}
label_reverse = {}
colordict = {
    'intranet': 'red',
    'A': 'blue',
    'B': 'yellow',
    'C': 'white',
    'D': 'c',
    'E': 'green'
}

network_raw, label_figid_to_node = ng.NetworkGenerator().generate(shared_path, hoplist, leaf_nodes, measure_nodes[:1])
print('nodes of real graph has: %s\n' % (len(network_raw.nodes())))
print('edges of real graph has: %s\n' % (len(network_raw.edges())))

logger.info('nodes of real graph has: %s\n' % (len(network_raw.nodes())))
logger.info('edges of real graph has: %s\n' % (len(network_raw.edges())))

map_node_to_figid = {}
for (figid, node) in label_figid_to_node.items():
    map_node_to_figid[node] = figid

figid_list_known = list(label_figid_to_node.keys())

figid_list_all = network_raw.nodes()

map_figid_to_color = {}
map_node_to_color = {}
colormap_known = []
colormap_all = []

for figid in figid_list_known:
    node = label_figid_to_node[figid]

    ip_type = which_net_type(node_serial_map_reverse[node])
    colormap_known.append(colordict[ip_type])
    map_node_to_color[node] = colordict[ip_type]
colormap_known[0] = '#a51391'

for figid in figid_list_all:
    if figid not in label_figid_to_node:
        colormap_all.append('m')
        map_figid_to_color[figid] = 'm'
    else:
        ip_type = which_net_type(node_serial_map_reverse[label_figid_to_node[figid]])
        colormap_all.append(colordict[ip_type])
        map_figid_to_color[figid] = colordict[ip_type]
map_figid_to_color[0] = '#a51391'
colormap_all[0] = '#a51391'

print('3: complete network_raw, colormap_known, colormap_all\n')

# print('nodelist_known: %d\n' % len(nodelist_known))
# print('colormap_known: %d\n' % len(colormap_known))
# print('label_figid_to_node: %d\n' % len(label_figid_to_node))
# nx.draw(network_raw, nodelist=nodelist_known, node_size=30, node_color=colormap_known, labels=label_figid_to_node)
# plt.axis('off')
# plt.show()
# plt.cla()

# nx.draw(network_raw, nodelist=nodelist_all, node_size=30, node_color=colormap_all, labels=label_figid_to_node, font_size=10)
# plt.axis('off')
# plt.show()
# plt.cla()

# 保存图文件
# pos = nx.spring_layout(network_raw)
# nx.set_node_attributes(network_raw, 'pos', pos)
# nx.write_gpickle(network_raw, "./topo/network_raw.gpickle")
# print('4: save ./topo/network_raw.gpickle')
# nx.draw(network_raw, nodelist=nodelist_known, edgelist=[], node_color=colormap_known, labels=label)
# nx.draw(network_raw, nodelist=nodelist_all, node_size=30, node_color=colormap_all, labels=label, font_size=10)


# 配置
nodeLevel = config.configNodeLevel()
MissPercent = 0.3
Epsilon = 1

NUM_LeaNode = 100
NUM_MeasureNode = 3
n_components_gmm = 3
# 配置 end

leafNodes = leaf_nodes[:NUM_LeaNode]
measureNodes = measure_nodes[:NUM_MeasureNode]

figid_list_known_raw = []
colormap_known_raw = []
for node in leafNodes:
    figid = map_node_to_figid[node]
    figid_list_known_raw.append(figid)
    colormap_known_raw.append(map_figid_to_color[figid])
figid_list_known_raw.insert(0, map_node_to_figid[measureNodes[0]])
colormap_known_raw.insert(0, '#a51391')

# nx.draw(network_raw, nodelist=figid_list_known_raw, node_size=30, node_color=colormap_known_raw, labels=label_figid_to_node)
# plt.axis('off')
# plt.show()
# plt.cla()

logger.info('leafNodes:\n' + str(leafNodes))
logger.info('measureNodes:\n' + str(measureNodes))
print('1:nodes add\n')

sharedPath = shared_path
hopSet = {}
hopList = []
hopCountSum = [0 for i in range(NUM_MeasureNode)]

for leafNode in leafNodes:
    hopSet[leafNode] = []
    for i in range(NUM_MeasureNode):
        hopSet[leafNode].append(path[leafNode][measureNodes[i]] - 1)
        hopCountSum[i] += path[leafNode][measureNodes[i]] - 1
    # hopList.append(hopSet[leafNode])

hopList = hoplist[:len(hoplist)]

# random select hop missed
hoplist_raw = copy.deepcopy(hopList)
miss_index_hop = []
for j in range(NUM_MeasureNode):
    randomMiss = random.sample(range(len(hopList)), int(NUM_LeaNode * MissPercent))
    for i in randomMiss:
        hopCountSum[j] = hopCountSum[j] - hopList[i][j]
        hoplist_raw[i][j] = 10000
        miss_index_hop.append([i, j])
logger.info('random incomplete hopList is:\n %s\n' % str(hoplist_raw))
logger.info('random select missed hop is:\n%s\n' % (str(miss_index_hop)))

# caculate hopCountAverage without miss hop
hop_average_raw = [0 for i in range(NUM_MeasureNode)]
for i in range(len(hop_average_raw)):
    hop_average_raw[i] = int(round(hopCountSum[i] / (NUM_LeaNode - NUM_LeaNode * MissPercent)))
logger.info('hopCountAverage without miss hop is: ' + str(hop_average_raw))

# replace miss hop with average of other hops
tempMap = map(replaceMissHop, hoplist_raw)
hoplist_mean = [el for el in tempMap]
logger.info('hopList recovering by mean is:\n%s\n' % str(hoplist_mean))

# caculate hopCountAverage after replacing miss hop
hop_average_mean = copy.deepcopy(hop_average_raw)
for i in range(len(hopCountSum)):
    hop_average_mean[i] = int(round((hopCountSum[i] + (NUM_LeaNode * MissPercent) * hop_average_raw[i]) / NUM_LeaNode))
logger.info('hopAverage after replacing miss hop is: ' + str(hop_average_mean))

# hop-count contrast
hoplist_contrast_mean = copy.deepcopy(hoplist_raw)

for i in range(len(hoplist_raw)):
    for j in range(NUM_MeasureNode):
        hoplist_contrast_mean[i][j] = round(math.fabs(hoplist_mean[i][j] - hop_average_mean[j]))
logger.info('hoplist_contrast_mean is:\n%s\n' % str(hoplist_contrast_mean))
print('4:param\n')

train_data = np.array(hoplist_contrast_mean)
dpgmm = mixture.GaussianMixture(n_components_gmm,
                                        covariance_type='full').fit(train_data, miss_index_hop=miss_index_hop, hoplist_contrast_mean=hoplist_contrast_mean)
logger.info('hoplist_contrast_mean after gmm.fit is:\n%s\n' % str(hoplist_contrast_mean))

means = dpgmm.means_
logger.info('dpgmm\'s means is: %s\n' % (means))
print('dpgmm\'s means is: %s\n' % (means))

# predict label
predictResult = dpgmm.predict(train_data)
predictResultList = predictResult.tolist()
logger.info('result of predicting leafnode: \n' + str(predictResultList))

# replacing miss hop with gmm.mean_
hoplist_gmm = copy.deepcopy(hoplist_raw)

for i in range(NUM_LeaNode):
    for j in range(NUM_MeasureNode):
        if hoplist_raw[i][j] == 10000:
            hoplist_gmm[i][j] = round(hop_average_mean[j] + means[predictResultList[i]][j])
logger.info('hopList recovering by gmm is:\n%s\n' % (str(hoplist_gmm)))
print('5:predictResult\n')

# shared path estimation
likehood = predict.caculateLikehoodMap(hoplist_gmm, leafNodes, measureNodes, Epsilon)
print('6:likehood\n')
alphaMap = predict.getAlpha(likehood['map_measure'], sharedPath, path, measureNodes)
print('7:alphaMap\n')

estimation_predict, estimation_predict_percent, shared_path_predict = estimation.estimateByPredict(leafNodes, measureNodes[:1], likehood['map'], alphaMap, path, sharedPath)
shared_path_predict['RMSE'] = estimation_predict
logger.info('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), '1'))
logger.info('estimation with approach of predict\'s result is(percent): %s when Epsilon is %s.' % (str(1 - estimation_predict_percent), '1'))
print('8:shared_path_predict\n')
# print('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), '1'))
print('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(1 - estimation_predict_percent), '1'))

network_gmm, label_figid_to_node_gmm = ng.NetworkGenerator().generate(shared_path_predict, hoplist_gmm, leafNodes, measureNodes[:1])

map_node_to_figid_gmm = {}
for (figid, node) in label_figid_to_node_gmm.items():
    map_node_to_figid_gmm[node] = figid

figid_list_known_gmm = list(label_figid_to_node_gmm.keys())
colormap_known_gmm = []
for figid in figid_list_known_gmm:
    node = label_figid_to_node_gmm[figid]

    colormap_known_gmm.append(map_node_to_color[node])
colormap_known_gmm[0] = '#a51391'


print('nodelist: %d\n' % len(figid_list_known_gmm))
print('nodecolor: %d\n' % len(colormap_known_gmm))

nx.draw(network_gmm, nodelist=figid_list_known_gmm, node_color=colormap_known_gmm, node_size=30, labels=label_figid_to_node_gmm, font_size=10)
plt.axis('off')
plt.show()
plt.cla()

# 度分布
# degree_gmm = nx.degree_histogram(network_gmm)
# x = []
# for i in range(len(degree_gmm)):
#     for j in range(degree_gmm[i]):
#         x.append(i)
# the histogram of the data
# n, bins, patches = plt.hist(x, 5, normed=1, facecolor='green', alpha=0.5)
# plt.show()
# plt.cla()

# 图的直径
diameter_gmm = nx.diameter(network_gmm)
print('diameter_gmm is: %s\n' % (str(diameter_gmm)))
logger.info('diameter_gmm is: %s\n' % (str(diameter_gmm)))

# 最大连通子图
# ccs = [len(c) for c in sorted(nx.connected_components(network_gmm), key=len, reverse=True)]
# print(ccs)
# print('\n')
# largest_cc = max(nx.connected_components(network_gmm), key=len)

# node_color = ['green'] + ['red' for i in range(len(network_gmm.nodes()) - 1)]
# for index in largest_cc:
#     node_color[int(index)] = 'c'
# nx.draw(network_gmm, node_size=30, node_color=node_color, labels=label_figid_to_node)

# plt.axis('off')
# plt.show()
