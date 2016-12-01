#! /usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import math
import copy
import logging
import logging.handlers as logHandler
import datetime
import json
import random
import sys

import adoug.config as config
import adoug.findSharedPath as fsp
import adoug.estimation as estimation
import adoug.predict as predict
import adoug.network_generator as ng

nodeLevel = config.configNodeLevel()
# MissPercent = float(sys.argv[3])
MissPercent = 0.3

# logging.basicConfig(level=logging.WARNING and logging.INFO,
#                 format='%(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='./log/info3.log',
#                 filemode='a')
# logHandler.RotatingFileHandler('./log/info3.log', 'a', 1024 * 1024 * 20, 510)

LOG_FILE = './log/info4.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024*1024*20, backupCount=510)
fmt = '%(message)s'

formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger = logging.getLogger('info')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def replaceMissHop(hopVector):
    for i in range(len(hopVector)):
        if hopVector[i] == 10000:
            hopVector[i] = hop_average_raw[i]
    return hopVector

now = datetime.datetime.now()
timeStyle = now.strftime("%Y-%m-%d %H:%M:%S")
logger.info('****%s: drawTopo program start****\n' % (timeStyle))

for level in nodeLevel:
    if level != '2K_1000':
        continue

    now = datetime.datetime.now()
    timeStyle = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info('****%s, %s: start****\n' % (timeStyle, level))
    logger.info('****Miss percent is: %s\n****' % (str(MissPercent)))

    NUM_LeaNode = nodeLevel[level]['numOfLeafnode']
    # NUM_MeasureNode = nodeLevel[level]['numOfMeasurenode']
    NUM_MeasureNode = 15
    # NUM_LeaNode = int(sys.argv[1])
    # NUM_MeasureNode = int(sys.argv[2])
    Epsilon = nodeLevel['param']['Epsilon']

    print('跳数缺失率：%s\n' % str(MissPercent))
    print('叶子节点数目：%s\n' % str(NUM_LeaNode))
    print('观测节点数目：%s\n' % str(NUM_MeasureNode))

    levelConfig = config.config()[level]

    f = open(nodeLevel[level]['file'], 'r')
    G = nx.Graph()
    for linefile in f:
        list = linefile.split(' ')
        x = list[0]
        list[1] = list[1].replace('\n', '')
        y = list[1]
        G.add_edge(x, y)

    # select leaf nodes & 8measurement nodes
    nodes = G.nodes()
    degree = G.degree()

    leaf = []
    for node in degree.keys():
        if degree[node] == 1:
            leaf.append(node)
    print('仿真网络的已知节点数目: %s\n' % (len(leaf)))
    print('仿真网络的已知边数: %s\n' % (len(G.edges())))
    nx.draw(G, node_size=30)
    plt.axis('off')
    plt.show()
    plt.cla()
    logger.info('nodes of real graph has: %s\n' % (len(leaf)))
    logger.info('edges of real graph has: %s\n' % (len(G.edges())))

    if level == '':
        leaf_nodes = leaf[0:-NUM_MeasureNode]
        measure_nodes = leaf[-NUM_MeasureNode:len(leaf)]
    else:
        leaf_nodes = config.config()[level]['leaf_nodes'][:NUM_LeaNode]
        measure_nodes = config.config()[level]['measure_nodes'][:NUM_MeasureNode]
    logger.info('leaf_nodes:\n' + str(leaf_nodes))
    logger.info('measure_nodes:\n' + str(measure_nodes))
    print('Step 1: 选择观测节点、叶子节点\n')

    # shortest distance as hop between nodes
    # if level == '2K_1000':
    #     path = nx.all_pairs_shortest_path(G)
        # with open(nodeLevel[level]['path'], 'w') as f:
        #    json.dump(path, f)
    # else:
    #     with open(nodeLevel[level]['path_extract'], 'r') as f:
    #         path = json.load(f)
    if level == '2K_1000':
        path_extract = {}
        for leaf in leaf_nodes:
            if leaf not in path_extract:
                path_extract[leaf] = {}
            for measure in measure_nodes:
                path_extract[leaf][measure] = nx.shortest_path(G, source=leaf, target=measure)
    path = path_extract
    print('Step 2:计算参数path\n')

    # caculate shared_path between nodes
    if level == '2K_1000':
        shared_path = fsp.byNode(path, leaf_nodes, measure_nodes)
        # with open(nodeLevel[level]['shared_path'], 'w') as f:
        #     json.dump(shared_path, f)
    else:
        with open(nodeLevel[level]['shared_path'], 'r') as f:
            shared_path = json.load(f)
    print('Step 3:计算参数shared_path\n')

    if level == '2K_1000':
        hopSet = {}
        hopList = []
        hopCountSum = [0 for i in range(NUM_MeasureNode)]

        for leafNode in leaf_nodes:
            hopSet[leafNode] = []
            for i in range(NUM_MeasureNode):
                hopSet[leafNode].append(len(path[leafNode][measure_nodes[i]]) - 1)
                hopCountSum[i] += len(path[leafNode][measure_nodes[i]]) - 1
            hopList.append(hopSet[leafNode])

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
    else:
        hop_average_mean = levelConfig['hop_average_mean']
        hoplist_raw = levelConfig['hoplist_raw']
        hoplist_contrast_mean = levelConfig['hoplist_contrast_mean']
    print('Step 4:计算跳数矩阵和跳数差值矩阵\n')

    # Gaussian mixture model(GMM) for clutering leaf nodes
    n_components_gmm = nodeLevel[level]['n_components']
    print('n_components_gmm is : %d\n' % n_components_gmm)
    if level == '2K_1000':
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
    else:
        predictResultList = levelConfig['predict_result']
        hoplist_gmm = levelConfig['hoplist_gmm']
    print('Step 5:计算参数predictResult\n')

    # shared path estimation

    # estimation with predict using gmm-data: get alpha
    # hoplist_gmm = hoplist_mean
    if level == '2K_1000':
        likehood = predict.caculateLikehoodMap(hoplist_gmm, leaf_nodes, measure_nodes, Epsilon)
        # with open(nodeLevel[level]['likehood'], 'w') as f:
        #     json.dump(likehood, f)
    else:
        with open(nodeLevel[level]['likehood'], 'r') as f:
            likehood = json.load(f)
    print('Step 6:计算参数likehood\n')
    alphaMap = predict.getAlpha(likehood['map_measure'], shared_path, path, measure_nodes)
    print('Step 7:计算alphaMap\n')

    # estimation with predict using gmm-data: get alpha
    if level == '2K_1000':
        estimation_predict, estimation_predict_percent, shared_path_predict = estimation.estimateByPredict(leaf_nodes, measure_nodes[:1], likehood['map'], alphaMap, path, shared_path)
        shared_path_predict['RMSE'] = estimation_predict
        # with open(nodeLevel[level]['shared_path_predict'], 'w') as f:
        #     json.dump(shared_path_predict, f)
    else:
        with open(nodeLevel[level]['shared_path_predict'], 'r') as f:
            shared_path_predict = json.load(f)
            estimation_predict = shared_path_predict['RMSE']
    logger.info('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), '1'))
    logger.info('estimation with approach of predict\'s result is(percent): %s when Epsilon is %s.' % (str(1 - estimation_predict_percent), '1'))
    print('Step 8:计算参数shared_path_predict\n')
    print('网络拓扑推断误差率(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), '1'))
    print('网络拓扑推断准确度: %s when Epsilon is %s.' % (str(1 - estimation_predict_percent), '1'))

    network_gmm, label_ip = ng.NetworkGenerator().generate(shared_path_predict, hoplist_gmm, leaf_nodes, measure_nodes[:1])
    pos = nx.fruchterman_reingold_layout(network_gmm)
    nx.draw(network_gmm, pos=pos, node_size=30, labels=label_ip)
    plt.axis('off')
    plt.show()

    # 度分布
    # degree_gmm = nx.degree_histogram(network_gmm)
    # x = []
    # for i in range(len(degree_gmm)):
    #     for j in range(degree_gmm[i]):
    #         x.append(i)
    # n, bins, patches = plt.hist(x, 5, normed=1, facecolor='green', alpha=0.5)
    # plt.show()
    # plt.cla()

    # 图的直径
    diameter_gmm = nx.diameter(network_gmm)
    print('拓扑推断图的直径是: %s\n' % (str(diameter_gmm)))
    logger.info('diameter_gmm is: %s\n' % (str(diameter_gmm)))

    # 最大连通子图
    # ccs = [len(c) for c in sorted(nx.connected_components(network_gmm), key=len, reverse=True)]
    # print(ccs)
    # print('\n')
    # largest_cc = max(nx.connected_components(network_gmm), key=len)
    #
    # node_color = ['green'] + ['red' for i in range(len(network_gmm.nodes()) - 1)]
    # for index in largest_cc:
    #     node_color[int(index)] = 'c'
    # nx.draw(network_gmm, node_size=30, node_color=node_color, labels=label_ip)
    #
    # plt.axis('off')
    # plt.show()
