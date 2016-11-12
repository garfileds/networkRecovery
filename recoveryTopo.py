#! /usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')

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

import adoug.mapColor as mc
import adoug.config as config
import adoug.findSharedPath as fsp
import adoug.estimation as estimation
import adoug.clustering as clu
import adoug.predict as predict
import adoug.network_generator as ng

nodeLevel = config.configNodeLevel()
MissPercent = 0.9

# logging.basicConfig(level=logging.WARNING and logging.INFO,
#                 format='%(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='./log/info3.log',
#                 filemode='a')
# logHandler.RotatingFileHandler('./log/info3.log', 'a', 1024 * 1024 * 20, 510)

LOG_FILE = './log/info2.log'

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
    NUM_MeasureNode = nodeLevel[level]['numOfMeasurenode']
    # NUM_MeasureNode = 13
    Epsilon = nodeLevel['param']['Epsilon']

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

    if level == '':
        leafNodes = leaf[:NUM_LeaNode]
        measureNodes = leaf[NUM_LeaNode:NUM_LeaNode+NUM_MeasureNode]
    else:
        leafNodes = config.config()[level]['leafnodes'][:NUM_LeaNode]
        measureNodes = config.config()[level]['measurenodes'][:NUM_MeasureNode]
    logger.info('leafNodes:\n' + str(leafNodes))
    logger.info('measureNodes:\n' + str(measureNodes))
    print('1:nodes add\n')

    # shortest distance as hop between nodes
    # if level == '':
    #    path = nx.all_pairs_shortest_path(G)
        # with open(nodeLevel[level]['path'], 'w') as f:
        #    json.dump(path, f)
    # else:
    #    with open(nodeLevel[level]['path_extract'], 'r') as f:
    #        path = json.load(f)
    if level == '2K_1000':
        path_extract = {}
        for leaf in leafNodes:
            if leaf not in path_extract:
                path_extract[leaf] = {}
            for measure in measureNodes:
                path_extract[leaf][measure] = nx.shortest_path(G, source=leaf, target=measure)
    path = path_extract
    print('2:path\n')

    # caculate sharedPath between nodes
    if level == '2K_1000':
        sharedPath = fsp.byNode(path, leafNodes, measureNodes)
        # with open(nodeLevel[level]['sharedPath'], 'w') as f:
        #    json.dump(sharedPath, f)
    else:
        with open(nodeLevel[level]['sharedPath'], 'r') as f:
            sharedPath = json.load(f)
    print('3:sharedPath\n')

    if level == '2K_1000':
        hopSet = {}
        hopList = []
        hopCountSum = [0 for i in range(NUM_MeasureNode)]

        for leafNode in leafNodes:
            hopSet[leafNode] = []
            for i in range(NUM_MeasureNode):
                hopSet[leafNode].append(len(path[leafNode][measureNodes[i]]) - 1)
                hopCountSum[i] += len(path[leafNode][measureNodes[i]]) - 1
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
    print('4:param\n')

    # Gaussian mixture model(GMM) for clutering leaf nodes
    n_components_gmm = nodeLevel[level]['n_components']
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
    print('5:predictResult\n')

    # shared path estimation

    # estimation with predict using gmm-data: get alpha
    # hoplist_gmm = hoplist_mean
    if level == '2K_1000':
        likehood = predict.caculateLikehoodMap(hoplist_gmm, leafNodes, measureNodes, Epsilon)
        # with open(nodeLevel[level]['likehood'], 'w') as f:
        #     json.dump(likehood, f)
    else:
        with open(nodeLevel[level]['likehood'], 'r') as f:
            likehood = json.load(f)
    print('6:likehood\n')
    alphaMap = predict.getAlpha(likehood['map_measure'], sharedPath, path, measureNodes)
    print('7:alphaMap\n')


    # estimation with predict using gmm-data: get alpha
    if level == '2K_1000':
        estimation_predict, shared_path_predict = estimation.estimateByPredict(leafNodes, measureNodes[:1], likehood['map'], alphaMap, path, level)
        shared_path_predict['RMSE'] = estimation_predict
        # with open(nodeLevel[level]['shared_path_predict'], 'w') as f:
        #     json.dump(shared_path_predict, f)
    else:
        with open(nodeLevel[level]['shared_path_predict'], 'r') as f:
            shared_path_predict = json.load(f)
            estimation_predict = shared_path_predict['RMSE']
    logger.info('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), str(MissPercent)))
    print('8:shared_path_predict\n')
    print('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), str(MissPercent)))

    network_gmm, label_ip = ng.NetworkGenerator().generate(shared_path_predict, hoplist_gmm, leafNodes, measureNodes[:1])
    print('edges numbers: %s\n' % (len(network_gmm.edges())))
    node_color = ['green' for i in range(5)] + ['red' for i in range(len(network_gmm.nodes()) - 5)]
    nx.draw(network_gmm, node_size=30, node_color=node_color, labels=label_ip)
    #
    plt.axis('off')
    plt.savefig("topo.png", format="PNG")
    plt.show()
