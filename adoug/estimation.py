# -*- coding: utf-8 -*-

'''
   fileOverview: shared path estimation for two kinds of cluster method
   method1: unique hop-contrast as cluster
   method2: GMM
'''

import json
import math
import copy
import datetime
import logging

import adoug.config as config

# logging.basicConfig(level=logging.WARNING and logging.INFO,
#                 format='%(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='./log/info5.log',
#                 filemode='a')
# logHandler.RotatingFileHandler('./log/info5.log', 'a', 20480, 510)

logger = logging.getLogger('info')

# nodeLevel = {'test': {
#             'file': './topo/2K_1000TopoGen.txt',
#             'path': './cache/path_test.json',
#             'sharedPath': './cache/sharedPath_test.json',
#             'sharedPath_gmm': './cache/sharedPath_gmm_test.json',
#             'sharedPath_hop': './cache/sharedPath_hop_test.json',
#             'numOfLeafnode': 10,
#             'numOfMeasurenode': 2,
#             'n_components': 2,
#             'C_likehood': 1,
#             'likehood': './cache/likehood_C' + str(1) + '_E' + str(1) + '_2K_1000.json'
#         },
#         'param': {
#             'Epsilon': 1
#         }
#     }


def getSharedPath(method, level):
    nodeLevel = config.configNodeLevel()
    levelDict = {'node': 'sharedPath', 'gmm': 'sharedPath_gmm', 'hop': 'sharedPath_hop'}
    with open(nodeLevel[level][levelDict[method]], 'r') as f:
        sharedPath = json.load(f)

    return sharedPath


def estimate(leafNodes, measureNodes, method, labelMap, nodeMap, level):
    interSum = 0
    RMSE = 0

    sharedPath_node = getSharedPath('node', level)

    now = datetime.datetime.now()
    nowFormat = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info('****estimate shared path with level of %s start.%s****' % (level, nowFormat))

    for i in range(len(leafNodes)):
        for j in range(i + 1, len(leafNodes)):
            source = leafNodes[i]
            dest = leafNodes[j]

            sourceMap = nodeMap[labelMap[source]]
            destMap = nodeMap[labelMap[dest]]

            if sourceMap != destMap:
                for measure in measureNodes:
                    try:
                        myShared = sharedPath_node[sourceMap][destMap][measure]
                    except KeyError:
                        myShared = sharedPath_node[nodeMap[labelMap[dest]]][nodeMap[labelMap[source]]][measure]
                    realShared = sharedPath_node[source][dest][measure]
                    interSum += math.fabs((myShared - realShared) ** 2)
    RMSE = (interSum / (len(leafNodes) * (len(leafNodes) + 1) / 2 - 2)) ** 0.5
    logger.info('%s-cluster\'s RMSE is: %s.\nThere are %s clusters.' % (level, str(RMSE), str(len(nodeMap))))

    now = datetime.datetime.now()
    nowFormat = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info('****estimate shared path with level of %s end.%s****' % (level, nowFormat))

    return RMSE


def estimateByPredict(leafNodes, measureNodes, likehood_map, alphaMap, path, level):
    sharedPath_node = getSharedPath('node', level)
    shared_path_predict = copy.deepcopy(sharedPath_node)

    interSum = 0

    for i in range(len(leafNodes)):
        for j in range(i + 1, len(leafNodes)):
            source = leafNodes[i]
            dest = leafNodes[j]

            if source != dest:
                for measure in measureNodes:
                    try:
                        likehood = likehood_map[source][dest][measure]
                    except KeyError:
                        likehood = likehood_map[dest][source][measure]

                    myShared = alphaMap[likehood][measure] * min(len(path[source][measure]), len(path[dest][measure]))
                    shared_path_predict[source][dest][measure] = myShared
                    realShared = sharedPath_node[source][dest][measure]
                    interSum += (myShared - realShared) ** 2
    RMSE = (interSum / (len(leafNodes) * (len(leafNodes) + 1) / 2 * len(measureNodes))) ** 0.5
    # RMSE = interSum ** 0.5

    return RMSE, shared_path_predict
