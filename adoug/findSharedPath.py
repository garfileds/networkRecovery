# -*- coding: utf-8 -*-

'''find shared path with LCS'''

import logging
import json

import adoug.lcs as lcs

# logging.basicConfig(level=logging.WARNING and logging.INFO,
#                 format='%(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='./log/info5.log',
#                 filemode='a')
# logHandler.RotatingFileHandler('./log/info5.log', 'a', 20480, 510)

logger = logging.getLogger('info')


def byNode(path, leafNodes, measureNodes):
    sharedPath = {}
    for i in range(len(leafNodes)):
        for j in range(i + 1, len(leafNodes)):
            for measure in measureNodes:
                source = leafNodes[i]
                dest = leafNodes[j]

                if source not in sharedPath:
                    sharedPath[source] = {}
                    sharedPath[source][dest] = {}
                elif dest not in sharedPath[source]:
                    sharedPath[source][dest] = {}

                sourcePath = path[source][measure]
                destPath = path[dest][measure]
                sharedPath[source][dest][measure], commonPath = lcs.lcs_dp(sourcePath, destPath)

    return sharedPath


def byCluster(sharedPathFile, clusterNodes, measureNodes):
    sharedPath = {}
    with open(sharedPathFile, 'r') as f:
        sharedPath = json.load(f)
    for i in range(len(clusterNodes)):
        for j in range(i + 1, len(clusterNodes)):
            source = clusterNodes[i]
            dest = clusterNodes[j]
            for measure in measureNodes:
                if source not in sharedPath:
                    sharedPath[source] = {}
                    sharedPath[source][dest] = {}
                elif dest not in sharedPath[source]:
                    sharedPath[source][dest] = {}

                try:
                    sharedPath[source][dest][measure] = sharedPath[source][dest][measure]
                except KeyError:
                    sharedPath[source][dest][measure] = sharedPath[dest][source][measure]

    return sharedPath
