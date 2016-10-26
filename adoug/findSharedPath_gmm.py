# -*- coding: utf-8 -*-

'''find shared path with LCS'''

import logging
import json
import datetime

import adoug.config as config

logging.basicConfig(level=logging.WARNING and logging.INFO,
                format='%(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='../log/error.log',
                filemode='a')


def find(sharedPathFile, clusterNodes, measureNodes):
    sharedPath_cluster = {}
    with open(sharedPathFile, 'r') as f:
        sharedPath = json.load(f)
    for i in range(len(clusterNodes)):
        for j in range(i + 1, len(clusterNodes)):
            for measure in measureNodes:
                source = clusterNodes[i]
                dest = clusterNodes[j]
                # if source not in sharedPath_cluster:
                #     sharedPath_cluster[source] = {}
                # elif dest not in sharedPath_cluster[source]:
                #     sharedPath_cluster[source][dest] = {}
                # elif measure not in sharedPath_cluster[source][dest]:
                #     if source in sharedPath:
                #         sharedPath_cluster[source][dest][measure] = sharedPath[source][dest][measure]
                #     elif dest in sharedPath:
                #         sharedPath_cluster[source][dest][measure] = sharedPath[dest][source][measure]
                #     else:
                #         logging.warning('####error: sharedPath_cluster not found: %s and %s to %s.####' % (source, dest, measure))
                if source not in sharedPath_cluster:
                    sharedPath_cluster[source] = {}
                    sharedPath_cluster[source][dest] = {}
                elif dest not in sharedPath_cluster[source]:
                    sharedPath_cluster[source][dest] = {}

                try:
                    sharedPath_cluster[source][dest][measure] = sharedPath[source][dest][measure]
                except KeyError:
                    sharedPath_cluster[source][dest][measure] = sharedPath[dest][source][measure]

    return sharedPath_cluster

now = datetime.datetime.now()
timeStyle = now.strftime("%Y-%m-%d %H:%M:%S")
logging.info('****%s findSharedPath program start****' % timeStyle)

leafNodes = config.config()['leafNodes']
clusterNodes = ['780', '262', '480', '368', '348', '152', '158']
measureNodes = ['254', '51', '192', '889', '752', '812', '67', '320']
sharedPath_cluster = find('../cache/sharedPath.json', clusterNodes, measureNodes)
with open('../cache/sharedPath_gmm.json', 'w') as f:
    json.dump(sharedPath_cluster, f)
