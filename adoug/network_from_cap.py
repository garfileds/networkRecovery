# /usr/bin/usr
# -*- coding: utf-8 -*-

from scapy.all import *
import os


def _walkDir(rootDir):
    filelist = []
    for file in os.listdir(rootDir):
        path = os.path.join(rootDir, file)

        if not os.path.isdir(path):
            filelist.append(path)
        else:
            walkDir(path)

    return filelist


def _findHop(hop, hopDict):
    hopFilter = {}
    for src in hopDict:
        for dst in hopDict[src]:
            if hopDict[src][dst] == hop:
                if src not in hopFilter:
                    hopFilter[src] = {}
                hopFilter[src][dst] = hop

    return hopFilter


def process(pcapDir):
    serial_number = 0
    node_serial_map = {}
    hopDict = {}
    appearCount = {}
    appearCountList = [0 for i in range(10000)]

    pcaplist = _walkDir('pcapDir')
    for pcap in pcaplist:
        packages = rdpcap(pcap)
        for package in packages:
            source = package.src
            dest = package.dst
            ttl = package.ttl

            if not ttl:
                hop = 10000
            elif ttl <= 128:
                hop = 128 - ttl
            else:
                hop = 255 - ttl

            if source not in node_serial_map:
                node_serial_map[source] = serial_number
                serial_number += 1

                appearCount[source] = 1
            else:
                appearCount[source] += 1

            if dest not in node_serial_map:
                node_serial_map[dest] = serial_number
                serial_number += 1

                appearCount[dest] = 1
            else:
                appearCount[dest] += 1

            appearCountList[node_serial_map[source]] += 1
            appearCountList[node_serial_map[dest]] += 1

            source_serial = node_serial_map[source]
            dest_serial = node_serial_map[dest]

            if source not in hopDict:
                hopDict[source_serial] = {}
            hopDict[source_serial][dest_serial] = hop

    nodelist = node_serial_map.values()
    nodelist_ip = node_serial_map.keys()

    return {
        nodelist: nodelist,
        nodelist_ip: nodelist_ip,
        hop_dict: hopDict,
        appear_count: appearCount,
        appear_countlist: appearCountList
    }
