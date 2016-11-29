import networkx as nx


class NetworkGenerator():

    '''generate network by shared hop'''

    def __init__(self):
        self.serial_number = 0

    def generate_edges_by_path(self, start_node, end_node, path_length):
        edges = []
        middle_sequence = []

        if start_node >= self.serial_number:
            self.serial_number += 1
        if end_node >= self.serial_number:
            self.serial_number += 1

        if path_length == 1:
            edges.append((start_node, end_node))
        else:
            middle_start = self.serial_number
            middle_end = self.serial_number + (path_length - 2)
            middle_sequence = [i for i in range(middle_start,  middle_end + 1)]

            for i in range(len(middle_sequence) - 1):
                edges.append((middle_sequence[i], middle_sequence[i + 1]))

            edges.append((start_node, middle_start))
            edges.append((middle_end, end_node))

        middle_sequence.append(end_node)
        middle_sequence.insert(0, start_node)

        self.serial_number += len(edges) - 1

        return edges, middle_sequence

    def generate(self, shared_path, hoplist, leafnodes, measurenodes):
        num_leafnode = len(leafnodes)
        num_measurenode = len(measurenodes)
        label = {}

        G = nx.Graph()
        sequence_measure = [i for i in range(self.serial_number, self.serial_number + num_measurenode)]
        G.add_nodes_from(sequence_measure, size=100, color='yellow')
        self.serial_number += num_measurenode

        for i in range(num_measurenode):
            label[i] = measurenodes[i]

        for i in range(num_measurenode):
            # initial path between first leafnode and measurenode
            if num_measurenode not in label:
                label[num_measurenode] = leafnodes[0]

            edges, path_temp = self.generate_edges_by_path(num_measurenode, i, int(hoplist[0][i]))
            G.add_edges_from(edges)

            for j in range(1, num_leafnode):
                try:
                    shared_path_temp = round(shared_path[leafnodes[j - 1]][leafnodes[j]][measurenodes[i]])
                except KeyError:
                    shared_path_temp = round(shared_path[leafnodes[j]][leafnodes[j - 1]][measurenodes[i]])

                if shared_path_temp + 1 > len(path_temp):
                    shared_path_temp = len(path_temp) - 1
                shared_node_end = path_temp[-shared_path_temp - 1]

                if shared_path_temp >= hoplist[j][i]:
                    if shared_node_end not in label and i == 0:
                        label[shared_node_end] = leafnodes[j]

                    path_temp = path_temp[-shared_path_temp-1:-1]
                else:
                    if self.serial_number not in label and i == 0:
                        label[self.serial_number] = leafnodes[j]

                    edges, path_temp = self.generate_edges_by_path(self.serial_number, i, int(hoplist[j][i] - shared_path_temp))
                    G.add_edges_from(edges)

        return G, label
