import networkx as nx
import matplotlib.pyplot as plt

import adoug.network_generator as ng

leafNodes = ['622', '422', '322']
measureNodes = ['12']
hopList_gmm = [[3, 2], [2, 2], [1, 1]]
sharedPath = {
    '622': {
        '422': {
            '12': 2,
            '7': 1
        }
    },
    '422': {
        '322': {
            '12': 0,
            '7': 0
        }
    }
}

network_gmm, label = ng.NetworkGenerator().generate(sharedPath, hopList_gmm, leafNodes, measureNodes)
# pos = nx.spring_layout(network_gmm)
nx.draw(network_gmm, labels=label)
#
# G = nx.Graph()
# G.add_edges_from([(0, 1), (0, 3)])
# pos = nx.spring_layout(G)
# nx.draw(G)

plt.axis('off')
plt.show()
