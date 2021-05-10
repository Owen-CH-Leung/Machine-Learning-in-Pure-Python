import networkx as nx

def breadth_first_search(graph, start):
    visited = []
    queue = [start]
    while queue:        
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbors = graph.neighbors(node)           
            for n in neighbors:
                queue.append(n)
    return visited

def depth_first_search(graph, node, path=[]):
    path += [node]
    for n in graph.neighbors(node):
        if n not in path:
            path = depth_first_search(graph, n, path)
    return path

if __name__ == '__main__':
    graph_dict = {'A':['B', 'C', 'D'],
             'B':['E'],
             'C':['F'],
             'D':['G', 'H'],
             'E':[],
             'F':['I', 'J'],
             'G':[],
             'H':[],
             'I':[],
             'J':[]
            } 
    G = nx.DiGraph(graph_dict)
    nx.draw(G, with_labels=True)
    print(breadth_first_search(G, 'A'))
    print(depth_first_search(G, 'A'))