import random
from collections import defaultdict, deque
import heapq
from tqdm import tqdm


def triples_to_graph(triples):
    """
    Convert a list of triples to an adjacency list representation of a graph.

    Args:
    - triples: a list of triples in the format (subject, predicate, object)

    Returns:
    - a dictionary where the keys are nodes and the values are lists of connected nodes
    """
    graph = defaultdict(list)
    for subject, predicate, obj in tqdm(triples):
        graph[subject].append((obj, 1))
        graph[obj].append((subject, 1))
    return graph


def dijkstra(graph, start_node):
    """
    Compute the shortest path from the start node to all other nodes in the graph using Dijkstra's algorithm.

    Args:
    - graph: a dictionary where the keys are nodes and the values are lists of tuples representing the edges (neighbor, weight)
    - start_node: the node to start the shortest path computation from

    Returns:
    - a dictionary where the keys are nodes and the values are the shortest path length from the start node
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances


def getTopk_node(start_node, k, distance_limit):
    """
    Get the k nodes with the longest distance to the start node in the graph with a distance greater than the given limit.

    Args:
    - triples: a list of triples in the format (subject, predicate, object)
    - start_node: the node to start the shortest path computation from
    - k: the number of nodes to return
    - distance_limit: the minimum distance from the start node that nodes must have to be considered

    Returns:
    - a list of k nodes with the longest distance to the start node and a distance greater than the given limit
    """
    distances = dijkstra(graph, start_node)
    # if start_node == '好':
    #     print(distances)
    topk_heap = [(-distance, node) for node, distance in distances.items() if distance > distance_limit or distance == 'inf']
    heapq.heapify(topk_heap)
    topk_nodes = []

    while len(topk_nodes) < k and topk_heap:
        distance, node = heapq.heappop(topk_heap)
        topk_nodes.append(node)

    return topk_nodes


def read_triples_from_file(file_path):
    """
    Read triples from a file in the format [head, tail, relation] and return them in the format (subject, predicate, object).
    Also switch the head and tail entities and add the resulting triple to the list of triples.

    Args:
    - file_path: the path to the input file

    Returns:
    - a tuple containing:
      - a list of triples in the format (subject, predicate, object)
      - a set of all entities that appear as heads in the triples
      - a set of all entities that appear as tails in the triples
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    triples = [(line.strip().split()[0], line.strip().split()[2], line.strip().split()[1]) for line in lines]
    newtriples = []
    for head, relation, tail in tqdm(triples):
        newtriples.append((tail, relation, head))
    triples = list(set(newtriples + triples))
    heads = set([triple[0] for triple in triples])
    tails = set([triple[2] for triple in triples])
    return triples, heads, tails


def find_negative_samples( entities, distance_limit, output_file, k=10):
    """
    Find all nodes in the graph that are farther than a given distance limit from each node in the entities set.
    Write the top k triples with the largest head-tail distances to a file in the format [head, tail, 'ANT'].

    Args:
    - graph: a dictionary representing the graph in adjacency list format
    - entities: a set of nodes to use as starting points
    - distance_limit: the maximum distance from each starting node
    - output_file: the path to the output file
    - k: the number of top triples to output

    Returns:
    - None
    """
    startNode2nodes = {}
    for start in tqdm(entities):
        nodes = getTopk_node(start, k, distance_limit)
        startNode2nodes[start] = nodes
    with open(output_file, 'w') as f:
        for head, tails in startNode2nodes.items():
            for tail in tails:
                f.write(f"{head}\t{tail}\tANT\n")

if __name__ == '__main__':
    triples, heads, tails = read_triples_from_file('./data/train.txt')
    entities = list(heads.union(tails))
    # print(triples, entities)
    start_node = entities[len(entities)//2]
    print(start_node)
    num_sets = 20
    path_length = 5
    graph = triples_to_graph(triples)

    node_sets = getTopk_node(start_node, num_sets, path_length)
    print(node_sets)
    find_negative_samples(entities, path_length, './data/negTrain.txt', num_sets)