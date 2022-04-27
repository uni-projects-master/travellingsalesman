import random
import matplotlib.pyplot as plt
import os
import gc
from time import perf_counter_ns


# VERTEX STRUCTURE FOR PRIM'S ALGORITHM
class vertex:
    def __init__(self, name):
        INF = 999999
        # VERTEX NAME
        self.Name = str(name)
        # VERTEX COST
        self.Value = INF
        # VERTEX PARENT
        self.Parent = None
        self.Children = []

    def reset(self):
        INF = 999999
        # VERTEX COST
        self.Value = INF
        # VERTEX PARENT
        self.Parent = None


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = []
        self.edges = {}
        self.num_vertex = len(vertices)
        self.num_edges = len(edges)
        # CREATE OBJECT VERTEX AND ADD TO VERTICES
        for v in vertices:
            new_vertex = vertex(name=v.Name)
            self.vertices.append(new_vertex)
            self.edges[new_vertex] = {}
        self.add_edges(edges)

    def reset(self):
        for v in self.vertices:
            v.reset()

    def add_edges(self, list_edges):
        for i in list_edges:
            edge = i.split()
            if edge[0] != edge[1]:
                u = self.vertices[int(edge[0])-1]
                v = self.vertices[int(edge[1])-1]
                w = int(list_edges[i])
                # SINCE GRAPH IS UNDIRECTED, ADD EDGE IN TWO WAYS
                self.edges[u][v] = w
                self.edges[v][u] = w

    def get_weight(self, u, v):
        return self.edges[u][v]

    def get_graph(self):
        for v in self.vertices:
            print("node ",v.Name," connect to ")
            for u in self.find_adj(v):
                print(" ",u.Name," with weight ",self.get_weight(u,v))
            print("")

    def find_adj(self, v):
        # find adjacents of a given vertex
        return self.edges[v].keys()



class minHeap:
    # Constructor to initialize a heap
    def __init__(self):
        self.Heap = []
        self.Size = 0
        self.valueToIndex = {}

    def extractHeapify(self, idx):
        # Update Heap to keep data min atg root
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2
        # if the child is smaller than parent, change indexes
        if left < len(self.Heap) and self.Heap[left].Value < self.Heap[smallest].Value:
            smallest = left
        if right < len(self.Heap) and self.Heap[right].Value < self.Heap[smallest].Value:
            smallest = right
        # if the smallest index has chnaged swap nodes and heapify again
        if smallest != idx:
            self.Heap[idx], self.Heap[smallest] = self.Heap[smallest], self.Heap[idx]
            self.valueToIndex[self.Heap[smallest]] = smallest
            self.valueToIndex[self.Heap[idx]] = idx
            self.extractHeapify(smallest)

    def extractMin(self):
        # Extract the minimum value (in root)
        if self.isEmpty():
            return
        root = self.Heap[0]
        # Substitite the root with last element
        self.Heap[0] = self.Heap[len(self.Heap) - 1]
        self.valueToIndex[self.Heap[0]] = 0
        self.Heap.pop()
        self.valueToIndex[root] = None
        # Update heap
        self.extractHeapify(0)

        return root

    def insertHeapify(self, idx):
        # update heap after insert
        parent = int(((idx - 1) / 2))
        # check if the inserted element is smaller than its parent
        if self.Heap[idx].Value < self.Heap[parent].Value:
            self.Heap[idx], self.Heap[parent] = self.Heap[parent], self.Heap[idx]
            self.valueToIndex[self.Heap[parent]] = parent
            self.valueToIndex[self.Heap[idx]] = idx
            self.insertHeapify(parent)

    def insert(self, v):
        # insert node at the end of heap
        self.Heap.append(v)
        self.valueToIndex[v] = len(self.Heap) - 1
        self.insertHeapify(self.valueToIndex[v])
        return

    def updateKey(self, v):
        # VALUE IS ALWAYS DECREASING, MOVE NODE UP THE TREE
        idx = self.valueToIndex[v]
        parent = int(((idx - 1) / 2))
        if self.Heap[idx].Value < self.Heap[parent].Value:
            self.Heap[idx], self.Heap[parent] = self.Heap[parent], self.Heap[idx]
            self.valueToIndex[self.Heap[parent]] = parent
            self.valueToIndex[self.Heap[idx]] = idx
            self.insertHeapify(parent)

    def isEmpty(self):
        return len(self.Heap) == 0

    def isInMinHeap(self, v):
        return self.valueToIndex[v] is not None

    def print_Heap(self):
        print("HEAP: ")
        for i in self.Heap:
            print("Name: ", i.Name, ", Value: ", i.Value)

def prim(G, r):
    # CHOOSE A STARTING POINT AT RANDOM
    start = r
    start.Value = 0
    A = set()
    # INITIALIZE HEAP FOR PRIM WITH NODE VALUES = INFINITY
    Q = minHeap()
    for ver in G.vertices:
        Q.insert(ver)
    # WHILE EVERY VERTEX IS NOT INCLUDED IN THE TREE
    while not Q.isEmpty():
        # EXTRACT NODE WITH MINIMUM VALUE AND FIND ITS ADJACENTS
        u = Q.extractMin()
        A.add(u)
        u_adj = G.find_adj(u)
        for v in u_adj:
            u_v_weight = G.get_weight(u,v)
            # IF EDGE (u,v) HAS A LOWER COST UPADTE THE VALUE OF v
            if v not in A and u_v_weight < v.Value:
                v.Parent = u
                v.Value = u_v_weight
                Q.updateKey(v)

    # COMPUTING WEIGHT OF THE TREE
    A_weight = 0
    for ver in G.vertices:
        # IF VERTEX IS NOT THE ROOT
        if ver.Parent is not None:
            ver_parent = ver.Parent
            ver_parent.Children.append(ver)
            A_weight += G.get_weight(ver, ver_parent)

    '''for ver in G.vertices:
        # IF VERTEX IS NOT THE ROOT
        print("CHILDREN OF ", ver.Name)
        for c in ver.Children:
            print(c.Name, end=',')
        print("")'''
    return A_weight


def measure_run_times(g, num_calls, num_instances):
    sum_times = 0.0
    for i in range(num_instances):
        gc.disable()
        start_time = perf_counter_ns()
        for j in range(num_calls):
            g.reset()
            prim(g)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    # return average time in nanoseconds
    return avg_time


if __name__ == '__main__':

    f = open('input_random_13_80.txt', 'r')

    line = f.readline().split()
    edge_list = f.read().splitlines()
    g = Graph(int(line[0]), int(line[1]))
    g.add_edges(edge_list)
    A_w = prim(g)
    print(A_w)
