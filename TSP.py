import random
import os
import gc
import copy
import math
from prim import prim
from prim import Graph
from time import perf_counter_ns
import xlsxwriter


# class object node, for representing points
class Node:
    def __init__(self, name=None, x=0, y=0):
        self.Name = name
        # x coordinates
        self.x = x
        # y coordiantes
        self.y = y


# class object TSP, for representing TSP graph
class TSP:
    def __init__(self, dim, problem_type, points):
        self.dimension = dim
        self.edge_weight_type = problem_type
        # array of Node objects
        self.points = []
        # dictionary for storing the distances from each point
        self.distances = {}
        # reading dataset file into object TSP
        for n in points:
            point = n.split()
            if point and 'EOF' not in point:
                # Point Name, X coordiantes, Y coordinates
                self.points.append(Node(point[0], float(point[1]), float(point[2])))
        # Checking the distance Type, EUC or GEO
        if self.edge_weight_type == 'GEO':
            self.geo_format()
        # Storing precomputed distances
        for v in self.points:
            for w in self.points:
                # Using Nodes as key for edges
                edge1 = str(v.Name) + ' ' + str(w.Name)
                edge2 = str(v.Name) + ' ' + str(w.Name)
                if self.edge_weight_type == 'GEO':
                    d = geo_distance(v, w)
                else:
                    d = euc_distance(v, w)
                # saving the distances as value for undirected graph
                self.distances[edge1] = d
                self.distances[edge2] = d

    def geo_format(self):
        PI = 3.141592
        # Converting Latitude and Longitude to Radians
        for i in self.points:
            latitude = int(i.x)
            min_x = abs(i.x - latitude)
            radian_x = PI * (latitude + 5.0 * min_x / 3.0) / 180.0
            longitude = int(i.y)
            min_y = i.y - longitude
            radian_y = PI * (longitude + 5.0 * min_y / 3.0) / 180.0
            i.x = radian_x
            i.y = radian_y

    def get_TSP(self):
        print('--------------------dimension----------------')
        print(self.dimension)
        print('--------------------file type----------------')
        print(self.edge_weight_type)
        print('--------------------vertex-------------------')
        for i in self.points:
            print(i.Name, '', i.x, '', i.y)

        print('-------------------distances------------------')
        for d in self.distances:
            print(d, '', self.distances[d])


# Function for computing the Geometric distances
def geo_distance(point1, point2):
    RRR = 6378.388
    q1 = math.cos(point1.y - point2.y)
    q2 = math.cos(point1.x - point2.x)
    q3 = math.cos(point1.x + point2.x)
    geo_d = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)))
    return geo_d


# Function for computing Euclidean distances
def euc_distance(point1, point2):
    d1 = (point1.x - point2.x) ** 2
    d2 = (point1.y - point2.y) ** 2
    euc_d = round(math.sqrt(d1 + d2))
    return euc_d


# [Recursive] Preorder Tree Traversal for 2-Approximation algorithm
def preorder(traverse, root):
    # While all the nodes of the tree are not visited
    if root:
        # visit the node (add to list)
        traverse.append(root.Name)
        # call function on all the children of that node
        for i in root.Children:
            preorder(traverse, i)


# 2-Approximation ALgorithm based on MST
def approx_metric_tsp(problem):
    # Cycle Initialization
    H_cycle = []
    H_cycle_cost = 0
    # Building a MST graph data structure to apply prim's algorithm
    tsp_graph = Graph(problem.points, problem.distances)
    # Starting from a random point
    r = random.choice(tsp_graph.vertices)
    prim(tsp_graph, r)
    preorder(H_cycle, r)
    # connceting the last node to the root to make the cycle
    H_cycle.append(r.Name)
    # computing the hamiltonian cycle cost
    for i in range(len(H_cycle) - 1):
        H_cycle_cost += tsp.distances[str(H_cycle[i]) + ' ' + str(H_cycle[i + 1])]
    return H_cycle_cost


# Nearest Neighbor Heuristic
def nearest_neighbor(tsp):
    # creating a duplicate of TSP
    tsp_copy = copy.deepcopy(tsp)
    # initializing solution
    H_cycle = []
    H_cycle_cost = 0
    # creating a queue of points to extract from
    Q = tsp_copy.points
    # starting from a random point
    r = random.choice(tsp_copy.points)
    # adding to solution
    H_cycle.append(r.Name)
    Q.remove(r)
    u = r
    # While all the points are not included in the cycle
    while Q:
        nearest_value = 99999999
        nearest = Node()
        # finding the nearest neighbor
        for v in Q:
            # create the partial circuit u,v
            current_edge = u.Name + ' ' + v.Name
            if tsp_copy.distances[current_edge] < nearest_value:
                nearest_value = tsp_copy.distances[current_edge]
                nearest = v
        # adding the nearest neighbor of u to the cycle
        Q.remove(nearest)
        H_cycle.append(nearest.Name)
        # updating the cycle cost
        H_cycle_cost += nearest_value
        u = nearest
    # adding the starting node to complete cycle
    H_cycle.append(r.Name)
    H_cycle_cost += tsp_copy.distances[str(r.Name) + ' ' + str(H_cycle[-1])]

    return H_cycle_cost


# Random Insertion Heuristic
def random_insertion(tsp):
    # Creating a duplicate of TSP graph
    tsp_copy = copy.deepcopy(tsp)
    H_cycle = []
    H_cycle_cost = 0

    Q = tsp_copy.points
    # Starting from a random point
    root = random.choice(tsp_copy.points)
    Q.remove(root)

    min_init_value = 999999
    min_init_edge = ''
    min_init_vertex = Node()
    # Create the partial circuit (root, v) that minimizes w(root, v)
    for v in Q:
        current_edge = root.Name + ' ' + v.Name
        if tsp_copy.distances[current_edge] < min_init_value:
            min_init_value = tsp_copy.distances[current_edge]
            min_init_edge = current_edge
            min_init_vertex = v
    Q.remove(min_init_vertex)
    H_cycle.append(min_init_edge)
    H_cycle_cost += min_init_value

    # While all the points are not included in the cycle
    while Q:
        # Randomly select a vertex k not in the circuit
        k = random.choice(Q)
        Q.remove(k)
        min_insert_edge = ''
        min_insert_value = 9999999
        min_insert_index = -1
        # find the edge u,v that minimizes the property w(u,k) + w(k,v) - w(u,v)
        for i in range(len(H_cycle)):
            v_pair = H_cycle[i].split()
            # buliding the edge u,k
            temp_edge_1 = v_pair[0] + ' ' + k.Name
            # building the edge k,v
            temp_edge_2 = k.Name + ' ' + v_pair[1]
            current_path_cost = tsp_copy.distances[temp_edge_1] + tsp_copy.distances[temp_edge_2] - \
                                tsp_copy.distances[H_cycle[i]]
            if current_path_cost < min_insert_value:
                min_insert_value = current_path_cost
                min_insert_edge = H_cycle[i]
                min_insert_index = i
        # inserting the random vertex k, into the minimizing edge w(u,v)
        # For inserting node 4 into edge (3,5) in this cycle:
        # '1 2'  '3 5'  '6 9'
        # First remove '3 5'
        # '1 2' '6 9'
        # Inserting the two edges '3 4' '4 5'
        # '1 2' '3 4' '4 5' '6 9'
        removing_edge = H_cycle[min_insert_index].split()
        insertion_edge_1 = removing_edge[0] + ' ' + str(k.Name)
        insertion_edge_2 = str(k.Name) + ' ' + removing_edge[1]
        # index removing the edge
        H_cycle.pop(min_insert_index)
        # replacing the two minimizing edges
        H_cycle.insert(min_insert_index, insertion_edge_1)
        H_cycle.insert(min_insert_index + 1, insertion_edge_2)

    # Completing the cycle
    cycle_start = H_cycle[0].split()
    cycle_end = H_cycle[-1].split()
    cycle_edge = cycle_end[1] + ' ' + cycle_start[0]
    H_cycle.append(cycle_edge)

    # Computing the cycle cost
    for i in H_cycle:
        H_cycle_cost += tsp_copy.distances[i]

    return H_cycle_cost


# Function for measuring the run-time of the algorithms
def measure_run_times(tsp, num_calls, num_instances, function_called):
    sum_times = 0.0
    for i in range(num_instances):
        gc.disable()
        start_time = perf_counter_ns()
        for j in range(num_calls):
            if function_called == 'approx_metric_tsp':
                approx_metric_tsp(tsp)
            elif function_called == 'nearest_neighbor':
                nearest_neighbor(tsp)
            elif function_called == 'random_insertion':
                random_insertion(tsp)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time) / num_calls
    avg_time = int(round(sum_times / num_instances))
    # return average time in nanoseconds
    return avg_time


if __name__ == '__main__':

    dir_name = 'tsp_dataset'
    directory = os.fsencode(dir_name)

    num_calls = 100
    num_instances = 5
    # Storing Optimal Solutions for computing the errors
    optimal_solution = {'burma14.tsp': 3323,
                        'ulysses16.tsp': 6859,
                        'ulysses22.tsp': 7013,
                        'eil51.tsp': 426,
                        'berlin52.tsp': 7542,
                        'kroD100.tsp': 21294,
                        'kroA100.tsp': 21282,
                        'ch150.tsp': 6528,
                        'gr202.tsp': 40160,
                        'gr229.tsp': 134602,
                        'pcb442.tsp': 50778,
                        'd493.tsp': 35002,
                        'dsj1000.tsp': 18659688}
    # Saving runtimes for plotting results
    run_times_prim = []
    run_times_nearest = []
    run_times_cheapest = []
    run_times_random = []
    graph_sizes = []

    # opening a xlsx table to write the results of the experiments
    workbook = xlsxwriter.Workbook('result_table.xlsx')
    worksheet = workbook.add_worksheet()
    row = 2
    worksheet.write('A' + str(row), 'INSTANCE')
    # NEAREST NEIGHBBOR
    worksheet.write('B' + '1', 'NEAREST NEIGHBOR')
    worksheet.write('B' + str(row), 'SOLUTION')
    worksheet.write('C' + str(row), 'TIME')
    worksheet.write('D' + str(row), 'ERROR')
    # RANDOM INSERTION2
    worksheet.write('E' + '1', 'RANDOM INSERTION')
    worksheet.write('E' + str(row), 'SOLUTION')
    worksheet.write('F' + str(row), 'TIME')
    worksheet.write('G' + str(row), 'ERROR')
    # 2-APPROXIMATION
    worksheet.write('H' + '1', '2-APPROXIMATION')
    worksheet.write('H' + str(row), 'SOLUTION')
    worksheet.write('I' + str(row), 'TIME')
    worksheet.write('J' + str(row), 'ERROR')
    # CHEAPEST INSERTION
    worksheet.write('K' + '1', 'CHEAPEST INSERTION')
    worksheet.write('K' + str(row), 'SOLUTION')
    worksheet.write('L' + str(row), 'TIME')
    worksheet.write('M' + str(row), 'ERROR')

    # Reading all TSP datasets
    for file in sorted(os.listdir(directory)):
        filename = os.fsencode(file)
        filename = filename.decode("utf-8")

        if filename.endswith('.tsp'):
            row += 1
            print('looking at: ', filename)
            worksheet.write('A' + str(row), filename)

            # read all the needed information from the file
            f = open(dir_name + '/' + filename)
            f.readline()
            f.readline()
            f.readline()
            line = f.readline().split()
            dimension = line[1]
            graph_sizes.append(dimension)
            line = f.readline().split()
            edge_weight_type = line[1]

            while line != "NODE_COORD_SECTION\n":
                line = f.readline()

            points = f.read().splitlines()
            # create the tsp graph for the experiments
            tsp = TSP(dimension, edge_weight_type, points)

            '''
            FOREACH FUNCTION USED TO APPROXIMATE A SOLUTION FOR THE TSP PROBLEM:
            calculate the cost of the solution which is written in the final table
            run the function a fixed amount of times to calculate its times of execution
            append this results to a list 
            calculate the error and write all the data on the final table
            '''
            print('------------------------------------------------')
            print('CURRENTLY APPROXIMATING WITH PRIM')

            approx_cost = approx_metric_tsp(tsp)
            measured_time_prim = measure_run_times(tsp, num_calls, num_instances, 'approx_metric_tsp')
            run_times_prim.append(measured_time_prim)

            worksheet.write('H' + str(row), approx_cost)
            worksheet.write('I' + str(row), measured_time_prim)
            prim_error = (approx_cost - optimal_solution[filename]) / optimal_solution[filename]
            worksheet.write('J' + str(row), prim_error)

            print('------------------------------------------------')
            print('CURRENTLY APPROXIMATING WITH NEAREST NEIGHBOR')

            NN_cost = nearest_neighbor(tsp)
            measured_time_nearest = measure_run_times(tsp, num_calls, num_instances, 'nearest_neighbor')
            run_times_nearest.append(measured_time_nearest)

            worksheet.write('B' + str(row), NN_cost)
            worksheet.write('C' + str(row), measured_time_nearest)
            nearest_error = (NN_cost - optimal_solution[filename]) / optimal_solution[filename]
            worksheet.write('D' + str(row), nearest_error)

            print('------------------------------------------------')
            print('CURRENTLY APPROXIMATING WITH RANDOM INSERTION')

            RI_cost = random_insertion(tsp)
            measured_time_random = measure_run_times(tsp, num_calls, num_instances, 'random_insertion')
            run_times_random.append(measured_time_random)

            worksheet.write('E' + str(row), RI_cost)
            worksheet.write('F' + str(row), measured_time_random)
            random_error = (RI_cost - optimal_solution[filename]) / optimal_solution[filename]
            worksheet.write('G' + str(row), random_error)

            print('------------------------------------------------')

            f.close()

    # writing the results of the execution times on 'graphs.py' for plotting results
    with open('graph_results.txt', 'w+') as f_result:
        f_result.write("Sizes\tPrim\tNearest\tRandom\n")
        for i in range(len(graph_sizes)):
            f_result.write(
                "%s\t%s\t%s\t%s\n" % (graph_sizes[i], run_times_prim[i], run_times_nearest[i], run_times_random[i]))

    workbook.close()
