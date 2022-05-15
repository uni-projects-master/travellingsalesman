import random
import os
import gc
import copy
import math
from prim import prim
from prim import Graph
from time import perf_counter_ns
import xlsxwriter


class Node:
	def __init__(self, name=None, x=0, y=0):
		self.Name = name
		self.x = x
		self.y = y


class TSP:
	def __init__(self, dim, type, points):
		self.dimension = dim
		self.edge_weight_type = type
		self.points = []
		# we use a dictionary for storing the distances from each point 
		self.distances = {}

		for n in points:
			point = n.split()
			if point and 'EOF' not in point:
				self.points.append(Node(point[0], float(point[1]), float(point[2])))

		if self.edge_weight_type == 'GEO':
			self.geo_format()

		for v in self.points:
			for w in self.points:
				edge1 = str(v.Name) + ' ' + str(w.Name)
				edge2 = str(v.Name) + ' ' + str(w.Name)
				if self.edge_weight_type == 'GEO':
					d = geo_distance(v, w)
				else:
					d = euc_distance(v, w)
				# since we are dealing with indirected graphs we save the distances for both points
				self.distances[edge1] = d
				self.distances[edge2] = d

	def geo_format(self):
		PI = 3.141592
		# CONVERTING LATITUDE AND LONGTITUDE TO RADIAN
		for i in self.points:
			latitude = int(i.x)
			min_x = abs(i.x - latitude)
			radian_x = PI*(latitude + 5.0 * min_x / 3.0)/180.0
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


# distances function as reported in the FAQ
def geo_distance(point1, point2):
	RRR = 6378.388
	q1 = math.cos(point1.y - point2.y)
	q2 = math.cos(point1.x - point2.x)
	q3 = math.cos(point1.x + point2.x)
	geo_d = int(RRR*math.acos(0.5*((1.0 + q1)*q2 - (1.0 - q1)*q3)))
	return geo_d


def euc_distance(point1, point2):
	d1 = (point1.x - point2.x)**2
	d2 = (point1.y - point2.y)**2
	euc_d = round(math.sqrt(d1 + d2))
	return euc_d


def preorder(traverse, root):
	if root:
		traverse.append(root.Name)
		for i in root.Children:
			preorder(traverse, i)


def approx_metric_tsp(problem):
	# we need to convert our TSP graph to a similar data structure used in Prim
	tsp_graph = Graph(problem.points, problem.distances)
	r = random.choice(tsp_graph.vertices)
	H_cycle_cost = 0
	prim(tsp_graph, r)
	H_cycle = []
	preorder(H_cycle, r)
	# calculate the final weight of the cycle
	for i in range(len(H_cycle)-1):
		H_cycle_cost += tsp.distances[str(H_cycle[i]) + ' ' + str(H_cycle[i+1])]

	H_cycle_cost += tsp.distances[str(H_cycle[-1]) + ' ' + str(r.Name)]
	# complete the cycle
	H_cycle.append(r.Name)
	return H_cycle_cost


def nearest_neighbor(tsp):
	# create a deep copy of the original tsp graph, since most of the euristics end up deleting nodes from the original data structure
	tsp_copy = copy.deepcopy(tsp)
	H_cycle = []
	H_cycle_cost = 0
	Q = tsp_copy.points
	r = random.choice(tsp_copy.points)
	H_cycle.append(r.Name)
	Q.remove(r)
	u = r
	# while there are still points in the data structure
	while Q:
		nearest_value = 99999999
		nearest = Node()
		# FIND THE NEAREST NEIGHBOR
		for v in Q:
			current_edge = u.Name + ' ' + v.Name
			if tsp_copy.distances[current_edge] < nearest_value:
				nearest_value = tsp_copy.distances[current_edge]
				nearest = v
		# remove the node from the list of non inserted nodes and insert it in the solution
		Q.remove(nearest)
		H_cycle.append(nearest.Name)
		H_cycle_cost += nearest_value
		u = nearest
	H_cycle_cost += tsp_copy.distances[str(r.Name) + ' ' + str(H_cycle[-1])]
	H_cycle.append(r.Name)

	return H_cycle_cost


def random_insertion(tsp):
	# create a deep copy of the original tsp graph, since most of the euristics end up deleting nodes from the original data structure
	tsp_copy = copy.deepcopy(tsp)
	H_cycle = []
	H_cycle_cost = 0

	Q = tsp_copy.points
	root = random.choice(tsp_copy.points)
	Q.remove(root)

	min_init_value = 999999
	min_init_edge = ''
	min_init_vertex = Node()
	# Create the partial circuit (0,j)
	for v in Q:
		current_edge = root.Name + ' ' + v.Name
		if tsp_copy.distances[current_edge] < min_init_value:
			min_init_value = tsp_copy.distances[current_edge]
			min_init_edge = current_edge
			min_init_vertex = v

	Q.remove(min_init_vertex)
	H_cycle.append(min_init_edge)
	H_cycle_cost += min_init_value
	while Q:
		# Randomly select a vertex k not in the circuit
		rand_v = random.choice(Q)
		Q.remove(rand_v)
		min_insert_edge = ''
		min_insert_value = 9999999
		min_insert_index = -1
		for i in range(len(H_cycle)):
			v_pair = H_cycle[i].split()
			temp_edge_1 = v_pair[0] + ' ' + rand_v.Name
			temp_edge_2 = v_pair[1] + ' ' + rand_v.Name
			current_path_cost = tsp_copy.distances[temp_edge_1] + tsp_copy.distances[temp_edge_2] - tsp_copy.distances[H_cycle[i]]
			if current_path_cost < min_insert_value:
				min_insert_value = current_path_cost
				min_insert_edge = H_cycle[i]
				min_insert_index = i

		# '1 2'  '3 5'  '6 9'
		# REMOVING '3 5'
		# '1 2' '6 9'
		# ADDING VERTEX 4 BETWEEN EDGE '3 5'
		# '1 2' '3 4' '4 5' '6 9'
		#H_cycle_cost -= H_
		solution_pair = H_cycle[min_insert_index].split()
		sol_edge1 = solution_pair[0] + ' ' + str(rand_v.Name)
		sol_edge2 = str(rand_v.Name) + ' ' + solution_pair[1]
		H_cycle.pop(min_insert_index)
		H_cycle.insert(min_insert_index, sol_edge1)
		H_cycle.insert(min_insert_index+1, sol_edge2)

		#The cost
		#H_cycle_cost -= tsp_copy.distances[min_insert_edge]
		#H_cycle_cost += tsp_copy.distances[sol_edge1] + tsp_copy.distances[sol_edge2]

	# making it a cycle
	for i in H_cycle:
		H_cycle_cost += tsp_copy.distances[i]

	cycle_start = H_cycle[0].split()
	cycle_end = H_cycle[-1].split()
	cycle_edge = cycle_end[1] + ' ' + cycle_start[0]
	H_cycle.append(cycle_edge)
	H_cycle_cost += tsp_copy.distances[cycle_edge]
	print(H_cycle)
	print(H_cycle_cost)
	return H_cycle_cost


def cheapest_insertion(tsp):
	# create a deep copy of the original tsp graph, since most of the euristics end up deleting nodes from the original data structure	tsp_copy = copy.deepcopy(tsp)
	r = random.choice(tsp_copy.points)
	H_cycle = []
	H_cycle_edges = []
	H_cycle_cost = 0

	H_cycle.append(r.Name)
	nearest_value = 999999
	nearest = Node()
	for vertex in tsp_copy.points:
		current_edge = r.Name + ' ' + vertex.Name
		if tsp_copy.distances[current_edge] < nearest_value and tsp_copy.distances[current_edge] != 0:
			nearest_value = tsp_copy.distances[current_edge]
			nearest = vertex
	H_cycle.append(nearest.Name)
	new_edge = r.Name + ' ' + nearest.Name
	H_cycle_edges.append(new_edge)
	
	# repeat until there are no more nodes
	Q = tsp_copy.points
	Q.remove(r)
	Q.remove(nearest)
	while Q:
		candidate = Node()
		node_i = Node()
		node_j = Node()
		new_weight = 999999
		# selection
		for k in tsp_copy.points:
			# find a vertex k not in the circuit
			if k.Name not in H_cycle: 
				#find an edge (i,j) of the circuit that minimizes w(i,k) + w(k,j) - w(i,j)
				for edge in H_cycle_edges:
					vertices_i_j = edge.split()
					node_i.Name = vertices_i_j[0]
					node_j.Name = vertices_i_j[1]
					weight_i_k = tsp_copy.distances[vertices_i_j[0] + ' ' + k.Name]
					weight_k_j = tsp_copy.distances[k.Name + ' ' + vertices_i_j[1]]
					weight_i_j = tsp_copy.distances[vertices_i_j[0] + ' ' + vertices_i_j[1]]
					if (weight_i_k + weight_k_j - weight_i_j) < new_weight:
						new_weight = weight_i_k + weight_k_j - weight_i_j
						candidate = k

		# insertion
		index_of_i = H_cycle.index(node_i.Name)
		# insertion in the solution
		H_cycle.insert(index_of_i + 1, candidate.Name)
		H_cycle_cost += new_weight
		# insertion in the edge list to keep integrity
		for edge in H_cycle_edges:
			vertices_i_j = edge.split()
			if vertices_i_j[0] == node_i.Name and vertices_i_j[1] == node_j.Name:
				index_of_i_j = H_cycle_edges.index(edge)
				H_cycle_edges[index_of_i_j] = vertices_i_j[0] + ' ' + candidate.Name
				H_cycle_edges.insert(index_of_i_j + 1, candidate.Name + ' ' + vertices_i_j[1])

		Q.remove(candidate)

	H_cycle.append(r.Name)
	return H_cycle_cost


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
			elif function_called == 'cheapest_insertion':
				cheapest_insertion(tsp)

		end_time = perf_counter_ns()
		gc.enable()
		sum_times += (end_time - start_time)/num_calls
	avg_time = int(round(sum_times/num_instances))
	# return average time in nanoseconds
	return avg_time


if __name__ == '__main__':

	dir_name = 'tsp_dataset'
	directory = os.fsencode(dir_name)

	num_calls = 100
	num_instances = 5
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
	# RANDOM INSERTION
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

	# foreach file in the dataset
	for file in sorted(os.listdir(directory)):
		filename = os.fsencode(file)
		filename = filename.decode("utf-8")

		if filename.endswith('.tsp'):
			row += 1
			print('looking at: ', filename)
			worksheet.write('A' + str(row), filename)

			# read all the needed informations from the file
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

			'''
			THE CHEAPEST INSERTION EURISTIC WAS IMPLEMENTED BUT NOT COVERED IN THE RESULTS OF THE EXPERIMENT. 
			ITS RESULTS WERE SHOWN TO BE CORRECT
			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH CHEAPEST INSERTION')

			
			CI_cost = cheapest_insertion(tsp)
			measured_time_cheapest = measure_run_times(tsp, num_calls, num_instances, 'cheapest_insertion')
			run_times_cheapest.append(measured_time_cheapest)

			worksheet.write('K' + str(row), CI_cost)
			worksheet.write('L' + str(row), measured_time_cheapest)
			cheapest_error = (CI_cost - optimal_solution[filename]) / optimal_solution[filename]
			worksheet.write('M' + str(row), cheapest_error)
			'''

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

	# we write on a file the results for the execution times, which are processed in the 'graphs.py' file
	with open('graph_results.txt', 'w+') as f_result:
		f_result.write("Sizes\tPrim\tNearest\tRandom\n")
		for i in range(len(graph_sizes)):
			f_result.write("%s\t%s\t%s\t%s\n" % (graph_sizes[i], run_times_prim[i], run_times_nearest[i], run_times_random[i]))

	workbook.close()
