import random
import os
import gc
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
				#self.distances.append(edge_name + ' ' + str(d))
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


def preorder(traverse, traverse_cost, root):
	if root:
		traverse.append(root.Name)
		for i in root.Children:
			preorder(traverse, traverse_cost, i)


def approx_metric_tsp(problem):
	tsp_graph = Graph(problem.points, problem.distances)
	r = random.choice(tsp_graph.vertices)
	H_cycle_cost = prim(tsp_graph, r)
	H_cycle = []
	preorder(H_cycle, H_cycle_cost, r)
	H_cycle_cost += tsp.distances[str(r.Name) + ' ' + str(H_cycle[-1])]
	H_cycle.append(r.Name)
	return H_cycle_cost


def nearest_neighbor(tsp):
	H_cycle = []
	H_cycle_cost = 0
	Q = tsp.points
	r = random.choice(tsp.points)
	Q.remove(r)
	u = r
	while Q:
		nearest_value = 999999
		nearest = Node()
		# FIND THE NEAREST NEIGHBOR
		for v in Q:
			current_edge = u.Name + ' ' + v.Name
			if tsp.distances[current_edge] < nearest_value:
				nearest_value = tsp.distances[current_edge]
				nearest = v
		Q.remove(nearest)
		H_cycle.append(nearest.Name)
		H_cycle_cost += nearest_value
		u = nearest
	H_cycle.append(r.Name)
	H_cycle_cost += tsp.distances[str(r.Name) + ' ' + str(H_cycle[-1])]
	print('Cycle cost in nearest neighbor: ', H_cycle_cost)
	return H_cycle_cost


def random_insertion(tsp):

	H_cycle = []
	H_cycle_cost = 0

	Q = tsp.points
	root = random.choice(tsp.points)
	#root = tsp.points[0]
	Q.remove(root)

	min_init_value = 999999
	min_init_edge = ''
	min_init_vertex = Node()
	# Create the partial circuit (0,j)
	for v in Q:
		current_edge = root.Name + ' ' + v.Name
		if tsp.distances[current_edge] < min_init_value:
			min_init_value = tsp.distances[current_edge]
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
		min_insert_index = -9999
		for i in range(len(H_cycle)):
			v_pair = H_cycle[i].split()
			temp_edge_1 = v_pair[0] + ' ' + rand_v.Name
			temp_edge_2 = v_pair[1] + ' ' + rand_v.Name
			current_path_cost = tsp.distances[temp_edge_1] + tsp.distances[temp_edge_2] - tsp.distances[H_cycle[i]]
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
		H_cycle_cost -= tsp.distances[min_insert_edge]
		H_cycle_cost += tsp.distances[sol_edge1] + tsp.distances[sol_edge2]

	#making it a cycle
	cycle_start = H_cycle[0].split()
	cycle_end = H_cycle[-1].split()
	cycle_edge = cycle_start[0] + ' ' + cycle_end[1]
	H_cycle.append(cycle_edge)
	H_cycle_cost += tsp.distances[cycle_edge]
	#print(H_cycle)
	#print(H_cycle_cost)
	return H_cycle_cost


def cheapest_insertion(tsp):
	# initialization
	r = random.choice(tsp.points)
	H_cycle = []
	H_cycle_edges = []
	H_cycle_cost = 0

	H_cycle.append(r.Name)
	nearest_value = 999999
	nearest = Node()
	for vertex in tsp.points:
		current_edge = r.Name + ' ' + vertex.Name
		if tsp.distances[current_edge] < nearest_value and tsp.distances[current_edge] != 0:
			nearest_value = tsp.distances[current_edge]
			nearest = vertex
	H_cycle.append(nearest.Name)
	new_edge = r.Name + ' ' + nearest.Name
	H_cycle_edges.append(new_edge)
	
	# repeat until there are no more nodes
	Q = tsp.points
	Q.remove(r)
	Q.remove(nearest)
	while Q:
		candidate = Node()
		node_i = Node()
		node_j = Node()
		new_weight = 999999
		# selection
		for k in tsp.points:
			# find a vertex k not in the circuit
			if k.Name not in H_cycle: 
				#find an edge (i,j) of the circuit that minimizes w(i,k) + w(k,j) - w(i,j)
				for edge in H_cycle_edges:
					vertices_i_j = edge.split()
					node_i.Name = vertices_i_j[0]
					node_j.Name = vertices_i_j[1]
					weight_i_k = tsp.distances[vertices_i_j[0] + ' ' + k.Name]
					weight_k_j = tsp.distances[k.Name + ' ' + vertices_i_j[1]]
					weight_i_j = tsp.distances[vertices_i_j[0] + ' ' + vertices_i_j[1]]
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

	print('Cycle cost in cheapest insertion: ', H_cycle_cost)
	H_cycle.append(r.Name)
	return H_cycle


if __name__ == '__main__':

	dir_name = 'tsp_dataset'
	directory = os.fsencode(dir_name)

	workbook = xlsxwriter.Workbook('table.xlsx')
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

	for file in sorted(os.listdir(directory)):
		filename = os.fsencode(file)
		filename = filename.decode("utf-8")

		if filename.endswith('.tsp'):
			row += 1
			print('looking at: ', filename)
			worksheet.write('A' + str(row), filename)

			f = open(dir_name + '/' + filename)
			f.readline()
			f.readline()
			f.readline()
			line = f.readline().split()
			dimension = line[1]
			line = f.readline().split()
			edge_weight_type = line[1]

			while line != "NODE_COORD_SECTION\n":
				line = f.readline()

			points = f.read().splitlines()
			tsp = TSP(dimension, edge_weight_type, points)
			tsp_copy = TSP(dimension, edge_weight_type, points)
			for i in tsp.points:
				print(i.Name, " ", i.x, " ", i.y)
			#tsp.get_TSP()
			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH PRIM')
			gc.disable()
			start_time = perf_counter_ns()
			approx_cost = approx_metric_tsp(tsp)
			end_time = perf_counter_ns()
			gc.enable()
			print(approx_cost)
			approx_time = end_time - start_time
			worksheet.write('H' + str(row), approx_cost)
			worksheet.write('I' + str(row), approx_time)

			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH NEAREST NEIGHBOR')
			gc.disable()
			start_time = perf_counter_ns()
			NN_cost = nearest_neighbor(tsp_copy)
			print(NN_cost)
			end_time = perf_counter_ns()
			gc.enable()
			NN_time = end_time - start_time
			worksheet.write('B' + str(row), NN_cost)
			worksheet.write('C' + str(row), NN_time)

			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH RANDOM INSERTION')
			gc.disable()
			start_time = perf_counter_ns()
			#RI_cost = random_insertion(tsp)
			end_time = perf_counter_ns()
			gc.enable()
			RI_time = end_time - start_time
			#worksheet.write('E' + str(row), RI_cost)
			worksheet.write('F' + str(row), RI_time)

			print('------------------------------------------------')
	workbook.close()
