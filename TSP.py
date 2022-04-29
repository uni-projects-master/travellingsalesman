import random
import os
import gc
import math
from prim import prim
from prim import Graph
from time import perf_counter_ns


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
			min_x = i.x - latitude
			radian_x = PI*(latitude + 5.0 * min_x / 3.0)/180.0
			longitude = int(i.y)
			min_y = i.x - longitude
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


def preorder(traverse, root):
	if root:
		traverse.append(root.Name)
		for i in root.Children:
			preorder(traverse, i)


def approx_metric_tsp(problem):
	tsp_graph = Graph(problem.points, problem.distances)
	r = random.choice(tsp_graph.vertices)
	H_cycle_cost = prim(tsp_graph, r)
	H_cycle = []
	preorder(H_cycle, r)
	H_cycle_cost += tsp.distances[str(r.Name) + ' ' + str(H_cycle[-1])]
	H_cycle.append(r.Name)
	print('Cycle cost in approx metric tsp: ', H_cycle_cost)
	return H_cycle


def nearest_neighbor(tsp):
	r = random.choice(tsp.points)
	H_cycle = []
	H_cycle_cost = 0
	Q = tsp.points
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

	print('Cycle cost in nearest neighbor: ', H_cycle_cost)
	return H_cycle


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

	for file in sorted(os.listdir(directory)):
		filename = os.fsencode(file)
		filename = filename.decode("utf-8")

		if filename.endswith('.tsp'):
			print('looking at: ', filename)
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
			#tsp.get_TSP()
			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH PRIM')
			approx_metric_tsp(tsp)
			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH NEAREST NEIGHBOR')
			nearest_neighbor(tsp_copy)
			print('------------------------------------------------')
			print('CURRENTLY APPROXIMATING WITH CHEAPEST INSERTION')
			cheapest_insertion(tsp)
			print('------------------------------------------------')

