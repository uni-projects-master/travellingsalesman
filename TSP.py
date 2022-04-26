import random
import os
import gc
import math
from prim import prim
from prim import Graph
from time import perf_counter_ns


class Node:
	def __init__(self, name, x, y):
		self.Name = name
		self.x = x
		self.y = y


class TSP:
	def __init__(self, dim, type, points):
		self.dimension = dim
		self.edge_weight_type = type
		self.points = []
		self.distances = []
		
		for n in points:
			point = n.split()
			if point and 'EOF' not in point:
				self.points.append(Node(point[0], float(point[1]), float(point[2])))

		if self.edge_weight_type == 'GEO':
			self.geo_format()

		for v in self.points:
			for w in self.points:
				if self.edge_weight_type == 'GEO':
					d = geo_distance(v, w)
				else:
					d = euc_distance(v, w)
				self.distances.append(str(v.Name) + ' ' + str(w.Name) + ' ' + str(d))

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
			print(d)


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
	T_star = prim(tsp_graph, r)
	H_cycle = []
	preorder(H_cycle, r)
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
			f.readline()
			f.readline()
			f.readline()
			points = f.read().splitlines()
			tsp = TSP(dimension, edge_weight_type, points)
			#tsp.get_graph()
			print(approx_metric_tsp(tsp))
			#nearest_neighbor(tsp)
