import random
import os
import gc
import math
from time import perf_counter_ns


class Vertex:
	def __init__(self, name, x, y):
		self.name = name
		self.x = x 
		self.y = y 


class Graph:
	def __init__(self, dimension, file_type, verteces):
		self.dimension = dimension
		self.file_type = file_type
		self.vertex_list = []
		
		for vertex in verteces:
			vertex = vertex.split()
			if vertex and 'EOF' not in vertex:
				self.vertex_list.append(Vertex(vertex[0], float(vertex[1]), float(vertex[2])))
		#print("BEFORE CONVERSION")
		#print(self.vertex_list[0].x, "  ", self.vertex_list[0].y)
		if self.file_type == 'GEO':
			self.geo_format()
		#print("AFTER CONVERSION")
		#print(self.vertex_list[0].x, "  ", self.vertex_list[0].y)

	def geo_format(self):
		PI = 3.141592
		# CONVERTING LATITUDE AND LONGTITUDE TO RADIAN
		for i in self.vertex_list:
			latitude = int(i.x)
			min_x = i.x - latitude
			radian_x = PI*(latitude + 5.0 * min_x / 3.0)/180.0
			longitude = int(i.y)
			min_y = i.x - longitude
			radian_y = PI * (longitude + 5.0 * min_y / 3.0) / 180.0
			i.x = radian_x
			i.y = radian_y

	def get_graph(self):
		print('--------------------dimension----------------')
		print(self.dimension)
		print('--------------------file type----------------')
		print(self.file_type)
		print('--------------------vertex-------------------')
		for i in range(len(self.vertex_list)):
			print(self.vertex_list[i].name)
			print(self.vertex_list[i].x)
			print(self.vertex_list[i].y)


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
			file_type = line[1]
			f.readline()
			f.readline()
			f.readline()
			verteces = f.read().splitlines()
			g = Graph(dimension, file_type, verteces)



