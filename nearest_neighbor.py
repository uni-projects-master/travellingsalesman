import random
import os
import gc
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

			if vertex or vertex[0] != 'EOF':
				print(vertex)
				self.vertex_list.append(Vertex(vertex[0], vertex[1], vertex[2]))

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


if __name__ == '__main__':

	dir_name = 'tsp_dataset'
	directory = os.fsencode(dir_name)

	for file in sorted(os.listdir(directory)):
		filename = os.fsencode(file)
		filename = filename.decode("utf-8")

		if(filename.endswith('.tsp')):
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
			g.get_graph()

