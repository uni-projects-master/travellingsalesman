import matplotlib.pyplot as plt


sizes = []
prim_times = []
nearest_times  = []
random_times = []
f = open('graph_results.txt')
line = f.readline()
lines = f.read().splitlines()
for i in range(len(lines)):
	row = lines[i].split()
	for j in range(len(row)):
		if j == 0:
			sizes.append(int(row[0]))
		if j == 1:
			prim_times.append(int(row[1]))
		if j == 2:
			nearest_times.append(int(row[2]))
		if j == 3:
			random_times.append(int(row[3]))

sizes.sort()
prim_times.sort()
nearest_times.sort()
random_times.sort()

# --------------------------------------FIRST FIGURE: COMPLETE GRAPH------------------------------------------------
plt.figure()
plt.plot(sizes, prim_times, label='Prim times')
plt.plot(sizes, nearest_times, label='Nearest neighbor times')
plt.plot(sizes, random_times, label='Random insertion times')
plt.legend()
plt.xlabel('Number of nodes')
plt.ylabel('Run times (ns)')

# --------------------------------------SECOND FIGURE: PRIM GRAPH----------------------------------------------------
plt.figure()
plt.plot(sizes, prim_times, label='Prim times')
plt.legend()
plt.xlabel('Number of nodes')
plt.ylabel('Run times (ns)')

# --------------------------------------THIRD FIGURE: NEAREST GRAPH------------------------------------------------
plt.figure()
plt.plot(sizes, nearest_times, label='Nearest neighbor times')
plt.legend()
plt.xlabel('Number of nodes')
plt.ylabel('Run times (ns)')

# --------------------------------------FOURTH FIGURE: RANDOM GRAPH------------------------------------------------
plt.figure()
plt.plot(sizes, random_times, label='Random insertion times')
plt.legend()
plt.xlabel('Number of nodes')
plt.ylabel('Run times (ns)')

plt.show()
