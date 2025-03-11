# Travelling Salesman Problem Solutions

This repository contains implementations of various algorithms to solve the Travelling Salesman Problem (TSP), aiming to find the shortest possible route that visits a set of cities and returns to the origin point.

## Algorithms Implemented

- **Brute Force**: Evaluates all possible permutations to determine the shortest route.
- **Greedy Algorithm**: Constructs a solution by selecting the nearest unvisited city at each step.
- **2-Opt Heuristic**: Improves an existing route by iteratively swapping two edges to reduce the total distance.
- **k-Opt Heuristic**: Generalization of the 2-Opt method, removing 'k' edges and reconnecting the fragments to form a shorter tour.
- **Lin-Kernighan Algorithm**: An advanced heuristic that systematically explores different edge exchanges to find near-optimal solutions.
