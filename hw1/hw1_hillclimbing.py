#111210552 林小蓮

import random

def generate_distances(n):
    d = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = random.randint(10, 100)
            d[i][j] = d[j][i] = dist
    return d

def hill_climbing_tsp(distances):
    n = len(distances)
    solution = list(range(n))

    def height(route):
        total = 0
        for i in range(len(route)):
            total += distances[route[i]][route[(i+1) % len(route)]]
        return -total

    def neighbor(route):
        i, j = sorted(random.sample(range(n), 2))
        route[i:j+1] = reversed(route[i:j+1])
        return route

    current = solution
    current_height = height(current)
    for _ in range(1000):
        new = neighbor(current[:])
        if height(new) > current_height:
            current = new
            current_height = height(new)
    return current

# Generate test case
n = 5
distances = generate_distances(n)

print("Distance matrix:")
for row in distances:
    print(row)
print()

# Random restart - jalanin 10x, ambil yang terbaik
best = None
best_dist = float('inf')
for _ in range(10):
    route = hill_climbing_tsp(distances)
    dist = sum(distances[route[i]][route[(i+1)%len(route)]] for i in range(len(route)))
    if dist < best_dist:
        best, best_dist = route, dist

print("Best route:", " => ".join(str(c+1) for c in best) + " => " + str(best[0]+1))
print("Total distance:", best_dist)
