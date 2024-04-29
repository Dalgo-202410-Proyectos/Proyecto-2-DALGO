from collections import defaultdict
import functools
import random
import time

# Variables globales
pesos = {}
total_peso = 0

# Función para calcular el costo mínimo entre dos átomos libres
@functools.lru_cache(maxsize=None)
def hallar_min_costo_atm_libres(atm1, atm2, w1, w2):
    if (atm1, atm2, "atmLibres") in pesos:
        return pesos[(atm1, atm2, "atmLibres")]

    if (atm1 > 0 and atm2 > 0) or (atm1 < 0 and atm2 < 0):
        peso = 1 + abs(abs(atm1) - abs(atm2)) % w1
    else:
        peso = w2 - abs(abs(atm1) - abs(atm2)) % w2

    pesos[(atm1, atm2, "atmLibres")] = peso
    return peso

def precompute_costs(atoms, w1, w2):
    n = len(atoms)
    cost_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if (atoms[i] > 0 and atoms[j] > 0) or (atoms[i] < 0 and atoms[j] < 0):
                cost = 1 + abs(abs(atoms[i]) - abs(atoms[j])) % w1
            else:
                cost = w2 - abs(abs(atoms[i]) - abs(atoms[j])) % w2

            cost_matrix[i][j] = cost
            cost_matrix[j][i] = cost
    return cost_matrix

def find_eulerian_path(graph):
    def find_start_vertex():
        for v in graph:
            if len(graph[v]) % 2 != 0:
                return v
        return next(iter(graph))

    def dfs(v):
        path = []
        stack = [v]
        while stack:
            u = stack[-1]
            if graph[u]:
                w = graph[u].pop()
                graph[w].remove(u)
                stack.append(w)
            else:
                path.append(stack.pop())
        return path

    odd_count = sum(len(graph[v]) % 2 != 0 for v in graph)
    if odd_count not in [0, 2]:
        return "NO SE PUEDE"

    start_vertex = find_start_vertex()
    return dfs(start_vertex)

def find_weights(path, cost_matrix, atoms_index):
    global total_peso
    total_peso = 0
    result = []

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        a_index = atoms_index[a]
        b_index = atoms_index[b]
        peso = cost_matrix[a_index][b_index]
        result.append(f"({a},{b}) [{peso}]")
        total_peso += peso

    result.append(f"Total peso: {total_peso}")
    return ", ".join(result)

def generarGrafoValido(cantidad):
    graph = defaultdict(list)
    atm_libres = []
    nodos = []
    for i in range(cantidad):
        nodo = random.randint(-1000, 1000)
        while nodo in nodos:
            nodo = random.randint(-2000, 2000)
        nodos.append(nodo)

    for i in range(cantidad):
        graph[nodos[i]].append(nodos[(i + 1) % cantidad])
        graph[nodos[(i + 1) % cantidad]].append(nodos[i])
        atm_libres.append(nodos[i])
        atm_libres.append(-nodos[i])

    atoms_index = {atom: idx for idx, atom in enumerate(set(atm_libres))}
    cost_matrix = precompute_costs(atm_libres, 3, 5)  # Ejemplo de w1 y w2

    return graph, atm_libres, cost_matrix, atoms_index

if __name__ == "__main__":
    tInicio = time.time()
    graph, atm_libres, cost_matrix, atoms_index = generarGrafoValido(100)

    path = find_eulerian_path(graph)
    if isinstance(path, list):
        print("Tengo el camino euleriano")
        camino_con_pesos = find_weights(path, cost_matrix, atoms_index)
        print(camino_con_pesos)
    else:
        print(path)  # "NO SE PUEDE"
    tFinal = time.time()
    print(f"Tiempo de ejecución: {tFinal - tInicio:.2f} segundos")
