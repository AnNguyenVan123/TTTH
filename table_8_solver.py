import networkx as nx
import random
import math
import numpy as np
from scipy.io import mmread
import sys
import os
import glob
import time

# =====================================================================
# Graph Loading (Robust)
# =====================================================================

def load_graph_from_mtx(path):
    """
    Loads a graph from a .mtx file.
    - If Square: Loads as standard Adjacency Matrix.
    - If Rectangular: Loads as Biadjacency Matrix (Bipartite Graph).
    Ensures the graph is Undirected (simple) as required by the DRDP paper.
    """
    try:
        M = mmread(path)
        if hasattr(M, 'tocsr'):
            M = M.tocsr()
            
        # Standardize data to 1 (unweighted)
        if M.dtype not in [np.int32, np.int64, np.float32, np.float64]:
            M.data[:] = 1
            
        rows, cols = M.shape

        if rows == cols:
            # Square -> Adjacency Matrix
            G = nx.from_scipy_sparse_array(M, create_using=nx.Graph)
        else:
            # Rectangular -> Biadjacency Matrix (Bipartite Graph)
            # This handles ash958 (958x292) -> 1250 nodes
            G = nx.bipartite.from_biadjacency_matrix(M)
            # Note: NetworkX bipartite creates nodes 0..rows-1 and rows..rows+cols-1
            # We convert to simple Graph to remove bipartite attribute constraints for the solver
            G = nx.Graph(G)
        
        # Remove self-loops (simple graph requirement)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        return G
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# =====================================================================
# Double Roman Domination Solver
# =====================================================================

class DoubleRomanDomination:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)

        # ACO Parameters [cite: 440]
        self.ACO_PARAMS = {
            'num_constructions': 5,      
            'rho': 0.2,                  
            'd_rate': 0.9,               
            'd_rate_aco': 0.7,           
            'd_min': 0.2,                
            'd_max': 0.5,                
            'r_aug': 0.05,               
            'k_max': 5,                  
            'rvns_max_itr': 150,         
            'max_noimpr': 10             
        }

    def calculate_weight(self, solution):
        return sum(solution.values())

    def is_feasible(self, solution):
        for u in self.nodes:
            label = solution.get(u, 0)
            neighbors = [solution.get(v, 0) for v in self.G.neighbors(u)]

            if label == 0:
                has_3 = any(l == 3 for l in neighbors)
                count_2 = sum(1 for l in neighbors if l == 2)
                if not has_3 and count_2 < 2:
                    return False
            elif label == 1:
                if not any(l >= 2 for l in neighbors):
                    return False
        return True

    def feasibility_repair(self, solution):
        """Algorithm 6: Feasibility check."""
        new_sol = solution.copy()
        for u in self.nodes:
            if new_sol[u] == 0:
                if not any(new_sol[v] == 3 for v in self.G.neighbors(u)):
                    new_sol[u] = 2
        return new_sol

    # ==================== Heuristics ====================

    def heuristic_1(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors: S[v] = 0
            V_prime -= set(neighbors) | {u}
            if len(V_prime) == 1:
                last = list(V_prime)[0]
                S[last] = 2
                V_prime.remove(last)
        return S

    def heuristic_2(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors: S[v] = 0
            V_prime -= set(neighbors) | {u}
            to_remove = []
            for v in list(V_prime):
                deg_in_prime = sum(1 for n in self.G.neighbors(v) if n in V_prime)
                if deg_in_prime == 0:
                    S[v] = 2
                    to_remove.append(v)
            for v in to_remove: V_prime.remove(v)
        return S

    def heuristic_3(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            candidates = list(V_prime)
            if not candidates: break
            max_deg = -1
            best_candidates = []
            for v in candidates:
                d = self.G.degree[v] # Static degree approximation
                if d > max_deg:
                    max_deg = d
                    best_candidates = [v]
                elif d == max_deg:
                    best_candidates.append(v)
            u = random.choice(best_candidates) # Random tie-break
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors: S[v] = 0
            V_prime -= set(neighbors) | {u}
            to_remove = []
            for v in list(V_prime):
                deg_in_prime = sum(1 for n in self.G.neighbors(v) if n in V_prime)
                if deg_in_prime == 0:
                    S[v] = 2
                    to_remove.append(v)
            for v in to_remove: V_prime.remove(v)
        return S

    # ==================== GA ====================

    def tournament_selection(self, population, k=3):
        candidates = random.sample(population, k)
        return min(candidates, key=self.calculate_weight)

    def roulette_wheel_selection(self, population):
        costs = [self.calculate_weight(s) for s in population]
        fitnesses = [1.0 / (c + 1e-9) for c in costs]
        total_fit = sum(fitnesses)
        pick = random.uniform(0, total_fit)
        current = 0
        for s, f in zip(population, fitnesses):
            current += f
            if current > pick: return s
        return population[-1]

    def ga_crossover(self, S1, S2):
        if self.n < 2: return S1
        idx1, idx2 = sorted(random.sample(range(self.n), 2))
        node_order = self.nodes
        child1, child2 = S1.copy(), S2.copy()
        for i in range(idx1, idx2):
            u = node_order[i]
            child1[u], child2[u] = child2[u], child1[u]
        child1 = self.feasibility_repair(child1)
        child2 = self.feasibility_repair(child2)
        return child1 if self.calculate_weight(child1) < self.calculate_weight(child2) else child2

    def run_genetic_algorithm(self, iterations=100, pop_size=100):
        """Runs Combined GA (40/40/20 split)."""
        if self.n == 0: return 0
        population = []
        count_h1 = int(pop_size * 0.4)
        count_h2 = int(pop_size * 0.4)
        for _ in range(count_h1): population.append(self.heuristic_1())
        for _ in range(count_h2): population.append(self.heuristic_2())
        while len(population) < pop_size: population.append(self.heuristic_3())

        curr_best_sol = min(population, key=self.calculate_weight)
        best_sol = curr_best_sol.copy()
        
        for it in range(iterations):
            new_pop = []
            while len(new_pop) < pop_size:
                S1 = self.tournament_selection(population)
                S2 = self.roulette_wheel_selection(population)
                child = self.ga_crossover(S1, S2)
                new_pop.append(child)
            population = new_pop
            curr_best_sol = min(population, key=self.calculate_weight)
            if self.calculate_weight(curr_best_sol) < self.calculate_weight(best_sol):
                best_sol = curr_best_sol.copy()
        
        return self.calculate_weight(best_sol)

    # ==================== ACO ====================

    def choose_vertex(self, V_prime, pheromones, is_construct_phase):
        rate = self.ACO_PARAMS['d_rate_aco'] if is_construct_phase else self.ACO_PARAMS['d_rate']
        r = random.random()
        if not V_prime: return None
        candidates = list(V_prime)
        f_vals = {u: self.G.degree[u] * pheromones.get(u, 0.5) for u in candidates}
        
        if r <= rate:
            return max(candidates, key=lambda u: f_vals[u])
        else:
            total_f = sum(f_vals.values())
            if total_f == 0: return random.choice(candidates)
            pick = random.uniform(0, total_f)
            current = 0
            for u in candidates:
                current += f_vals[u]
                if current > pick: return u
            return candidates[-1]

    def construct_solution(self, pheromones):
        S = {u: 0 for u in self.nodes} 
        V_prime = set(self.nodes)
        while V_prime:
            u = self.choose_vertex(V_prime, pheromones, is_construct_phase=True)
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors: S[v] = 0
            V_prime -= set(neighbors) | {u}
        return S

    def extend_solution(self, S, pheromones):
        V02 = [u for u in self.nodes if S[u] in [0, 2]]
        itr = int(self.ACO_PARAMS['r_aug'] * len(V02))
        V02_set = set(V02) 
        while itr > 0 and V02_set:
            u = self.choose_vertex(V02_set, pheromones, is_construct_phase=False)
            S[u] = 3
            V02_set.remove(u)
            itr -= 1
        return S

    def reduce_solution(self, S):
        sorted_nodes = sorted(self.nodes, key=lambda x: self.G.degree[x])
        for u in sorted_nodes:
            if S[u] == 3 or S[u] == 2:
                init_lab = S[u]
                S[u] = 0
                if not self.is_feasible(S):
                    S[u] = 2
                    if not self.is_feasible(S):
                        S[u] = init_lab
        return S

    def destroy_solution(self, S, k):
        d_min = self.ACO_PARAMS['d_min']
        d_max = self.ACO_PARAMS['d_max']
        k_max = self.ACO_PARAMS['k_max']
        d = d_min + (k - 1) * ((d_max - d_min) / (k_max - 1))
        itr = int(len(self.nodes) * d)
        V_prime = list(self.nodes)
        while itr > 0 and V_prime:
            u = random.choice(V_prime)
            if S[u] == 0 or S[u] == 2:
                S[u] = -1 
                itr -= 1 
            V_prime.remove(u)
        return S

    def ran_var_neigh_srch(self, S, pheromones):
        S_curr = S.copy()
        best_w = self.calculate_weight(S_curr)
        k = 1
        c_noimpr = 0
        while c_noimpr < self.ACO_PARAMS['max_noimpr'] and self.ACO_PARAMS['rvns_max_itr'] > 0:
            S_prime = S_curr.copy()
            S_prime = self.destroy_solution(S_prime, k)
            unlabeled = {u for u in self.nodes if S_prime[u] == -1}
            while unlabeled:
                u = self.choose_vertex(unlabeled, pheromones, is_construct_phase=False)
                S_prime[u] = 3
                for v in self.G.neighbors(u):
                    if S_prime.get(v) == -1: S_prime[v] = 0
                unlabeled = {u for u in self.nodes if S_prime[u] == -1}
            S_prime = self.extend_solution(S_prime, pheromones)
            S_prime = self.reduce_solution(S_prime)
            w_prime = self.calculate_weight(S_prime)
            if w_prime < best_w:
                S_curr = S_prime
                best_w = w_prime
                k = 1
                c_noimpr = 0
            else:
                k += 1
                c_noimpr += 1
            if k > self.ACO_PARAMS['k_max']: k = 1
        return S_curr

    def update_pheromone(self, pheromones, curr_best_sol, best_sol):
        rho = self.ACO_PARAMS['rho']
        w_curr = self.calculate_weight(curr_best_sol)
        w_best = self.calculate_weight(best_sol)
        for u in self.nodes:
            in_curr = 1.0 if curr_best_sol[u] == 3 else 0.0
            in_best = 1.0 if best_sol[u] == 3 else 0.0
            deposit_num = (in_curr * w_best) + (in_best * w_curr)
            deposit_den = w_curr + w_best
            delta = deposit_num / deposit_den if deposit_den > 0 else 0
            pheromones[u] = (1 - rho) * pheromones[u] + (rho * delta)
            pheromones[u] = max(0.001, min(0.999, pheromones[u]))

    def compute_convergence(self, pheromones):
        numerator = sum(max(0.999 - pheromones[u], pheromones[u] - 0.001) for u in self.nodes)
        denominator = self.n * (0.999 + 0.001)
        return (2 * (numerator / denominator) - 1) if denominator > 0 else 0

    def run_aco_table_8(self):
        """
        Runs ACO for exactly 20 iterations.
        Returns tuple: (weight_at_1, weight_at_10, weight_at_20)
        """
        if self.n == 0: return (0, 0, 0)
        pheromones = {u: 0.5 for u in self.nodes}
        best_sol = None
        best_weight = float("inf")
        
        results = {}

        for it in range(20): # 0 to 19
            curr_best_sol = None
            curr_best_weight = float("inf")
            
            for _ in range(self.ACO_PARAMS['num_constructions']):
                S = self.construct_solution(pheromones)
                S = self.extend_solution(S, pheromones)
                S = self.reduce_solution(S)
                S = self.ran_var_neigh_srch(S, pheromones)
                w = self.calculate_weight(S)
                if w < curr_best_weight:
                    curr_best_weight = w
                    curr_best_sol = S.copy()
            
            if curr_best_weight < best_weight:
                best_weight = curr_best_weight
                best_sol = curr_best_sol.copy()
            
            self.update_pheromone(pheromones, curr_best_sol, best_sol)
            if self.compute_convergence(pheromones) > 0.99:
                pheromones = {u: 0.5 for u in self.nodes}
            
            # Record at iteration 1 (idx 0), 10 (idx 9), 20 (idx 19)
            if it == 0: results['1'] = best_weight
            if it == 9: results['10'] = best_weight
            if it == 19: results['20'] = best_weight

        return results['1'], results['10'], results['20']

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Table 8: GA vs ACO Iterations")
    parser.add_argument("--data_folder", type=str, default="data", help="Folder containing .mtx files")
    parser.add_argument("--ga_iter", type=int, default=100, help="GA Iterations (Reference benchmark)")
    parser.add_argument("--ga_pop", type=int, default=100, help="GA Population Size")
    args = parser.parse_args()

    search_path = os.path.join(args.data_folder, "*.mtx")
    mtx_files = sorted(glob.glob(search_path))

    if not mtx_files:
        print(f"No .mtx files found in {args.data_folder}")
        sys.exit(1)

    print("\nREPRODUCING TABLE 8: Comparison of GA vs ACO")
    print("-" * 90)
    print(f"{'Graphs':<20} | {'GA (Ref)':<10} | {'ACO (1)':<10} | {'ACO (10)':<10} | {'ACO (20)':<10}")
    print("-" * 90)

    for file_path in mtx_files:
        file_name = os.path.basename(file_path)
        G = load_graph_from_mtx(file_path)
        if G is None or G.number_of_nodes() == 0: continue

        solver = DoubleRomanDomination(G)
        
        try:
            # 1. Run GA (Benchmark)
            ga_result = solver.run_genetic_algorithm(iterations=args.ga_iter, pop_size=args.ga_pop)
            
            # 2. Run ACO (Track checkpoints)
            aco_1, aco_10, aco_20 = solver.run_aco_table_8()
            
            print(f"{file_name:<20} | {ga_result:<10} | {aco_1:<10} | {aco_10:<10} | {aco_20:<10}")
            
        except Exception as e:
            print(f"{file_name:<20} | Error: {e}")

    print("-" * 90)