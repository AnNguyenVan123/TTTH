import networkx as nx
import random
import math
import numpy as np
from scipy.io import mmread
import sys
import os
import glob

# =====================================================================
# Graph Loading
# =====================================================================

def load_graph_from_mtx(path):
    """
    Loads a graph from a .mtx file.
    [cite_start]Ensures the graph is Undirected (simple) as required by the DRDP paper[cite: 32].
    """
    try:
        M = mmread(path).tocsr()
        # Convert to unweighted (pattern) if needed
        if M.dtype not in [np.int32, np.int64, np.float32, np.float64]:
            M.data[:] = 1
            
        # FORCE UNDIRECTED: create_using=nx.Graph
        # This fixes issues where 'neighbors()' only returns outgoing edges in directed sparse matrices
        G = nx.from_scipy_sparse_array(M, create_using=nx.Graph)
        
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

        # Parametric constants
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
        # [cite_start]"""Calculates weight of DROMDF: sum of labels[cite: 22]."""
        return sum(solution.values())

    def is_feasible(self, solution):
        """
        [cite_start]Verifies if solution is a valid DROMDF[cite: 22].
        Condition: 
        - Label 0: adj to at least two 2s OR one 3.
        - Label 1: adj to at least one >= 2.
        """
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
        """
        [cite_start]Algorithm 6: Feasibility check[cite: 281].
        Repairs a child solution to make it valid: 
        If u is 0 and not dominated by 3, set u to 2.
        """
        new_sol = solution.copy()
        for u in self.nodes:
            if new_sol[u] == 0:
                # Check for label 3 domination
                if not any(new_sol[v] == 3 for v in self.G.neighbors(u)):
                    new_sol[u] = 2
        return new_sol

    # ==================== 1. Heuristics (Section 4) ====================

    def heuristic_1(self):
        """
        [cite_start]Algorithm 2: Heuristic 1[cite: 159].
        """
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)

        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            
            for v in neighbors:
                S[v] = 0
            
            V_prime -= set(neighbors) | {u}

            if len(V_prime) == 1:
                last = list(V_prime)[0]
                S[last] = 2
                V_prime.remove(last)
        return S

    def heuristic_2(self):
        """
        [cite_start]Algorithm 3: Heuristic 2[cite: 166].
        """
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)

        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors:
                S[v] = 0
            V_prime -= set(neighbors) | {u}

            # Check for isolated vertices in remaining graph
            to_remove = []
            for v in list(V_prime):
                deg_in_prime = sum(1 for n in self.G.neighbors(v) if n in V_prime)
                if deg_in_prime == 0:
                    S[v] = 2
                    to_remove.append(v)
            
            for v in to_remove:
                V_prime.remove(v)
        return S

    def heuristic_3(self):
        """
        [cite_start]Algorithm 4: Heuristic 3[cite: 225].
        """
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)

        while V_prime:
            candidates = sorted(list(V_prime), key=lambda x: self.G.degree[x], reverse=True)
            u = candidates[0] 
            
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors:
                S[v] = 0
            V_prime -= set(neighbors) | {u}

            to_remove = []
            for v in list(V_prime):
                deg_in_prime = sum(1 for n in self.G.neighbors(v) if n in V_prime)
                if deg_in_prime == 0:
                    S[v] = 2
                    to_remove.append(v)
            for v in to_remove:
                V_prime.remove(v)
        return S

    # ==================== 2. Genetic Algorithm (Section 3) ====================

    def tournament_selection(self, population, k=3):
        # [cite_start]"""Selects parent using Tournament[cite: 268]."""
        candidates = random.sample(population, k)
        return min(candidates, key=self.calculate_weight)

    def roulette_wheel_selection(self, population):
        """
        [cite_start]Selects parent using Roulette Wheel[cite: 268].
        Probability ~ 1 / Cost.
        """
        costs = [self.calculate_weight(s) for s in population]
        # FIX: Added epsilon to prevent DivisionByZero if weight is 0
        fitnesses = [1.0 / (c + 1e-9) for c in costs]
        total_fit = sum(fitnesses)
        
        pick = random.uniform(0, total_fit)
        current = 0
        for s, f in zip(population, fitnesses):
            current += f
            if current > pick:
                return s
        return population[-1]

    def ga_crossover(self, S1, S2):
        # [cite_start]"""Algorithm 5: Crossover[cite: 254]."""
        if self.n < 2: return S1 
        idx1, idx2 = sorted(random.sample(range(self.n), 2))
        node_order = self.nodes 

        child1 = S1.copy()
        child2 = S2.copy()

        for i in range(idx1, idx2):
            u = node_order[i]
            child1[u], child2[u] = child2[u], child1[u]

        child1 = self.feasibility_repair(child1)
        child2 = self.feasibility_repair(child2)

        w1 = self.calculate_weight(child1)
        w2 = self.calculate_weight(child2)
        return child1 if w1 < w2 else child2

    def run_genetic_algorithm(self, iterations=100, pop_size=100):
        # [cite_start]"""Algorithm 1: GA[cite: 89]."""
        if self.n == 0: return {}, 0

        # [cite_start]Init population with 40/40/20 heuristic split [cite: 415]
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
            
        return best_sol, self.calculate_weight(best_sol)

    # ==================== 3. ACO Algorithm (Section 6) ====================

    def choose_vertex(self, V_prime, pheromones, is_construct_phase):
        # [cite_start]"""Vertex selection[cite: 335, 337]."""
        rate = self.ACO_PARAMS['d_rate_aco'] if is_construct_phase else self.ACO_PARAMS['d_rate']
        r = random.random()
        
        if not V_prime: return None
        
        candidates = list(V_prime)
        # [cite_start]f(u) = deg(u) * tau_u [cite: 334]
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
                if current > pick:
                    return u
            return candidates[-1]

    def construct_solution(self, pheromones):
        # [cite_start]"""Algorithm 8: Construct Solution[cite: 342]."""
        S = {u: 0 for u in self.nodes} 
        V_prime = set(self.nodes)
        while V_prime:
            u = self.choose_vertex(V_prime, pheromones, is_construct_phase=True)
            S[u] = 3
            neighbors = list(self.G.neighbors(u))
            for v in neighbors:
                S[v] = 0
            V_prime -= set(neighbors) | {u}
        return S

    def extend_solution(self, S, pheromones):
        # [cite_start]"""Algorithm 9: Extend Solution[cite: 352]."""
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
        # [cite_start]"""Algorithm 10: Reduce Solution[cite: 376]."""
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
        # [cite_start]"""Algorithm 11: Destroy Solution[cite: 357]."""
        d_min = self.ACO_PARAMS['d_min']
        d_max = self.ACO_PARAMS['d_max']
        k_max = self.ACO_PARAMS['k_max']
        
        # [cite_start]# [cite: 379] Eq 4
        d = d_min + (k - 1) * ((d_max - d_min) / (k_max - 1))
        itr = int(len(self.nodes) * d)
        V_prime = list(self.nodes)
        
        while itr > 0 and V_prime:
            u = random.choice(V_prime)
            if S[u] == 0 or S[u] == 2:
                S[u] = -1 # Unlabelled
                itr -= 1 
            V_prime.remove(u)
        return S

    def ran_var_neigh_srch(self, S, pheromones):
        # [cite_start]"""Algorithm 12: Random Variable Neighborhood Search[cite: 400]."""
        S_curr = S.copy()
        best_w = self.calculate_weight(S_curr)
        
        k = 1
        c_noimpr = 0
        max_itr = self.ACO_PARAMS['rvns_max_itr']
        max_noimpr = self.ACO_PARAMS['max_noimpr']
        k_max = self.ACO_PARAMS['k_max']

        while c_noimpr < max_noimpr and max_itr > 0:
            S_prime = S_curr.copy()
            S_prime = self.destroy_solution(S_prime, k)
            
            unlabeled = {u for u in self.nodes if S_prime[u] == -1}
            while unlabeled:
                u = self.choose_vertex(unlabeled, pheromones, is_construct_phase=False)
                S_prime[u] = 3
                for v in self.G.neighbors(u):
                    if S_prime.get(v) == -1: 
                        S_prime[v] = 0
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
            
            if k > k_max: k = 1
            max_itr -= 1
            
        return S_curr

    def compute_convergence(self, pheromones):
        # [cite_start]"""Eq 1: Convergence Factor[cite: 328]."""
        T_max = 0.999
        T_min = 0.001
        
        numerator = 0
        for u in self.nodes:
            tau = pheromones[u]
            numerator += max(T_max - tau, tau - T_min)
        
        denominator = self.n * (T_max + T_min)
        if denominator == 0: return 0 
        
        conv_fact = 2 * (numerator / denominator) - 1
        return conv_fact

    def update_pheromone(self, pheromones, curr_best_sol, best_sol):
        # [cite_start]"""Eq 3: Pheromone Update[cite: 348]."""
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

    def run_aco(self, iterations=20):
        # [cite_start]"""Algorithm 7: ACO Main Loop[cite: 308]."""
        if self.n == 0: return {}, 0
            
        pheromones = {u: 0.5 for u in self.nodes}
        best_sol = None
        best_weight = float("inf")

        for it in range(iterations):
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

        return best_sol, best_weight

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Double Roman Domination Solver (Batch Processing)")
    parser.add_argument("--data_folder", type=str, default="data", help="Folder containing .mtx files")
    parser.add_argument("--ga_iter", type=int, default=50, help="GA Iterations")
    parser.add_argument("--ga_pop", type=int, default=100, help="GA Population Size")
    parser.add_argument("--aco_iter", type=int, default=20, help="ACO Iterations")

    args = parser.parse_args()

    search_path = os.path.join(args.data_folder, "*.mtx")
    mtx_files = sorted(glob.glob(search_path))

    if not mtx_files:
        print(f"No .mtx files found in directory: {args.data_folder}")
        sys.exit(1)

    print(f"Found {len(mtx_files)} graphs to process in '{args.data_folder}'\n")
    print(f"{'Graph Name':<25} | {'Nodes':<6} | {'Edges':<7} | {'GA Best':<10} | {'ACO Best':<10} | {'Winner':<6}")
    print("-" * 75)

    results = []

    for file_path in mtx_files:
        file_name = os.path.basename(file_path)
        
        G = load_graph_from_mtx(file_path)
        if G is None: continue
            
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        
        if nodes == 0:
            print(f"{file_name:<25} | {nodes:<6} | {edges:<7} | {'Skipped':<10} | {'Skipped':<10} | {'N/A':<6}")
            continue

        solver = DoubleRomanDomination(G)

        # Run Algorithms
        try:
            _, ga_weight = solver.run_genetic_algorithm(iterations=args.ga_iter, pop_size=args.ga_pop)
            _, aco_weight = solver.run_aco(iterations=args.aco_iter)
        except Exception as e:
            print(f"{file_name:<25} | Error during execution: {e}")
            continue

        winner = "Tie"
        if aco_weight < ga_weight: winner = "ACO"
        elif ga_weight < aco_weight: winner = "GA"

        print(f"{file_name:<25} | {nodes:<6} | {edges:<7} | {ga_weight:<10} | {aco_weight:<10} | {winner:<6}")
        
        results.append({"name": file_name, "ga": ga_weight, "aco": aco_weight})

    print("\n" + "="*40 + "\n Final Summary\n" + "="*40)
    print(f"Total Graphs: {len(results)}")
    print(f"ACO Wins:     {sum(1 for r in results if r['aco'] < r['ga'])}")
    print(f"GA Wins:      {sum(1 for r in results if r['ga'] < r['aco'])}")
    print(f"Ties:         {sum(1 for r in results if r['ga'] == r['aco'])}")
    print("="*40)