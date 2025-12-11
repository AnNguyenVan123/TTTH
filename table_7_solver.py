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
        """Algorithm 6: Feasibility check and repair."""
        new_sol = solution.copy()
        for u in self.nodes:
            if new_sol[u] == 0:
                if not any(new_sol[v] == 3 for v in self.G.neighbors(u)):
                    new_sol[u] = 2
        return new_sol

    # ==================== 1. Heuristics (Section 4) ====================

    def heuristic_1(self):
        """Algorithm 2: Random selection."""
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
        """Algorithm 3: Random selection + Isolated handling."""
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
        """
        Algorithm 4: Greedy Degree selection.
        UPDATED: Random tie-breaking to ensure population diversity.
        """
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)

        while V_prime:
            # Calculate degrees in the original graph (static degree) or induced?
            # Paper says "sort V' ... degrees of G". We assume static degree in G.
            # Tie breaking: "breaking ties arbitrarily" 
            
            candidates = list(V_prime)
            if not candidates: break
            
            # Find max degree among candidates
            max_deg = -1
            best_candidates = []
            
            for v in candidates:
                d = self.G.degree[v]
                if d > max_deg:
                    max_deg = d
                    best_candidates = [v]
                elif d == max_deg:
                    best_candidates.append(v)
            
            # Randomly pick one of the best to break ties
            u = random.choice(best_candidates)
            
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

    # ==================== 2. Genetic Algorithm ====================

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

        child1 = S1.copy()
        child2 = S2.copy()

        for i in range(idx1, idx2):
            u = node_order[i]
            child1[u], child2[u] = child2[u], child1[u]

        child1 = self.feasibility_repair(child1)
        child2 = self.feasibility_repair(child2)

        return child1 if self.calculate_weight(child1) < self.calculate_weight(child2) else child2

    def run_genetic_algorithm(self, iterations=100, pop_size=100, mode='combined'):
        """
        Runs GA with specific initialization mode.
        modes: 'h1', 'h2', 'h3', 'combined'
        """
        if self.n == 0: return {}, 0

        population = []
        
        # --- Initialization Strategy ---
        if mode == 'h1':
            # 100% Heuristic 1
            for _ in range(pop_size): population.append(self.heuristic_1())
            
        elif mode == 'h2':
            # 100% Heuristic 2
            for _ in range(pop_size): population.append(self.heuristic_2())
            
        elif mode == 'h3':
            # 100% Heuristic 3
            for _ in range(pop_size): population.append(self.heuristic_3())
            
        else: # 'combined'
            # 40% H1, 40% H2, 20% H3 
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

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reproduce Table 7: Heuristic Comparison")
    parser.add_argument("--data_folder", type=str, default="data", help="Folder containing .mtx files")
    # Table 7 used 100,000 iterations[cite: 413]. Default lowered for speed.
    parser.add_argument("--ga_iter", type=int, default=100, help="GA Iterations (Paper used 100,000)")
    parser.add_argument("--ga_pop", type=int, default=100, help="GA Population Size")

    args = parser.parse_args()

    search_path = os.path.join(args.data_folder, "*.mtx")
    mtx_files = sorted(glob.glob(search_path))

    if not mtx_files:
        print(f"No .mtx files found in directory: {args.data_folder}")
        sys.exit(1)

    # Print Table Header similar to Table 7 [cite: 463]
    # "Graphs | LB | UB | Heuristic 1 | Heuristic 2 | Heuristic 3 | Combined"
    print("\nREPRODUCING TABLE 7 RESULTS")
    print(f"Settings: Iterations={args.ga_iter}, Population={args.ga_pop}")
    print("-" * 110)
    print(f"{'Graphs':<20} | {'Nodes':<6} | {'Edges':<7} | {'H1':<8} | {'H2':<8} | {'H3':<8} | {'Combined':<8}")
    print("-" * 110)

    for file_path in mtx_files:
        file_name = os.path.basename(file_path)
        
        G = load_graph_from_mtx(file_path)
        if G is None: continue
            
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        
        if nodes == 0:
            continue

        solver = DoubleRomanDomination(G)

        try:
            # 1. Heuristic 1 Only
            _, h1_score = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h1')
            
            # 2. Heuristic 2 Only
            _, h2_score = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h2')
            
            # 3. Heuristic 3 Only
            _, h3_score = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h3')
            
            # 4. Combined
            _, comb_score = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='combined')

            print(f"{file_name:<20} | {nodes:<6} | {edges:<7} | {h1_score:<8} | {h2_score:<8} | {h3_score:<8} | {comb_score:<8}")
            
        except Exception as e:
            print(f"{file_name:<20} | Error: {e}")

    print("-" * 110)