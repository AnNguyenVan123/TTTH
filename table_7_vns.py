import networkx as nx
import random
import numpy as np
from scipy.io import mmread
import sys
import os
import glob
import time

# =====================================================================
# 1. Graph Loading
# =====================================================================

def load_graph_from_mtx(path):
    """
    Đọc file .mtx và chuyển đổi thành đồ thị NetworkX vô hướng.
    """
    try:
        M = mmread(path)
        if hasattr(M, 'tocsr'):
            M = M.tocsr()
            
        if M.dtype not in [np.int32, np.int64, np.float32, np.float64]:
            M.data[:] = 1
            
        rows, cols = M.shape

        if rows == cols:
            G = nx.from_scipy_sparse_array(M, create_using=nx.Graph)
        else:
            G = nx.bipartite.from_biadjacency_matrix(M)
            G = nx.Graph(G)
        
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# =====================================================================
# 2. Solver Class (GA + VNS)
# =====================================================================

class DoubleRomanDomination:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.degrees = dict(graph.degree())

    # --- Core Logic ---

    def calculate_weight(self, solution):
        return sum(solution.values())

    def is_feasible(self, solution):
        for u in self.nodes:
            label = solution.get(u, 0)
            neighbors = [solution.get(v, 0) for v in self.G.neighbors(u)]

            if label == 0:
                has_3 = any(l == 3 for l in neighbors)
                count_2 = sum(1 for l in neighbors if l == 2)
                if not has_3 and count_2 < 2: return False
            elif label == 1:
                if not any(l >= 2 for l in neighbors): return False
        return True

    def feasibility_repair(self, solution):
        new_sol = solution.copy()
        for u in self.nodes:
            if new_sol[u] == 0:
                if not any(new_sol[v] == 3 for v in self.G.neighbors(u)):
                    new_sol[u] = 2
        return new_sol

    # --- Heuristics ---

    def heuristic_1(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            V_prime -= set(self.G.neighbors(u)) | {u}
            if len(V_prime) == 1:
                S[list(V_prime)[0]] = 2; V_prime.clear()
        return S

    def heuristic_2(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            u = random.choice(list(V_prime))
            S[u] = 3
            V_prime -= set(self.G.neighbors(u)) | {u}
            to_remove = [v for v in V_prime if sum(1 for n in self.G.neighbors(v) if n in V_prime) == 0]
            for v in to_remove: S[v] = 2; V_prime.remove(v)
        return S

    def heuristic_3(self):
        S = {u: 0 for u in self.nodes}
        V_prime = set(self.nodes)
        while V_prime:
            cand = list(V_prime)
            if not cand: break
            max_deg = max(self.degrees[v] for v in cand)
            best_cand = [v for v in cand if self.degrees[v] == max_deg]
            u = random.choice(best_cand) 
            S[u] = 3
            V_prime -= set(self.G.neighbors(u)) | {u}
            to_remove = [v for v in V_prime if sum(1 for n in self.G.neighbors(v) if n in V_prime) == 0]
            for v in to_remove: S[v] = 2; V_prime.remove(v)
        return S

    # --- Genetic Algorithm Methods ---

    def tournament_selection(self, population, k=3):
        candidates = random.sample(population, k)
        return min(candidates, key=self.calculate_weight)

    def roulette_wheel_selection(self, population):
        costs = [self.calculate_weight(s) for s in population]
        fitnesses = [1.0 / (c + 1e-9) for c in costs]
        pick = random.uniform(0, sum(fitnesses))
        curr = 0
        for s, f in zip(population, fitnesses):
            curr += f
            if curr > pick: return s
        return population[-1]

    def ga_crossover(self, S1, S2):
        if self.n < 2: return S1
        idx1, idx2 = sorted(random.sample(range(self.n), 2))
        c1, c2 = S1.copy(), S2.copy()
        for i in range(idx1, idx2):
            u = self.nodes[i]
            c1[u], c2[u] = c2[u], c1[u]
        c1 = self.feasibility_repair(c1)
        c2 = self.feasibility_repair(c2)
        return c1 if self.calculate_weight(c1) < self.calculate_weight(c2) else c2

    def run_genetic_algorithm(self, iterations=100, pop_size=100, mode='combined'):
        if self.n == 0: return 0
        population = []
        if mode == 'h1':
            for _ in range(pop_size): population.append(self.heuristic_1())
        elif mode == 'h2':
            for _ in range(pop_size): population.append(self.heuristic_2())
        elif mode == 'h3':
            for _ in range(pop_size): population.append(self.heuristic_3())
        else: 
            count_h1 = int(pop_size * 0.4)
            count_h2 = int(pop_size * 0.4)
            for _ in range(count_h1): population.append(self.heuristic_1())
            for _ in range(count_h2): population.append(self.heuristic_2())
            while len(population) < pop_size: population.append(self.heuristic_3())

        best_sol = min(population, key=self.calculate_weight)
        for _ in range(iterations):
            new_pop = []
            while len(new_pop) < pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.roulette_wheel_selection(population)
                child = self.ga_crossover(p1, p2)
                new_pop.append(child)
            population = new_pop
            curr_best = min(population, key=self.calculate_weight)
            if self.calculate_weight(curr_best) < self.calculate_weight(best_sol):
                best_sol = curr_best.copy()
        return self.calculate_weight(best_sol)

    # --- Variable Neighborhood Search (VNS) Methods ---

    def vns_reduce(self, S):
        """Greedy Reduction (Local Search): Try to downgrade labels."""
        # Ưu tiên xét các đỉnh bậc cao trước
        sorted_nodes = sorted(self.nodes, key=lambda x: self.degrees[x], reverse=True)
        for u in sorted_nodes:
            if S[u] in [2, 3]:
                original = S[u]
                S[u] = 0
                if not self.is_feasible(S):
                    S[u] = 2 # Thử hạ xuống 2
                    if not self.is_feasible(S):
                        S[u] = original # Hoàn tác
        return S

    def vns_shake(self, S, k):
        """Perturbation: Xóa k dominator ngẫu nhiên và sửa lại."""
        S_new = S.copy()
        dominators = [u for u in self.nodes if S_new[u] in [2, 3]]
        if not dominators: return S_new
        
        # 1. Destroy
        k_actual = min(len(dominators), k)
        targets = random.sample(dominators, k_actual)
        for u in targets:
            S_new[u] = 0
            
        # 2. Repair (Greedy filling)
        # Sửa nhanh bằng feasibility_repair (đã có) hoặc logic thông minh hơn
        # Ở đây dùng logic đơn giản để đảm bảo tốc độ, vns_reduce sẽ tối ưu lại sau
        return self.feasibility_repair(S_new)

    def run_vns(self, max_iter=300, k_max=10, max_no_impr=50):
        """
        Chạy thuật toán VNS.
        Khởi tạo bằng Heuristic 3 (tốt nhất) để so sánh công bằng.
        """
        if self.n == 0: return 0
        
        # Khởi tạo
        current_sol = self.heuristic_3()
        current_sol = self.vns_reduce(current_sol)
        best_sol = current_sol.copy()
        best_weight = self.calculate_weight(best_sol)
        
        k = 1
        no_impr_count = 0
        
        for _ in range(max_iter):
            # 1. Shaking
            shaken_sol = self.vns_shake(current_sol, k)
            
            # 2. Local Search
            local_sol = self.vns_reduce(shaken_sol)
            local_weight = self.calculate_weight(local_sol)
            
            # 3. Move or Stay
            if local_weight < best_weight:
                best_sol = local_sol.copy()
                best_weight = local_weight
                current_sol = local_sol.copy()
                k = 1
                no_impr_count = 0
            else:
                k += 1
                no_impr_count += 1
                
            if k > k_max: 
                k = 1
                
            if no_impr_count >= max_no_impr:
                break
                
        return best_weight

# =====================================================================
# 3. Main Execution
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare GA Strategies vs VNS")
    parser.add_argument("--data_folder", type=str, default="data", help="Thư mục chứa file .mtx")
    parser.add_argument("--ga_iter", type=int, default=100, help="Số vòng lặp GA")
    parser.add_argument("--ga_pop", type=int, default=100, help="Kích thước quần thể GA")
    parser.add_argument("--vns_iter", type=int, default=300, help="Số vòng lặp VNS")

    args = parser.parse_args()

    search_path = os.path.join(args.data_folder, "*.mtx")
    mtx_files = sorted(glob.glob(search_path))

    if not mtx_files:
        print(f"Không tìm thấy file .mtx nào trong: {args.data_folder}")
        sys.exit(1)

    print("\nCOMPARISON: GA Strategies vs VNS")
    print(f"GA Settings:  Iter={args.ga_iter}, Pop={args.ga_pop}")
    print(f"VNS Settings: Iter={args.vns_iter}, K_max=10")
    print("-" * 115)
    # Header bảng
    print(f"{'Graph Name':<20} | {'Nodes':<5} | {'H1 Only':<8} | {'H2 Only':<8} | {'H3 Only':<8} | {'Combined':<8} | {'VNS':<8}")
    print("-" * 115)

    for file_path in mtx_files:
        file_name = os.path.basename(file_path)
        
        G = load_graph_from_mtx(file_path)
        if G is None or G.number_of_nodes() == 0:
            continue
            
        nodes = G.number_of_nodes()
        solver = DoubleRomanDomination(G)

        try:
            # 1. Chạy GA với các chế độ khởi tạo khác nhau
            res_h1 = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h1')
            res_h2 = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h2')
            res_h3 = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='h3')
            res_comb = solver.run_genetic_algorithm(args.ga_iter, args.ga_pop, mode='combined')

            # 2. Chạy VNS
            res_vns = solver.run_vns(max_iter=args.vns_iter, k_max=10, max_no_impr=50)

            # In kết quả
            print(f"{file_name:<20} | {nodes:<5} | {res_h1:<8} | {res_h2:<8} | {res_h3:<8} | {res_comb:<8} | {res_vns:<8}")
            
        except Exception as e:
            print(f"{file_name:<20} | Error: {e}")

    print("-" * 115)