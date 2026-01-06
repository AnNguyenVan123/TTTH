import networkx as nx
import random
import numpy as np
from scipy.io import mmread
import sys
import os
import glob
import time
import argparse
import json

# =====================================================================
# 1. Graph Loading & Solver (GIỮ NGUYÊN LOGIC)
# =====================================================================

def load_graph_from_mtx(path):
    try:
        M = mmread(path)
        if hasattr(M, 'tocsr'): M = M.tocsr()
        if M.dtype not in [np.int32, np.int64, np.float32, np.float64]: M.data[:] = 1
        rows, cols = M.shape
        if rows == cols:
            G = nx.from_scipy_sparse_array(M, create_using=nx.Graph)
        else:
            G = nx.bipartite.from_biadjacency_matrix(M); G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except: return None

class DoubleRomanDomination:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.adj = {u: set(self.G.neighbors(u)) for u in self.nodes}
        self.degrees = dict(graph.degree())
        self.nodes_arr = np.array(self.nodes)
        self.degrees_arr = np.array([self.degrees[u] for u in self.nodes])
        # Params
        self.ACO_PARAMS = {'ants': 5, 'rho': 0.2, 'd_rate_aco': 0.7, 'd_rate': 0.9, 'd_min': 0.2, 'd_max': 0.5, 'r_aug': 0.05, 'k_max': 5, 'rvns_max_itr': 150, 'max_noimpr': 10}
        self.PSO_PARAMS = {'w_start': 0.9, 'w_end': 0.4, 'c1': 2.0, 'c2': 2.0, 'v_max': 0.5}

    def calculate_weight(self, solution): return int(sum(solution.values()))

    def is_feasible(self, solution):
        for u in self.nodes:
            label = solution.get(u, 0)
            if label == 0:
                has_3 = False; count_2 = 0
                for v in self.adj[u]:
                    l = solution.get(v, 0)
                    if l == 3: has_3 = True; break
                    if l == 2: count_2 += 1
                if not has_3 and count_2 < 2: return False
            elif label == 1:
                if not any(solution.get(v, 0) >= 2 for v in self.adj[u]): return False
        return True

    def feasibility_repair(self, solution):
        new_sol = solution.copy()
        for u in self.nodes:
            if new_sol[u] == 0:
                if not any(new_sol[v] == 3 for v in self.adj[u]): new_sol[u] = 2
        return new_sol

    def _reduce_solution(self, S):
        sorted_nodes = sorted(self.nodes, key=lambda u: (self.degrees[u], u))
        for u in sorted_nodes:
            if S[u] in [2, 3]:
                init_lab = S[u]; S[u] = 0
                if not self.is_feasible(S):
                    S[u] = 2
                    if not self.is_feasible(S): S[u] = init_lab
        return S

    # Heuristics
    def _h1(self):
        S={u:0 for u in self.nodes}; V=set(self.nodes)
        while V: 
            u=random.choice(list(V)); S[u]=3; V-=(self.adj[u]|{u})
            if len(V)==1: S[list(V)[0]]=2; V.clear()
        return S
    def _h2(self):
        S={u:0 for u in self.nodes}; V=set(self.nodes)
        while V:
            u=random.choice(list(V)); S[u]=3; V-=(self.adj[u]|{u})
            for v in [x for x in V if not (self.adj[x]&V)]: S[v]=2; V.remove(v)
        return S
    def _h3(self):
        S={u:0 for u in self.nodes}; V=set(self.nodes)
        while V:
            cand=list(V); u=max(cand, key=lambda x: self.degrees[x])
            S[u]=3; V-=(self.adj[u]|{u})
            for v in [x for x in V if not (self.adj[x]&V)]: S[v]=2; V.remove(v)
        return S

    # VNS Ops
    def _vns_reduce_op(self, S):
        nodes = sorted(self.nodes, key=lambda x: self.degrees[x], reverse=True)
        for u in nodes:
            if S[u] in [2,3]:
                orig=S[u]; S[u]=0
                if not self.is_feasible(S):
                    S[u]=2
                    if not self.is_feasible(S): S[u]=orig
        return S
    def _vns_shake_op(self, S, k):
        S_new = S.copy()
        doms=[u for u in self.nodes if S_new[u] in [2,3]]
        if doms:
            targets = random.sample(doms, min(len(doms), k))
            for u in targets: S_new[u]=0
            S_new = self.feasibility_repair(S_new)
        return S_new

    # --- ALGORITHMS (Trả về: Cost, History) ---
    def run_ga(self, iterations=100, pop_size=50):
        if self.n == 0: return 0, []
        pop = [self._h1() for _ in range(int(pop_size*0.4))] + [self._h2() for _ in range(int(pop_size*0.4))]
        while len(pop) < pop_size: pop.append(self._h3())
        best = min(pop, key=self.calculate_weight)
        history = []
        for _ in range(iterations):
            p1 = min(random.sample(pop, 3), key=self.calculate_weight)
            costs = [self.calculate_weight(s) for s in pop]
            inv = [1.0/(c+1e-9) for c in costs]
            r = random.uniform(0, sum(inv)); curr=0; p2=pop[-1]
            for i,v in enumerate(inv):
                curr+=v; 
                if curr>r: p2=pop[i]; break
            child = p1.copy()
            if self.n > 1:
                idx1, idx2 = sorted(random.sample(range(self.n), 2))
                c2_ref = p2.copy()
                for i in range(idx1, idx2): child[self.nodes[i]] = c2_ref[self.nodes[i]]
            if random.random() < 0.1:
                u = random.choice(self.nodes); child[u] = random.choice([0, 2, 3])
            child = self.feasibility_repair(child)
            pop[random.randint(0, pop_size-1)] = child
            if self.calculate_weight(child) < self.calculate_weight(best): best = child.copy()
            history.append(int(self.calculate_weight(best)))
        return int(self.calculate_weight(best)), history

    def run_aco(self, iterations=20):
        if self.n==0: return 0, []
        pheromones={u:0.5 for u in self.nodes}
        best_sol=None; best_w=float('inf'); history = []
        for iteration in range(1, iterations+1):
            iter_best_sol=None; iter_best_w=float('inf')
            for _ in range(self.ACO_PARAMS['ants']):
                S = self._construct_aco(pheromones); S = self._extend_aco(S, pheromones)
                S = self._reduce_solution(S); S = self._rvns_aco(S, pheromones)
                w = self.calculate_weight(S)
                if w < iter_best_w: iter_best_w=w; iter_best_sol=S.copy()
            if iter_best_w < best_w: best_w=iter_best_w; best_sol=iter_best_sol.copy()
            K_curr = 1.0/(iter_best_w+1e-9); K_best = 1.0/(best_w+1e-9)
            for u in self.nodes:
                val_curr=1.0 if iter_best_sol[u]==3 else 0.0
                val_best=1.0 if best_sol[u]==3 else 0.0
                upd = (K_curr*val_curr + K_best*val_best)/(K_curr+K_best)
                pheromones[u] = (1-self.ACO_PARAMS['rho'])*pheromones[u] + self.ACO_PARAMS['rho']*upd
            history.append(int(best_w))
        return int(best_w), history

    def _select_vertex_aco(self, candidates, pheromones, is_construct):
        threshold = self.ACO_PARAMS['d_rate_aco'] if is_construct else self.ACO_PARAMS['d_rate']
        cand_list = list(candidates)
        weights = [self.degrees[u] * pheromones.get(u, 0.5) for u in cand_list]
        if random.random() <= threshold: return cand_list[np.argmax(weights)]
        total_w = sum(weights)
        if total_w == 0: return random.choice(cand_list)
        pick = random.uniform(0, total_w); curr=0
        for u, w in zip(cand_list, weights):
            curr+=w; 
            if curr>pick: return u
        return cand_list[-1]
    def _construct_aco(self, pheromones):
        S={u:0 for u in self.nodes}; V=set(self.nodes)
        while V: u=self._select_vertex_aco(V, pheromones, True); S[u]=3; V-=(self.adj[u]|{u})
        return S
    def _extend_aco(self, S, pheromones):
        V02=[u for u in self.nodes if S[u] in [0,2]]; 
        if not V02: return S
        limit=int(self.ACO_PARAMS['r_aug']*len(V02)); curr_V02=set(V02)
        while limit>0 and curr_V02: u=self._select_vertex_aco(curr_V02, pheromones, False); S[u]=3; curr_V02.remove(u); limit-=1
        return S
    def _rvns_aco(self, S, pheromones):
        S_prime=S.copy(); k=1; c_noimpr=0; max_itr=self.ACO_PARAMS['rvns_max_itr']
        d_min, d_max, k_max = self.ACO_PARAMS['d_min'], self.ACO_PARAMS['d_max'], self.ACO_PARAMS['k_max']
        while c_noimpr < self.ACO_PARAMS['max_noimpr'] and max_itr > 0:
            d = d_min + (k-1)*((d_max-d_min)/(k_max-1)) if k_max>1 else d_min
            num_rm = int(self.n * d)
            S_temp = S_prime.copy()
            for u in random.sample(self.nodes, min(num_rm, self.n)):
                if S_temp[u] in [0,2]: S_temp[u]=0
            uncovered = set()
            for u in self.nodes:
                if S_temp[u]==0 and not any(S_temp[v]==3 for v in self.adj[u]): uncovered.add(u)
            while uncovered: u=self._select_vertex_aco(uncovered, pheromones, True); S_temp[u]=3; uncovered-=(self.adj[u]|{u})
            S_temp = self._extend_aco(S_temp, pheromones); S_temp = self._reduce_solution(S_temp)
            if self.calculate_weight(S_temp) < self.calculate_weight(S_prime): S_prime=S_temp; k=1; c_noimpr=0
            else: k+=1; c_noimpr+=1
            if k>k_max: k=1
            max_itr-=1
        return S_prime

    def run_vns(self, max_iter=300):
        if self.n==0: return 0, []
        S = self.feasibility_repair({u:0 for u in self.nodes}); V=set(self.nodes)
        while V: u=max(list(V), key=lambda x:self.degrees[x]); S[u]=3; V-=(self.adj[u]|{u})
        for v in [x for x in V if not (self.adj[x]&V)]: S[v]=2; V.remove(v)
        curr = self._vns_reduce_op(S); best=curr.copy(); best_w=self.calculate_weight(best)
        k=1; history = []
        for _ in range(max_iter):
            S_new = self._vns_shake_op(curr, k); local = self._vns_reduce_op(S_new); w=self.calculate_weight(local)
            if w < best_w: best_w=w; best=local.copy(); curr=local.copy(); k=1
            else: k = 1 if k > 10 else k+1
            history.append(int(best_w))
        return int(best_w), history

    def run_pso(self, iterations=100, swarm_size=30):
        if self.n==0: return 0, []
        X = np.random.rand(swarm_size, self.n); V = np.random.uniform(-0.1, 0.1, (swarm_size, self.n))
        P_best = X.copy(); P_best_val = np.full(swarm_size, np.inf)
        G_best_pos = X[0].copy(); G_best_val = np.inf
        history = []
        def decode(p_vec):
            scores = self.degrees_arr * p_vec; sorted_idx = np.argsort(-scores)
            S={u:0 for u in self.nodes}; covered=set()
            for idx in sorted_idx:
                u=self.nodes[idx]
                if u not in covered: S[u]=3; covered.add(u); covered.update(self.adj[u])
            return self.feasibility_repair(S)
        for it in range(iterations):
            w = self.PSO_PARAMS['w_start'] - (self.PSO_PARAMS['w_start']-self.PSO_PARAMS['w_end'])*(it/iterations)
            r1, r2 = np.random.rand(swarm_size, self.n), np.random.rand(swarm_size, self.n)
            V = w*V + self.PSO_PARAMS['c1']*r1*(P_best-X) + self.PSO_PARAMS['c2']*r2*(G_best_pos-X)
            np.clip(V, -0.5, 0.5, out=V); X+=V; np.clip(X, 0.01, 1.0, out=X)
            for i in range(swarm_size):
                S = decode(X[i]); cost = self.calculate_weight(S)
                if cost < P_best_val[i]: P_best_val[i]=cost; P_best[i]=X[i].copy()
                if cost < G_best_val: G_best_val=cost; G_best_pos=X[i].copy()
            history.append(int(G_best_val))
        return int(G_best_val), history

    def run_ga_vns_hybrid(self, iterations=100, pop_size=50, vns_depth=10):
        if self.n == 0: return 0, []
        pop = [self._h1() for _ in range(int(pop_size*0.4))] + [self._h2() for _ in range(int(pop_size*0.4))]
        while len(pop) < pop_size: pop.append(self._h3())
        best = min(pop, key=self.calculate_weight); history = []
        for _ in range(iterations):
            p1 = min(random.sample(pop, 3), key=self.calculate_weight)
            costs = [self.calculate_weight(s) for s in pop]
            inv = [1.0/(c+1e-9) for c in costs]
            r = random.uniform(0, sum(inv)); curr=0; p2=pop[-1]
            for i,v in enumerate(inv): curr+=v; 
            if curr>r: p2=pop[i]; break
            child = p1.copy()
            if self.n > 1:
                idx1, idx2 = sorted(random.sample(range(self.n), 2))
                c2_ref = p2.copy(); 
                for i in range(idx1, idx2): child[self.nodes[i]] = c2_ref[self.nodes[i]]
            if random.random() < 0.1: u = random.choice(self.nodes); child[u] = random.choice([0, 2, 3])
            child = self.feasibility_repair(child); child = self._vns_reduce_op(child)
            
            curr_child = child.copy(); curr_w = self.calculate_weight(curr_child); k=1
            for _ in range(vns_depth):
                shaken = self._vns_shake_op(curr_child, k); local = self._vns_reduce_op(shaken); w_local = self.calculate_weight(local)
                if w_local < curr_w: curr_child = local; curr_w = w_local; k=1
                else: k = 1 if k > 3 else k+1
            child = curr_child
            
            pop[random.randint(0, pop_size-1)] = child
            if self.calculate_weight(child) < self.calculate_weight(best): best = child.copy()
            history.append(int(self.calculate_weight(best)))
        return int(self.calculate_weight(best)), history

# =====================================================================
# MAIN: RUN EXPERIMENT + PRINT TABLE + SAVE JSON
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--output_file", type=str, default="experiment_results.json")
    args = parser.parse_args()

    # Tìm file mtx hoặc tạo Demo
    files = sorted(glob.glob(os.path.join(args.data_folder, "*.mtx")))
    if not files:
        print("No .mtx files found. Using DEMO Random Graph...")
        files = ["DEMO_RANDOM"]
    
    experiment_data = []

    # --- IN HEADER BẢNG SO SÁNH (NHƯ BẠN YÊU CẦU) ---
    print("\nBENCHMARK: GA vs ACO vs VNS vs PSO vs GA+VNS (Hybrid)")
    print("=" * 115)
    print(f"{'Graph':<20} | {'GA':<6} | {'ACO':<6} | {'VNS':<6} | {'PSO':<6} | {'Hybrid':<6} | {'Time(s)':<8}")
    print("=" * 115)

    for f in files:
        if f == "DEMO_RANDOM":
            G = nx.erdos_renyi_graph(100, 0.15); name = "Random_Graph_100"
        else:
            G = load_graph_from_mtx(f); name = os.path.basename(f).replace('.mtx','')
        
        if not G: continue
        
        solver = DoubleRomanDomination(G)
        graph_result = {
            "graph_name": name,
            "nodes": G.number_of_nodes(),
            "algorithms": {}
        }
        
        t_start_all = time.time()

        # 1. GA
        t0=time.time(); ga_val, ga_hist = solver.run_ga(iterations=100)
        graph_result["algorithms"]["GA"] = {"cost": ga_val, "time": time.time()-t0, "history": ga_hist}

        # 2. ACO
        t0=time.time(); aco_val, aco_hist = solver.run_aco(iterations=30)
        graph_result["algorithms"]["ACO"] = {"cost": aco_val, "time": time.time()-t0, "history": aco_hist}

        # 3. VNS
        t0=time.time(); vns_val, vns_hist = solver.run_vns(max_iter=300)
        graph_result["algorithms"]["VNS"] = {"cost": vns_val, "time": time.time()-t0, "history": vns_hist}

        # 4. PSO
        t0=time.time(); pso_val, pso_hist = solver.run_pso(iterations=100)
        graph_result["algorithms"]["PSO"] = {"cost": pso_val, "time": time.time()-t0, "history": pso_hist}

        # 5. Hybrid
        t0=time.time(); hyb_val, hyb_hist = solver.run_ga_vns_hybrid(iterations=50)
        graph_result["algorithms"]["Hybrid"] = {"cost": hyb_val, "time": time.time()-t0, "history": hyb_hist}

        t_total = time.time() - t_start_all
        experiment_data.append(graph_result)

        # --- IN KẾT QUẢ DÒNG (NHƯ BẠN YÊU CẦU) ---
        print(f"{name:<20} | {ga_val:<6} | {aco_val:<6} | {vns_val:<6} | {pso_val:<6} | {hyb_val:<6} | {t_total:.2f}")

    print("=" * 115)

    # LƯU FILE JSON
    with open(args.output_file, 'w') as outfile:
        json.dump(experiment_data, outfile, indent=4)
    
    print(f"\n[Done] Results saved to '{args.output_file}'.")