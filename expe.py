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
# 1. GRAPH LOADING UTILS
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

# =====================================================================
# 2. SOLVER CLASS (FULL ALGORITHMS)
# =====================================================================

class DoubleRomanDomination:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.adj = {u: set(self.G.neighbors(u)) for u in self.nodes}
        self.degrees = dict(graph.degree())
        self.nodes_arr = np.array(self.nodes)
        self.degrees_arr = np.array([self.degrees[u] for u in self.nodes])

        # Algorithm Parameters Defaults
        self.ACO_PARAMS = {'ants': 5, 'rho': 0.2, 'd_rate_aco': 0.7, 'd_rate': 0.9, 'd_min': 0.2, 'd_max': 0.5, 'r_aug': 0.05, 'k_max': 5, 'rvns_max_itr': 150, 'max_noimpr': 10}
        self.PSO_PARAMS = {'w_start': 0.9, 'w_end': 0.4, 'c1': 2.0, 'c2': 2.0, 'v_max': 0.5}

    # --- Core Helpers ---
    def calculate_weight(self, solution): 
        return int(sum(solution.values()))

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

    # --- Heuristics & VNS Ops ---
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
    def _vns_reduce_op(self, S):
        nodes = sorted(self.nodes, key=lambda x: self.degrees[x], reverse=True)
        for u in nodes:
            if S[u] in [2,3]:
                orig=S[u]; S[u]=0
                if not self.is_feasible(S): S[u]=2; 
                if not self.is_feasible(S): S[u]=orig
        return S
    def _vns_shake_op(self, S, k):
        S_new = S.copy(); doms=[u for u in self.nodes if S_new[u] in [2,3]]
        if doms:
            targets = random.sample(doms, min(len(doms), k))
            for u in targets: S_new[u]=0
            S_new = self.feasibility_repair(S_new)
        return S_new

    # ================= ALGORITHMS (IMPLEMENTATION) =================

    # 1. GENETIC ALGORITHM (GA)
    def run_ga(self, iterations=100, pop_size=50):
        t_start = time.time(); time_to_best = 0
        if self.n == 0: return 0, [], 0
        
        # Init Pop
        pop = [self._h1() for _ in range(int(pop_size*0.4))] + [self._h2() for _ in range(int(pop_size*0.4))]
        while len(pop) < pop_size: pop.append(self._h3())
        
        best = min(pop, key=self.calculate_weight)
        best_w = self.calculate_weight(best)
        history = [best_w]

        for _ in range(iterations):
            p1 = min(random.sample(pop, 3), key=self.calculate_weight)
            p2 = min(random.sample(pop, 3), key=self.calculate_weight) # Tournament
            
            # Crossover
            child = p1.copy()
            if self.n > 1:
                idx1, idx2 = sorted(random.sample(range(self.n), 2))
                for i in range(idx1, idx2): child[self.nodes[i]] = p2[self.nodes[i]]
            
            # Mutation
            if random.random() < 0.1: 
                u = random.choice(self.nodes); child[u] = random.choice([0, 2, 3])
            
            child = self.feasibility_repair(child)
            pop[random.randint(0, pop_size-1)] = child
            
            w_child = self.calculate_weight(child)
            if w_child < best_w:
                best = child.copy(); best_w = w_child
                time_to_best = time.time() - t_start
            
            history.append(int(best_w))
            
        return int(best_w), history, time_to_best

    # 2. ANT COLONY OPTIMIZATION (ACO)
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

    def run_aco(self, iterations=20, ants=5):
        t_start = time.time(); time_to_best = 0
        if self.n==0: return 0, [], 0
        pheromones={u:0.5 for u in self.nodes}
        best_sol=None; best_w=float('inf'); history = []
        
        self.ACO_PARAMS['ants'] = ants # Update param

        for iteration in range(1, iterations+1):
            iter_best_sol=None; iter_best_w=float('inf')
            for _ in range(self.ACO_PARAMS['ants']):
                S = self._construct_aco(pheromones); S = self._extend_aco(S, pheromones)
                S = self._reduce_solution(S); S = self._rvns_aco(S, pheromones)
                w = self.calculate_weight(S)
                if w < iter_best_w: iter_best_w=w; iter_best_sol=S.copy()
            
            if iter_best_w < best_w: 
                best_w=iter_best_w; best_sol=iter_best_sol.copy()
                time_to_best = time.time() - t_start
            
            # Pheromone Update
            K_curr = 1.0/(iter_best_w+1e-9); K_best = 1.0/(best_w+1e-9)
            for u in self.nodes:
                val_curr=1.0 if iter_best_sol[u]==3 else 0.0
                val_best=1.0 if best_sol[u]==3 else 0.0
                upd = (K_curr*val_curr + K_best*val_best)/(K_curr+K_best)
                pheromones[u] = (1-self.ACO_PARAMS['rho'])*pheromones[u] + self.ACO_PARAMS['rho']*upd
            
            history.append(int(best_w))
        return int(best_w), history, time_to_best

    # 3. VARIABLE NEIGHBORHOOD SEARCH (VNS)
    def run_vns(self, max_iter=300):
        t_start = time.time(); time_to_best = 0
        if self.n==0: return 0, [], 0
        S = self.feasibility_repair({u:0 for u in self.nodes}); V=set(self.nodes)
        while V: u=max(list(V), key=lambda x:self.degrees[x]); S[u]=3; V-=(self.adj[u]|{u})
        curr = self._vns_reduce_op(S); best=curr.copy(); best_w=self.calculate_weight(best)
        k=1; history = [best_w]
        
        for _ in range(max_iter):
            S_new = self._vns_shake_op(curr, k); local = self._vns_reduce_op(S_new); w=self.calculate_weight(local)
            if w < best_w: 
                best_w=w; best=local.copy(); curr=local.copy(); k=1
                time_to_best = time.time() - t_start
            else: k = 1 if k > 10 else k+1
            history.append(int(best_w))
        return int(best_w), history, time_to_best

    # 4. PARTICLE SWARM OPTIMIZATION (PSO)
    def run_pso(self, iterations=100, swarm_size=30):
        t_start = time.time(); time_to_best = 0
        if self.n==0: return 0, [], 0
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
            
            improved = False
            for i in range(swarm_size):
                S = decode(X[i]); cost = self.calculate_weight(S)
                if cost < P_best_val[i]: P_best_val[i]=cost; P_best[i]=X[i].copy()
                if cost < G_best_val: 
                    G_best_val=cost; G_best_pos=X[i].copy(); improved=True
            
            if improved: time_to_best = time.time() - t_start
            history.append(int(G_best_val))
            
        return int(G_best_val), history, time_to_best

    # 5. HYBRID (GA + VNS)
    def run_ga_vns_hybrid(self, iterations=100, pop_size=50, vns_depth=10):
        t_start = time.time(); time_to_best = 0
        if self.n == 0: return 0, [], 0
        
        pop = [self._h1() for _ in range(int(pop_size*0.4))] + [self._h2() for _ in range(int(pop_size*0.4))]
        while len(pop) < pop_size: pop.append(self._h3())
        
        best = min(pop, key=self.calculate_weight)
        best_w = self.calculate_weight(best)
        history = [best_w]
        
        for _ in range(iterations):
            p1 = min(random.sample(pop, 3), key=self.calculate_weight)
            p2 = min(random.sample(pop, 3), key=self.calculate_weight)
            
            child = p1.copy()
            if self.n > 1:
                idx1, idx2 = sorted(random.sample(range(self.n), 2))
                for i in range(idx1, idx2): child[self.nodes[i]] = p2[self.nodes[i]]
            
            if random.random() < 0.1: u = random.choice(self.nodes); child[u] = random.choice([0, 2, 3])
            
            child = self.feasibility_repair(child); child = self._vns_reduce_op(child)
            
            # Short VNS
            curr_child = child.copy(); curr_w = self.calculate_weight(curr_child); k=1
            for _ in range(vns_depth):
                shaken = self._vns_shake_op(curr_child, k); local = self._vns_reduce_op(shaken); w_local = self.calculate_weight(local)
                if w_local < curr_w: curr_child = local; curr_w = w_local; k=1
                else: k = 1 if k > 3 else k+1
            child = curr_child
            
            pop[random.randint(0, pop_size-1)] = child
            w_child = self.calculate_weight(child)
            if w_child < best_w:
                best = child.copy(); best_w = w_child
                time_to_best = time.time() - t_start
                
            history.append(int(best_w))
        return int(best_w), history, time_to_best

# =====================================================================
# MAIN EXPERIMENT EXECUTION
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--output_file", type=str, default="experiment_results_full.json")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per graph")
    
    # Hyperparameters from CLI
    parser.add_argument("--ga_iter", type=int, default=100)
    parser.add_argument("--ga_pop", type=int, default=50)
    parser.add_argument("--aco_iter", type=int, default=30)
    parser.add_argument("--aco_ants", type=int, default=10)
    parser.add_argument("--vns_iter", type=int, default=300)
    parser.add_argument("--pso_iter", type=int, default=100)
    parser.add_argument("--pso_swarm", type=int, default=30)
    parser.add_argument("--hybrid_iter", type=int, default=50)
    args = parser.parse_args()

    # 1. Store Config for Visualization
    config_info = {
        "GA":     f"Iter: {args.ga_iter}\nPop: {args.ga_pop}\nMut: 0.1",
        "ACO":    f"Iter: {args.aco_iter}\nAnts: {args.aco_ants}\nRho: 0.2",
        "VNS":    f"Iter: {args.vns_iter}\nK_max: 5",
        "PSO":    f"Iter: {args.pso_iter}\nSwarm: {args.pso_swarm}",
        "Hybrid": f"Iter: {args.hybrid_iter}\nPop: {args.ga_pop}\nVNS Depth: 10"
    }

    # 2. Get Files
    files = sorted(glob.glob(os.path.join(args.data_folder, "*.mtx")))
    if not files:
        print("[Info] No .mtx files found. Generating DEMO random graphs.")
        files = ["DEMO_RANDOM_1", "DEMO_RANDOM_2"]
    
    experiment_data = []

    print("\nBENCHMARK: GA vs ACO vs VNS vs PSO vs Hybrid")
    print("=" * 115)
    print(f"{'Graph':<15} | {'GA(Avg)':<8} | {'ACO(Avg)':<8} | {'VNS(Avg)':<8} | {'PSO(Avg)':<8} | {'Hyb(Avg)':<8} | {'Time(All)':<8}")
    print("=" * 115)

    for f in files:
        if "DEMO" in f:
            G = nx.erdos_renyi_graph(80, 0.15); name = f
        else:
            G = load_graph_from_mtx(f); name = os.path.basename(f).replace('.mtx','')
        
        if not G: continue
        
        solver = DoubleRomanDomination(G)
        
        # Data structure for this graph
        graph_results = {
            "graph_name": name,
            "config": config_info,
            "results": { "GA": [], "ACO": [], "VNS": [], "PSO": [], "Hybrid": [] }
        }
        
        avg_costs = {}
        t_start_all = time.time()
        
        # Execution Map
        algo_map = {
            "GA": lambda: solver.run_ga(iterations=args.ga_iter, pop_size=args.ga_pop),
            "ACO": lambda: solver.run_aco(iterations=args.aco_iter, ants=args.aco_ants),
            "VNS": lambda: solver.run_vns(max_iter=args.vns_iter),
            "PSO": lambda: solver.run_pso(iterations=args.pso_iter, swarm_size=args.pso_swarm),
            "Hybrid": lambda: solver.run_ga_vns_hybrid(iterations=args.hybrid_iter, pop_size=args.ga_pop)
        }

        # Run Trials
        for algo_name, func in algo_map.items():
            costs_sum = 0
            for i in range(args.trials):
                t0 = time.time()
                cost, hist, ttb = func()
                total_t = time.time() - t0
                
                graph_results["results"][algo_name].append({
                    "cost": cost,
                    "total_time": total_t,
                    "time_to_best": ttb,
                    "history": hist
                })
                costs_sum += cost
            
            avg_costs[algo_name] = costs_sum / args.trials

        t_total = time.time() - t_start_all
        experiment_data.append(graph_results)

        # Print Table Row
        print(f"{name:<15} | {avg_costs['GA']:<8.1f} | {avg_costs['ACO']:<8.1f} | {avg_costs['VNS']:<8.1f} | {avg_costs['PSO']:<8.1f} | {avg_costs['Hybrid']:<8.1f} | {t_total:.2f}s")

    print("=" * 115)

    # Save JSON
    with open(args.output_file, 'w') as outfile:
        json.dump(experiment_data, outfile, indent=4)
    
    print(f"\n[Done] Results saved to '{args.output_file}'. Upload to Colab for visualization.")