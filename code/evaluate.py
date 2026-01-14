import time
import numpy as np
from qiskit_aer.primitives import SamplerV2
from basic_solver import MaxXorSat, solve
from qaoa_solver import solve_qaoa
from grover_solver import build_realizable_solutions, build_grover_circuit, determine_opti_k


def evaluate_performance(entry: MaxXorSat):
    print(f"\nÉvaluation pour instance n={entry.n}, m={entry.m}")
    print(build_realizable_solutions(entry, entry.n))

    
    # --- 1. Exact ---
    start_exact = time.time()
    sol_exact, util_exact = solve(entry)
    time_exact = time.time() - start_exact
    print(f"[Exact] Temps: {time_exact:.4f}s | Utilité Max: {util_exact} | Solution: {sol_exact}")
    
    # --- 2. QAOA ---
    reps_qaoa = 1
    start_qaoa = time.time()
    try:
        final_qc_qaoa, best_params, best_cost = solve_qaoa(entry, reps=reps_qaoa)
        
        # Mesure
        measured_qc_qaoa = final_qc_qaoa.copy()
        measured_qc_qaoa.measure_all()
        
        sampler = SamplerV2()
        job = sampler.run([measured_qc_qaoa])
        result = job.result()
        
        # Récupération des comptes de mesure
        counts = result[0].data.meas.get_counts()
        
        # Trouver l'état binaire le plus fréquent
        best_bitstring_qaoa = max(counts, key=counts.get)
        
        sol_qaoa = np.array([int(c) for c in (best_bitstring_qaoa)])

        
        res = np.dot(entry.A, sol_qaoa) % 2
        util_qaoa = np.sum(res == entry.b)
        
        time_qaoa = time.time() - start_qaoa
        
        depth_qaoa = final_qc_qaoa.depth()
        gates_qaoa = sum(final_qc_qaoa.count_ops().values())
        
        print(f"[QAOA] Temps: {time_qaoa:.4f}s | Utilité: {util_qaoa}  | Profondeur: {depth_qaoa} | Portes: {gates_qaoa} | solution: {sol_qaoa}")

    except Exception as e:
        print(f"[QAOA] Erreur: {e}")

    # --- 3. Grover ---
    k_grover = determine_opti_k(entry)
    iterations_grover = int(np.pi/4 * np.sqrt(2**entry.m))
    
    start_grover = time.time()
    try:
        qc_grover = build_grover_circuit(entry, k=k_grover, iterations=iterations_grover)
        
        sampler = SamplerV2()
        job = sampler.run([qc_grover])
        result = job.result()
        
        counts = result[0].data.meas.get_counts()
        best_bitstring_grover = max(counts, key=counts.get)
        
        sol_grover = np.array([int(c) for c in (best_bitstring_grover)])
        
        # Inverser pour correspondre à l'ordre
        sol_grover = sol_grover[::-1]

        
        if len(sol_grover) == entry.m:
            res = np.dot(entry.A, sol_grover) % 2
            util_grover = np.sum(res == entry.b)
        else:
            print(f"[Grover] Dimension mismatch: solution length {len(sol_grover)} != m {entry.m}")
            util_grover = "N/A"
        
        time_grover = time.time() - start_grover

        depth_grover = qc_grover.depth()
        gates_grover = sum(qc_grover.count_ops().values())
        qubits_grover = qc_grover.num_qubits

        print(f"[Grover] Temps: {time_grover:.4f}s | Utilité: {util_grover} | Profondeur: {depth_grover} | Portes: {gates_grover} | Itérations: {iterations_grover} | solution: {sol_grover}")
        
    except Exception as e:
        print(f"[Grover] Erreur: {e}")



"""
===========================================
=================== Main ==================
===========================================
"""

if __name__ == "__main__":
    # Instance de test
    print("Lancement de l'évaluation...")
    entry = MaxXorSat(3, 4, 
                       [[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 1]], 
                       [1, 0, 1])
    evaluate_performance(entry)

    # Testons sur plus d'instances et plus grosses
    entry2 = MaxXorSat(5, 5,
                        [[1, 0, 1, 1, 0],
                         [0, 1, 1, 0, 1],
                         [1, 1, 0, 1, 0],
                         [0, 0, 1, 1, 1],
                         [1, 0, 0, 1, 1]],
                        [1, 0, 1, 1, 0])
    evaluate_performance(entry2)

    entry3 = MaxXorSat(4, 6,
                        [[1, 1, 0, 1, 0, 0],
                            [0, 1, 1, 0, 1, 0], 
                            [1, 0, 1, 0, 0, 1],
                            [0, 0, 1, 1, 1, 0]],
                        [1, 0, 1, 1])
    evaluate_performance(entry3)
