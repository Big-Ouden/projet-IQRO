import itertools
import time
import numpy as np


class MaxXorSat:
    def __init__(self, n, m, A, b):
        try:
            # A taille nxm
            if len(A) != n:
                raise ValueError("A n'a pas n lignes")
            if len(A[0]) != m:
                raise ValueError("A n'a pas m colonnes")

            # b taille nx1
            if len(b) != n:
                raise ValueError("b n'a pas n lignes")

            self.n = n
            self.m = m
            self.A = A
            self.b = b
        except Exception as e:
            print("Erreur dans la création de l'instance MaxXorSat:", e)
            exit(0)


# retourne la solution qui maximise et utilité = nombre d'égalité vérifié
def solve(entry: MaxXorSat):
    """
    Solveur Basique naif par énumération

    Args:
        entry: Instance de MaxXorSat

    Returns:
        (best_solution, max_utilite)

    """
    A = np.array(entry.A)
    b = np.array(entry.b)
    n = entry.n
    m = entry.m
    max_utilite = 0
    best_solution = None

    for bits in itertools.product([0, 1], repeat=m):
        bits = np.array(bits)

        # print(bits)
        res = np.dot(A, bits) % 2
        utilite = np.sum(res == b)
        if utilite > max_utilite:
            max_utilite = utilite
            best_solution = bits

    return (best_solution, max_utilite)




"""
===========================================
=========== QAOA pour MaxXorSat ===========
===========================================
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def build_hamiltonian(entry: MaxXorSat):
    """
    Construit l'hamiltonien pour le problème MaxXorSat.
    Pour chaque contrainte i: A[i] @ x = b[i] (mod 2)
    On veut maximiser le nombre de contraintes satisfaites (donc minimiser les non-satisfaites).
    """
    n = entry.n
    m = entry.m
    A = entry.A
    b = entry.b

    pauli_list = []

    # Pour chaque contrainte (ligne de la matrice A)
    for i in range(n): # itérer sur n (nombre de contraintes)
        # Initialisation de la chaîne avec l'identité partout
        pauli_string = ['I'] * m # m = nombre de qubits / variables
        
        # Le coefficient dépend de b[i] selon la formule du rapport Q7
        # Si b[i] = 0, on veut (1 - PROD Z)/2 -> coeff -0.5 pour le terme Z
        # Si b[i] = 1, on veut (1 + PROD Z)/2 -> coeff +0.5 pour le terme Z
        coefficient = 0.0
        
        if b[i] == 0:
            coefficient = -0.5
        else:
            coefficient = 0.5

        # Construction de la chaîne de Pauli pour l'interaction
        for j in range(m):
            if A[i][j] == 1:
                pauli_string[j] = 'Z' # On applique Z si la variable est dans l'équation
        
        # Ajout du terme d'interaction
        # On inverse la chaîne car Qiskit utilise l'ordre little-endian (q0 à droite)
        pauli_list.append(("".join(reversed(pauli_string)), coefficient))
        
        # OPTIONNEL MAIS RECOMMANDÉ : Ajout du terme constant (0.5 * I)
        # Sans cela, l'optimiseur trouvera la bonne solution (le bon état), 
        # mais la valeur de l'énergie (cost) ne sera pas égale au nombre d'erreurs.
        # Avec ce terme, Cost = 0 signifie "toutes les contraintes satisfaites".
        full_identity = "I" * m
        pauli_list.append((full_identity, 0.5))

    return SparsePauliOp.from_list(pauli_list)


def build_quantum_circuit(entry: MaxXorSat, reps=1):
    """
    Construit le circuit QAOA pour résoudre MaxXorSat.

    Args:
        entry: Instance de MaxXorSat
        reps: Nombre de répétitions QAOA (profondeur du circuit)

    Returns:
        Circuit QAOA paramétré et hamiltonien
    """
    hamiltonian = build_hamiltonian(entry)
    qc = QAOAAnsatz(hamiltonian, reps=reps)
    return qc, hamiltonian


def solve_qaoa(entry: MaxXorSat, reps=1):
    """
    Résout le problème MaxXorSat avec QAOA.

    Args:
        entry: Instance de MaxXorSat
        reps: Nombre de répétitions QAOA (profondeur du circuit), 1 par défaut

    Returns:
        circuit final, meilleurs paramètres, coût optimal
    """
    q, hamiltonian = build_quantum_circuit(entry, reps=reps)

    # Initialiser les paramètres aléatoirement entre 0 et 2π
    init_params = np.random.uniform(0, 2 * np.pi, q.num_parameters)

    # évaluation
    estimator = Estimator()
    def f(params):
        job = estimator.run(circuits=[q], observables=[hamiltonian], parameter_values=[params])
        result = job.result()
        cost = result.values[0]
        return cost

    # Opti
    result = minimize(f, init_params, method="COBYLA")
    best_params = result.x
    best_cost = result.fun

    final_circuit = q.assign_parameters(best_params)

    return final_circuit, best_params, best_cost





"""
===========================================
=========== Grover pour MaxXorSat =========
===========================================
"""

"""
Implementer ensuite une fonction qui commence par construire L
la liste de toutes les solutions réalisables dont le poids est supérieur
à k. Utiliser ensuite cette fonction pour construire un oracle.
"""
def build_realizable_solutions(entry: MaxXorSat, k: int):
    """
    Construit la liste L des solutions dont l'utilité est >= k.
    """
    n = entry.n
    m = entry.m
    A = entry.A
    b = entry.b

    realizable_solutions = []
    
    for bits in itertools.product([0, 1], repeat=m):
        bits_arr = np.array(bits)
        # Ax = b mod 2
        res = np.dot(A, bits_arr) % 2
        utilite = np.sum(res == b)
        
        if utilite >= k:
            realizable_solutions.append(bits_arr)
            
    return realizable_solutions

def build_grover_circuit(entry: MaxXorSat, k: int, iterations: int):
    """
    Construit le circuit Grover "triche" (basé sur la liste L).
    """
    m = entry.m
    realizable_solutions = build_realizable_solutions(entry, k)

    qc = QuantumCircuit(m)

    qc.h(range(m))

    for _ in range(iterations):
        
        # --- ORACLE (Question 13 & 14) ---
        # Marque les états présents dans la liste realizable_solutions
        for solution in realizable_solutions:
            # A. Préparation (X sur les 0)
            for i in range(m):
                if solution[i] == 0:
                    qc.x(i)
            
            # B. Inversion de phase (MCZ)
            # Implémentation H-MCX-H est valide.
            qc.h(m - 1)
            qc.mcx(list(range(m - 1)), m - 1) 
            qc.h(m - 1)
            
            # C. Restauration (X sur les 0)
            for i in range(m):
                if solution[i] == 0:
                    qc.x(i)

        # --- DIFFUSEUR (Standard) ---
        qc.h(range(m))
        qc.x(range(m))
        
        # MCZ du diffuseur (2|0><0| - I)
        qc.h(m - 1)
        qc.mcx(list(range(m - 1)), m - 1)
        qc.h(m - 1)
        
        qc.x(range(m))
        qc.h(range(m))

    qc.measure_all()
    
    return qc

"""
L'algorithme d'optimisation a pour objectif de renvoyer une liste de $x_i$ qui satisfassent un maximum de clauses tandis que le problème de décision renvoie seulement si le nombre de clauses satisfaites est supérieure où égale à une constante K donné.
\\

On pourrait donc obtenir l'optimisation en cherchant le K maximum en utilisant le problème de décision de manière dichotomique. Puis une fois K trouvé, on applique Grover pour obtenir cette solution:
"""    
def determine_opti_k(entry: MaxXorSat):
    """
    Détermine une valeur k optimale pour Grover. Par dichotomie.
    """
    # Dichotomie sur k entre 0 et n (nb max de contraintes satisfaisables)
    low = 0
    high = entry.n
    best_k = 0
    
    while low <= high:
        mid = (low + high) // 2
        # On vérifie s'il existe au moins une solution avec utilité >= mid
        solutions = build_realizable_solutions(entry, mid)
        if len(solutions) > 0:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return best_k


"""
===========================================
=========== Évaluation des méthodes =======
===========================================
"""


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
        
        sampler = Sampler()
        job = sampler.run(circuits=[measured_qc_qaoa])
        result = job.result()
        
        quasi_dists = result.quasi_dists[0]
        # Trouver la quasi-distribution maximale (état le plus probable)
        best_int_qaoa = max(quasi_dists, key=quasi_dists.get)
        best_bitstring_qaoa = format(best_int_qaoa, f'0{entry.m}b')
        
        sol_qaoa = np.array([int(c) for c in reversed(best_bitstring_qaoa)])
        
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
        
        sampler = Sampler()
        job = sampler.run(circuits=[qc_grover])
        result = job.result()
        
        quasi_dists = result.quasi_dists[0]
        # Trouver la quasi-distribution maximale (état le plus probable)
        best_int_grover = max(quasi_dists, key=quasi_dists.get)
        best_bitstring_grover = format(best_int_grover, f'0{entry.m}b')
        
        sol_grover = np.array([int(c) for c in reversed(best_bitstring_grover)])
        
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
