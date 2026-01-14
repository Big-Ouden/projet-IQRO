import itertools

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


def test_solve_classical():
    """Tests du solveur classique avec différentes instances"""
    print("=" * 60)
    print("TESTS SOLVEUR CLASSIQUE")
    print("=" * 60)
    
    # Instance 1: Petit exemple (2 variables, 2 contraintes)
    print("\n[Instance 1] n=2, m=2")
    entry1 = MaxXorSat(2, 2, [[1, 1], [0, 1]], [0, 1])
    solution1 = solve(entry1)
    print(f"  Solution: {solution1[0]}")
    print(f"  Utilité: {solution1[1]}/{entry1.n}")
    
    # Instance 2: Plus de contraintes (3 variables, 4 contraintes)
    print("\n[Instance 2] n=3, m=4")
    entry2 = MaxXorSat(3, 4, 
                       [[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 1]], 
                       [1, 0, 1])
    solution2 = solve(entry2)
    print(f"  Solution: {solution2[0]}")
    print(f"  Utilité: {solution2[1]}/{entry2.n}")
    
    # Instance 3: Système inconsistant
    print("\n[Instance 3] Système avec contraintes conflictuelles")
    entry3 = MaxXorSat(3, 2, 
                       [[1, 0],
                        [1, 0],
                        [0, 1]], 
                       [0, 1, 1])
    solution3 = solve(entry3)
    print(f"  Solution: {solution3[0]}")
    print(f"  Utilité: {solution3[1]}/{entry3.n}")
    
    return [(entry1, solution1), (entry2, solution2), (entry3, solution3)]


# Exécuter les tests classiques
if __name__ == "__main__":
    classical_results = test_solve_classical()


"""
===========================================
=========== QAOA pour MaxXorSat ===========
===========================================
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
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
    for i in range(m): # Attention: itérer sur m (nombre de contraintes), pas n
        # Initialisation de la chaîne avec l'identité partout
        pauli_string = ['I'] * n
        
        # Le coefficient dépend de b[i] selon la formule du rapport Q7
        # Si b[i] = 0, on veut (1 - PROD Z)/2 -> coeff -0.5 pour le terme Z
        # Si b[i] = 1, on veut (1 + PROD Z)/2 -> coeff +0.5 pour le terme Z
        coefficient = 0.0
        
        if b[i] == 0:
            coefficient = -0.5
        else:
            coefficient = 0.5

        # Construction de la chaîne de Pauli pour l'interaction
        for j in range(n):
            if A[i][j] == 1:
                pauli_string[j] = 'Z' # On applique Z si la variable est dans l'équation
        
        # Ajout du terme d'interaction
        # On inverse la chaîne car Qiskit utilise l'ordre little-endian (q0 à droite)
        pauli_list.append(("".join(reversed(pauli_string)), coefficient))
        
        # OPTIONNEL MAIS RECOMMANDÉ : Ajout du terme constant (0.5 * I)
        # Sans cela, l'optimiseur trouvera la bonne solution (le bon état), 
        # mais la valeur de l'énergie (cost) ne sera pas égale au nombre d'erreurs.
        # Avec ce terme, Cost = 0 signifie "toutes les contraintes satisfaites".
        full_identity = "I" * n
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
        solution binaire, meilleurs paramètres, coût optimal, et utilité
    """
    q, hamiltonian = build_quantum_circuit(entry, reps=reps)

    # Initialiser les paramètres aléatoirement entre 0 et 2π
    init_params = np.random.uniform(0, 2 * np.pi, q.num_parameters)

    # évaluation
    def f(params):
        pub = (q, hamiltonian, params)
        estimator = StatevectorEstimator()
        job = estimator.run([pub])
        result = job.result()

        # Accès aux valeurs d'expectation via l'attribut evs du DataBin
        # evs est un scalaire quand on passe un seul observable
        cost = result[0].data.evs  # type: ignore
        return cost

    # Opti
    result = minimize(f, init_params, method="COBYLA")
    best_params = result.x
    best_cost = result.fun

    # Extraire la solution binaire en échantillonnant le circuit optimisé
    final_circuit = q.assign_parameters(best_params)
    
    # Ajouter des mesures au circuit
    measured_circuit = final_circuit.measure_all(inplace=False)
    
    # Utiliser le Sampler pour obtenir les mesures
    sampler = StatevectorSampler()
    job = sampler.run([measured_circuit], shots=1000)
    result_counts = job.result()[0].data.meas.get_counts()
    
    # Trouver la chaîne de bits la plus probable
    most_probable_bitstring = max(result_counts, key=result_counts.get)
    
    # Convertir en tableau numpy (inverser car Qiskit utilise little-endian)
    binary_solution = np.array([int(bit) for bit in reversed(most_probable_bitstring)])
    
    # Calculer l'utilité (nombre de contraintes satisfaites)
    A = np.array(entry.A)
    b = np.array(entry.b)
    res = np.dot(A, binary_solution) % 2
    utilite = np.sum(res == b)

    return binary_solution, best_params, best_cost, utilite


def test_solve_qaoa():
    """Tests du solveur QAOA avec différentes instances"""
    print("\n" + "=" * 60)
    print("TESTS SOLVEUR QAOA")
    print("=" * 60)
    
    # Instance 1: Petit exemple (2 variables, 2 contraintes)
    print("\n[Instance 1 QAOA] n=2, m=2, reps=1")
    entry1 = MaxXorSat(2, 2, [[1, 1], [0, 1]], [0, 1])
    try:
        binary_solution1, best_params1, best_cost1, utilite1 = solve_qaoa(entry1, reps=1)
        print(f"  Solution binaire: {binary_solution1}")
        print(f"  Utilité: {utilite1}/{entry1.n}")
        print(f"  Coût optimal: {best_cost1}")
        print(f"  (Paramètres: {best_params1})")
    except Exception as e:
        print(f"  Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    # Instance 2: Même instance avec plus de répétitions
    print("\n[Instance 1 QAOA] n=2, m=2, reps=2")
    try:
        binary_solution2, best_params2, best_cost2, utilite2 = solve_qaoa(entry1, reps=2)
        print(f"  Solution binaire: {binary_solution2}")
        print(f"  Utilité: {utilite2}/{entry1.n}")
        print(f"  Coût optimal: {best_cost2}")
        print(f"  (Paramètres: {best_params2})")
    except Exception as e:
        print(f"  Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    # Instance 3: Plus grande (3 variables, 3 contraintes)
    print("\n[Instance 2 QAOA] n=3, m=3, reps=1")
    entry3 = MaxXorSat(3, 3, 
                       [[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1]], 
                       [1, 0, 1])
    try:
        binary_solution3, best_params3, best_cost3, utilite3 = solve_qaoa(entry3, reps=1)
        print(f"  Solution binaire: {binary_solution3}")
        print(f"  Utilité: {utilite3}/{entry3.n}")
        print(f"  Coût optimal: {best_cost3}")
        print(f"  (Paramètres: {best_params3})")
    except Exception as e:
        print(f"  Erreur: {e}")
        import traceback
        traceback.print_exc()





# Tests QAOA et comparaisons
if __name__ == "__main__":
    # Tests QAOA
    test_solve_qaoa()
    
