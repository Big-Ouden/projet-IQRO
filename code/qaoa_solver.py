import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from basic_solver import MaxXorSat

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
    q = q.decompose().decompose() # Décomposition nécessaire pour certains backends (QAOA -> PauliEvolution -> Portes standards)

    # Initialiser les paramètres aléatoirement entre 0 et 2π
    init_params = np.random.uniform(0, 2 * np.pi, q.num_parameters)

    # évaluation
    estimator = EstimatorV2()
    def f(params):
        # Format requis pour EstimatorV2 : (circuit, observables, valeurs_parametres)
        pub = (q, hamiltonian, params)
        job = estimator.run([pub])
        result = job.result()
        # Récupération de la valeur moyenne (coût)
        cost = result[0].data.evs
        return cost

    # Opti
    result = minimize(f, init_params, method="COBYLA")
    best_params = result.x
    best_cost = result.fun

    final_circuit = q.assign_parameters(best_params)

    return final_circuit, best_params, best_cost
