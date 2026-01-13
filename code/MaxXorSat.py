import numpy as np
import itertools

class MaxXorSat:

    def __init__(self, n  , m , A, b):
        try:
            # A taille nxm
            if (len(A) != n):
                raise ValueError("A n'a pas n lignes")
            if (len(A[0]) != m):
                raise ValueError("A n'a pas m colonnes")


            # b taille nx1
            if(len(b) != n):
                raise ValueError("b n'a pas n lignes")

            self.n = n
            self.m = m
            self.A = A
            self.b = b
        except Exception as e:
            print("Erreur dans la création de l'instance MaxXorSat:", e)
            exit(0)



#retourne la solution qui maximise et utilité = nombre d'égalité vérifié
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


# Test
entry = MaxXorSat(2,2,[[1,1],[0,1]], [0])
solution = solve(entry)
print("solution: " , solution[0])
print("utlilité : " , solution[1])


"""
===========================================
=========== QAOA pour MaxXorSat ===========
===========================================
"""


"""
on construit le circuit quantique a partir de l'hamiltonien et la liste de porte I,Z etc défini et de entry 
a la question 7 - 8 (il faut utiliser SparsePauliOp.from_list([("IZZ", 1), ("IZI", 2)]), q.num_parameters)
Rappelez vous que QAOA construit un circuit paramétré, il faut, pour le faire fonction-
ner donner une valeur à ses paramètres. Il faut ensuite trouver le bon set de paramètres
pour que ce circuit vous donne la meilleure solution. Vous devez donc optimiser les
paramètres du circuit avec un algorithme classique. Vous pouvez utiliser un optimiseur
déjà conçu pour ça.
Commencez par récupérer le nombre de paramètres du circuit:
q.num_parameters
Fixez ensuite ces paramèters à des valeurs aléatoires, par exemple entre 0 et 2π.
Il nous faut une fonction capable d’évaluer un set de paramètres params. On utilisera
la fonction suivante (à adapter en fonction de la version de qiskit que vous utilisez):


from qiskit.primitives import StatevectorEstimator
def f(params):
    #On suppose qu’on a deux variables globales.
    #q est le circuit paramétré renvoyé par QAOAAnsatz
    #hamiltonian est l’hamiltonien généré avec la fonction SparsePauliOp.from_list
    pub = [q, [hamiltonian], [params]]
    estimator = StatevectorEstimator()
    result = estimator.run(pubs=[pub]).result()
    cost = result[0].data.evs[0]
    
    return cost


Ce circuit évalue la qualité des paramètres params au travers de l’hamiltonien. Pour
simplifier grossièrement, la fonction estimator.run effectue les opérations suivantes:
– Instancier les paramètres du circuit q avec les valeurs données dans la liste params
– Exécuter le circuit q de nombreuses fois pour produire une distribution de qbits
– Tester chaque vecteur de cette distribution sur l’hamiltonien pour déterminer la
valeur propre moyenne sur cette distribution. Plus la valeur propre moyenne est
petite, plus le circuit a de grandes chances de sortir une solution réalisable de bonne
qualité.
Plus d’informaton dans le premier TP sur qiskit.
On peut ensuite utiliser ensuite la fonction de scipy nommée minimize pour trouver
un bon set de paramètres.
best_params = minimize(f, init_params, args=(), method="COBYLA").x
Enfin, on peut rappeler f avec ces paramètres pour avoir la valeur objective finale
ou mesurer manuellement la distribution finale et extraire la meilleure solution. Pour
instancier manuellement les paramètres, il faut utiliser:
q.assign_parameters

"""


from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize


def build_hamiltonian(entry: MaxXorSat):
    """
    Construit l'hamiltonien pour le problème MaxXorSat.
    Pour chaque contrainte i: A[i] @ x = b[i] (mod 2)
    On veut maximiser le nombre de contraintes satisfaites.
    """
    n = entry.n  
    m = entry.m 
    A = entry.A
    b = entry.b
    
    pauli_list = []
    # Pas certain de la suite, ça dépend de la q8 et des portes à mettre
    # TODO: Mettre les bonnes portes
    """    
    # Pour chaque contrainte
    for i in range(n):
        pauli_string = ['PORTE QUELCONQUE 1'] * m
        coefficient = 1.0
        
        for j in range(m):
            if A[i][j] == 1:
                pauli_string[j] = 'PORTE QUELCONQUE 2'
        
        if b[i] == 1:
            coefficient = -1.0
        """
        
        # littéralement faire SparsePauliOp.from_list([("IZZ", 1), ("IZI", 2)]), à chaque bouvle on détermine ("IZZ", 1)
        # TODO: décommenter
        # pauli_list.append((''.join(reversed(pauli_string)), coefficient))
    
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
        Circuit final, meilleurs paramètres, et coût optimal
    """
    q, hamiltonian = build_quantum_circuit(entry, reps=reps)
    
    # Initialiser les paramètres aléatoirement entre 0 et 2π
    init_params = np.random.uniform(0, 2*np.pi, q.num_parameters)
    
    # évaluation
    def f(params):
        pub = [q, [hamiltonian], [params]]
        estimator = StatevectorEstimator()
        result = estimator.run(pubs=[pub]).result()
        cost = result[0].data.evs[0]
        return cost
    
    # Opti
    result = minimize(f, init_params, method="COBYLA")
    best_params = result.x
    best_cost = result.fun
    
    final_circuit = q.assign_parameters(best_params)
    
    return final_circuit, best_params, best_cost


