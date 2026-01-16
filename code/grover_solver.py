import itertools
import numpy as np
from qiskit import QuantumCircuit
from basic_solver import MaxXorSat

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


if __name__ == "__main__":
    # Exemple d'utilisation
    A = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 0]])
    b = np.array([1, 0, 1])
    entry = MaxXorSat(n=3, m=3, A=A, b=b)
    
    k_optimal = determine_opti_k(entry)
    print(f"Valeur k optimale: {k_optimal}")
    
    grover_circuit = build_grover_circuit(entry, k_optimal, iterations=1)
    print(grover_circuit)