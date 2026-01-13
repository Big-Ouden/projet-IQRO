import numpy as np
import itertools

class MaxXorSat:

    def __init__(self, n  , m , A, b):
        # A taille nxm
        if (len(A) != n):
            print("A n'a pas n lignes")
        if (len(A[0]) != m):
            print("A n'a pas m colonnes")


        # b taille nx1
        if(len(b) != n):
            print("b n'a pas n lignes")

        self.n = n
        self.m = m
        self.A = A
        self.b = b

#retourne la solution qui maximise et utilité = nombre d'égalité vérifié
def solve(entry: MaxXorSat):
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
entry = MaxXorSat(2,2,[[1,1],[0,1]], [0,1])
solution = solve(entry)
print("solution: " , solution[0])
print("utlilité : " , solution[1])



from qiskit import QuantumCircuit
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
def build_quantum_circuit(entry: MaxXorSat):
    n = entry.n
    m = entry.m
    A = entry.A
    b = entry.b

    qc = QuantumCircuit(m)

    # on suppose l'hamiltonien a appliquer : [("IZZ", 1), ("IZI", 2)]?
    # Utiliser SparsePauliOp.from_list([("IZZ", 1), ("IZI", 2)]) et QAOAAnsatz 




    return qc
