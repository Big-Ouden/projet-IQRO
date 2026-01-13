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

# Ecrire une fonction qui prend en entrée une instance du problème MaxXorSat et résout
#l’instance. La réponse de la fonction doit être la solution et son utilité.
#Vous pouvez utiliser un algorithme d’énumération de toutes les solutions, ou un programme
#linéaire - vous trouverez une formulation linéaire dans la section 5.2.2 de https: // arxiv.
#org/ abs/ 1309. 6827 .
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

entry = MaxXorSat(2,2,[[1,1],[0,1]], [0,1])
print(solve(entry))