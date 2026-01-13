import numpy as np

class MaxXorSat:
    def __init__(self, n  , m , A, b):
        # A taille nxm
        if (len(A) != n):
            print("A n'a pas n lignes");
        if (len(A[0]) != m):
            print("A n'a pas m colonnes")


        # b taille nx1
        if(len(b) != n):
            print("b n'a pas n lignes")

        self.n = n
        self.m = m
        self.A = A
        seld.b = b


#retourne la solution qui maximise et utilité = nombre d'égalité vérifié
def solve(entry: MaxXorSat):

