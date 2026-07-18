import numpy as np
import torch

OHE = {"N":np.array([0,0,0,0]), "A":np.array([1,0,0,0]), "C":np.array([0,1,0,0]), "G":np.array([0,0,1,0]), "T":np.array([0,0,0,1])}
ALPHA = "NACGT"

def no_encoding(strand):
    return strand

def strand_to_vector(strand):
    strand = strand.upper()
    vector = []
    for bp in strand:
        vector.append(np.array(OHE[bp]))
    return np.array(vector)

def strand_to_int(strand):
    strand = strand.upper()
    vector = []
    for bp in strand:
        vector.append(ALPHA.index(bp))
    return np.array(vector)

def vector_to_strand(vector):
    BASE = "NACGT"
    strand = ""
    for v in vector:
        for i in range(len(v)):
            if v[i] == 1:
                strand += BASE[i]
                break
    return strand