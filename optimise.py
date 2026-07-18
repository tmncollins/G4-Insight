import torch
import re
import numpy as np
import random
from model import G4_args, G4_RNN, load_model
from encoding import strand_to_int as encode
from ushuffle import shuffle as dishuffle

# Returns if string is a PQS of a given size
def is_pqs_size(string, size):
    string = string.upper()
    s = str(size)
    regex = re.compile('G{' + s + '}[ACGT]{1,7}G{' + s + '}[ACGT]{1,7}G{' + s + '}[ACGT]{1,7}G{' + s + '}')
    if regex.search(string): return True
    return False

# Returns if string is a PQS
def is_pqs(string):
    for i in range(3,6):
        if is_pqs_size(string, i): return True
    return False

# Mutates a DNA string with fixed chance for each NT
def mutate(string, mutation_chance=0.05):
    string2 = ""
    for c in string:
        if random.uniform(0, 1) < mutation_chance:
            string2 += "ACGT"[random.randint(0, 3)]
        else: string2 += c
    return string2

# Generates a random DNA PQS string of a given length
def random_pqs(length):
    while True:
        dna = "ACT"[random.randint(0,2)]
        g_stickiness = 0.5
        while len(dna) < length:
            if dna[-1] == 'G':
                if random.uniform(0,1) < g_stickiness:
                    dna += 'G'
                else:
                    dna += "ACT"[random.randint(0,2)]
            else:
                dna += 'ACGT'[random.randint(0,2)]
        if is_pqs(dna):
            return dna

def _construct_dna_graph(seq):
    bases = 'ACGT'
    node = {b:0 for b in bases}
    graph = {b:dict(node) for b in bases}

    for i in range(1, len(seq)):
        u = seq[i-1]; v = seq[i]
        graph[u][v] += 1

    return graph

def _euler_walk_dna_graph(graph, length, u=-1):
    if length == 0:
        return 'X'

    bases = list('ACGT')
    random.shuffle(bases)

    if u == -1:
        for b in bases:
            x = _euler_walk_dna_graph(graph, length-1, b)
            if x: return b + x
        return False
    
    for b in bases:
        if graph[u][b] > 0:
            graph[u][b] -= 1
            x = _euler_walk_dna_graph(graph, length-1, b)
            if x: return b + x
            graph[u][b] += 1
    return False

# Shuffles DNA string, preserving k-mer counts
def kshuffle_string(s, k):
    s = s.encode("utf-8")
    return dishuffle(s, k).decode("utf-8")

# Shuffles DNA string preserving dimer counts
def dishuffle_string(s):
    return kshuffle_string(s, 2)

# Shuffles DNA string
def shuffle_string(s):
    s = list(s)
    random.shuffle(s)
    return ''.join(s)

# Generates random DNA string og a given length
def random_dna(length):
    s = []
    for _ in range(length): s.append('ACGT'[random.randint(0,3)])
    return ''.join(s)

# Calculates average background model prediction score for a sequence using shuffle
def score_kernel_shuffle(model, seq, kernel_size, start, n_shuffle=20, device="cpu"):
    scores = []
    s = seq[start:start+kernel_size]
    left = seq[:start]
    right = seq[start+kernel_size:]
    for _ in range(n_shuffle):
        s2 = shuffle_string(s)
        seq2 = left + s2 + right
        pqs = torch.tensor(encode(seq2)).int().to(device)
        up = torch.tensor(encode("")).int().to(device)
        down = torch.tensor(encode("")).int().to(device)
        score = model(up, pqs, down).item()
        scores.append(score)
    
    return sum(scores) / n_shuffle

# Calculates average background model prediction score for a sequence using kshuffle
def score_kernel_kshuffle(model, seq, kernel_size, start, k=2, n_shuffle=20, device="cpu"):
    scores = []
    s = seq[start:start+kernel_size]
    left = seq[:start]
    right = seq[start+kernel_size:]
    for _ in range(n_shuffle):
        s2 = kshuffle_string(s, k)
        seq2 = left + s2 + right
        pqs = torch.tensor(encode(seq2)).int().to(device)
        up = torch.tensor(encode("")).int().to(device)
        down = torch.tensor(encode("")).int().to(device)
        score = model(up, pqs, down).item()
        scores.append(score)
    
    return sum(scores) / n_shuffle

# Calculates average background model prediction score for a sequence using random
def score_kernel_random(model, seq, kernel_size, start, n_random=20, device="cpu"):
    scores = []
    left = seq[:start]
    right = seq[start+kernel_size:]
    for _ in range(n_random):
        s2 = random_dna(kernel_size)
        seq2 = left + s2 + right
        pqs = torch.tensor(encode(seq2)).int().to(device)
        up = torch.tensor(encode("")).int().to(device)
        down = torch.tensor(encode("")).int().to(device)
        score = model(up, pqs, down).item()
        scores.append(score)
    
    return sum(scores) / n_random

def analyse_kernel_shuffle(model, seq, kernel_size=16, n_shuffle=20, device="cpu"):
    model.to(device)   

    k_weights = []

    pqs = torch.tensor(encode(seq)).int().to(device)
    up = torch.tensor(encode("")).int().to(device)
    down = torch.tensor(encode("")).int().to(device)

    orig_score = model(up, pqs, down).item()

    for start in range(len(seq)+1-kernel_size):
        shuffle_score = score_kernel_shuffle(model, seq, kernel_size, start, n_shuffle, device)
        k_weights.append(orig_score - shuffle_score)

    weights = [0 for _ in range(len(seq))]

    for i in range(len(k_weights)):
        for j in range(kernel_size):
            weights[i+j] += k_weights[i]

    return weights

def predict(model, seq, device="cpu"):
    pqs = torch.tensor(encode(seq)).int().to(device)
    up = torch.tensor(encode("")).int().to(device)
    down = torch.tensor(encode("")).int().to(device)

    return model(up, pqs, down).item() > 0

def predict_logit(model, seq, device="cpu"):
    pqs = torch.tensor(encode(seq)).int().to(device)
    up = torch.tensor(encode("")).int().to(device)
    down = torch.tensor(encode("")).int().to(device)

    return model(up, pqs, down).item()


def analyse_kernel_kshuffle(model, seq, kernel_size=16, k=2, n_shuffle=20, device="cpu"):
    model.to(device)   

    k_weights = []

    pqs = torch.tensor(encode(seq)).int().to(device)
    up = torch.tensor(encode("")).int().to(device)
    down = torch.tensor(encode("")).int().to(device)

    orig_score = model(up, pqs, down).item()

    for start in range(len(seq)+1-kernel_size):
        shuffle_score = score_kernel_kshuffle(model, seq, kernel_size, start, k, n_shuffle, device)
        k_weights.append(orig_score - shuffle_score)

    weights = [0 for _ in range(len(seq))]

    for i in range(len(k_weights)):
        for j in range(kernel_size):
            weights[i+j] += k_weights[i]

    return weights

def analyse_kernel_random(model, seq, kernel_size=16, n_random=20, device="cpu"):
    model.to(device)   

    k_weights = []

    pqs = torch.tensor(encode(seq)).int().to(device)
    up = torch.tensor(encode("")).int().to(device)
    down = torch.tensor(encode("")).int().to(device)

    orig_score = model(up, pqs, down).item()

    for start in range(len(seq)+1-kernel_size):
        random_score = score_kernel_random(model, seq, kernel_size, start, n_random, device)
        k_weights.append(orig_score - random_score)

    weights = [0 for _ in range(len(seq))]

    for i in range(len(k_weights)):
        for j in range(kernel_size):
            weights[i+j] += k_weights[i]

    return weights

# Genetic algorithm to optimise a PQS by mutating flank sequences only
def optimise_pqs(model, device='cpu', length=60, flank=10, n_gen=100, gen_survival=10, mutations=[0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.5],
                start=[], verbose=False, method="max"):
    if len(start) == 0:
        start = [random_pqs(length) for _ in range(5)]
    gen = list(start)

    model.eval()
    with torch.no_grad():
        for g in range(n_gen):
            if verbose:
                print('At generation', g)
            next_gen = list(gen)
            for s in gen:
                for m in mutations:
                    next_gen.append(mutate(s, m))
            next_gen_scores = []
            for s in next_gen:
                pqs = torch.tensor(encode(s)).int().to(device)
                up = torch.tensor(encode("")).int().to(device)
                down = torch.tensor(encode("")).int().to(device)
                score = model(up, pqs, down)

                pqs_m = 10*is_pqs(s[flank:-flank])
                if method == "max":
                    next_gen_scores.append((max(score, score * pqs_m), s, score))
                elif method == "min":
                    next_gen_scores.append((min(score, score * pqs_m), s, score))
            next_gen_scores.sort()
            if method == "max":
                next_gen_scores = next_gen_scores[::-1]
            if verbose and ((g+1) % 20 == 0):
                print(next_gen_scores[0], is_pqs(next_gen_scores[0][1]))
            gen = []
            for i in range(min(gen_survival, len(next_gen))):
                gen.append(next_gen_scores[i][1])
    return next_gen_scores[0][2], next_gen_scores[0][1]

# Genetic algorithm to optimise a PQS
def optimise_pqs_core(model, device='cpu', PQS="GGGAGGGAGGGAGGG", flank=20, n_gen=100, gen_survival=10, mutations=[0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.5],
                    verbose=False, method="max"):
    start = [[random_dna(flank), random_dna(flank)] for _ in range(5)]
    gen = list(start)

    model.eval()
    with torch.no_grad():
        for g in range(n_gen):
            if verbose:
                print('At generation', g)
            next_gen = list(gen)
            for s in gen:
                for m in mutations:
                    next_gen.append([mutate(s[0], m), mutate(s[1], m)])
            next_gen_scores = []
            pqs = torch.tensor(encode(PQS)).int().to(device)
            for s in next_gen:
                up = torch.tensor(encode(s[0])).int().to(device)
                down = torch.tensor(encode(s[1])).int().to(device)
                score = model(up, pqs, down)

                next_gen_scores.append((score, s))
            next_gen_scores.sort()
            if method == "max":
                next_gen_scores = next_gen_scores[::-1]
            if verbose and ((g+1) % 20 == 0):
                print(str(next_gen_scores[0][0]) + " : " + next_gen_scores[0][1][0] + "-" + PQS + "-" + next_gen_scores[0][1][1])
            gen = []
            for i in range(min(gen_survival, len(next_gen))):
                gen.append(next_gen_scores[i][1])
    return next_gen_scores[0][0], next_gen_scores[0][1][0] + PQS + next_gen_scores[0][1][1]
