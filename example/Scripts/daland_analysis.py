from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np


def build_result_arrays(avg_daland_results, model_results, key='unattested'):
    out = [[],[]]
    for k in avg_daland_results:
        if k == key:
            for ons in avg_daland_results[k]:
                out[0].append(avg_daland_results[k][ons])
                out[1].append(model_results[k][ons])
    return(np.array(out))

def evaluate(daland_path, plots=False):
    daland_file = open(daland_path, 'r')
    header = daland_file.readline()

    raw_daland_results = defaultdict(lambda: defaultdict(list))
    for line in daland_file:
        line = line.strip().split(',')
        ons = ' '.join(line[-2].split()[:2]).replace('"', '')
        att = line[1].replace('"', '')
        raw_daland_results[att][ons].append(float(line[2]))


    avg_daland_results = defaultdict(lambda: defaultdict(float))
    for k1 in raw_daland_results:
        for k2 in raw_daland_results[k1]:
            avg_daland_results[k1][k2] = np.mean(raw_daland_results[k1][k2])


    clusters = ['dg', 'dn', 'fn', 'km', 'lm', 'ln', 'lt', 'ml',
    'mr', 'nl', 'pk', 'pw', 'rd', 'rg', 'rl', 'rn', 'vw', 'zr', 'tl']

    sons = [0, 1, 1, 1, -1, -1, -2, 1, 2, 1, 0, 3, -3, -3, -1, -2, 3, 3, 2]


    for k,v in list(avg_daland_results['unattested'].items()):
        if k[-1] in ['W', 'L', 'R']:
            print(k, v)
        else: 
            print(k, v)

   
evaluate("Daland_etal_2011__AverageScores.csv")