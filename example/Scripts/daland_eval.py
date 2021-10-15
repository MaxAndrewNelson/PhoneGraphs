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

def evaluate(model_path, daland_path, plots=False):
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

    model_file = open(model_path, 'r')
    model_results = defaultdict(lambda: defaultdict(float))

    for line in model_file:

        line = line.strip().split('\t')
        # print(line)
        ons = ' '.join(line[0].strip().split(' ')[:2])
        score = float(line[1])
        for k in avg_daland_results:
            if ons in avg_daland_results[k]:
                model_results[k][ons] = score



    unattested = build_result_arrays(avg_daland_results, model_results, 'unattested')

    from scipy import stats
    r, p = stats.kendalltau(unattested[0], unattested[1])
    sr, sp = stats.pearsonr(unattested[0], unattested[1])
    

    if plots:
    ### plot human by model judgments ###

        clusters = ['dg', 'dn', 'fn', 'km', 'lm', 'ln', 'lt', 'ml',
        'mr', 'nl', 'pk', 'pw', 'rd', 'rg', 'rl', 'rn', 'vw', 'zr', 'tl']

        poly = np.polyfit(unattested[1], unattested[0], deg=2)
        poly_f = np.poly1d(poly)
        func_min, func_max = np.min(unattested[1]), np.max(unattested[1])
        func_in = np.arange(func_min-1, func_max+1, (func_max-func_min)/100)

        for i,k in enumerate(clusters):
            plt.annotate(k, xy=(unattested[1,i], unattested[0,i]), weight='bold')

        plt.xlabel('Harmony Score')
        plt.ylabel('Human Well-Formedness Score')

        plt.scatter(unattested[1], unattested[0], color='white')
        plt.plot(func_in, poly_f(func_in), color='black')

        plt.show()

        ### sonority plots ###
        # clusters = ['dg', 'dn', 'fn', 'km', 'lm', 'ln', 'lt', 'ml',
        # 'mr', 'nl', 'pk', 'pw', 'rd', 'rg', 'rl', 'rn', 'vw', 'zr', 'tl']

        # sons = [0, 1, 1, 1, -1, -1, -2, 1, 2, 1, 0, 3, -3, -3, -1, -2, 3, 3, 2]

        # for i,k in enumerate(clusters):
        #     # plt.annotate(k, xy=(unattested[1,i], unattested[0,i]), weight='bold')
        #     plt.annotate(k, xy=(sons[i], unattested[1,i]), weight='bold')


        # from sklearn.linear_model import LinearRegression
        # lr = LinearRegression()
        # lr.fit(unattested[1].reshape(-1,1), unattested[0].reshape(-1,1))


        # print(lr.score(unattested[1].reshape(-1,1), unattested[0].reshape(-1,1)))

        # resid = lr.predict(unattested[1].reshape(-1,1)) - unattested[0].reshape(-1,1)

        # print(unattested[1].shape, resid[:,0].shape)
        # srr_resid,p_resid = stats.pearsonr(unattested[1], resid[:,0])
        # print(srr_resid, p_resid)


        # lr2 = LinearRegression()
        # lr2.fit(np.array(sons).reshape(-1,1), unattested[1].reshape(-1,1))
        # son_r = np.sqrt(lr2.score(np.array(sons).reshape(-1,1), unattested[1].reshape(-1,1)))

        # # plt.xlabel('Harmony Score')
        # # plt.ylabel('Human Well-Formedness Score')

        # plt.xlabel('Sonority Profile')
        # plt.ylabel('Harmony Score')

        # # plt.scatter(unattested[1], unattested[0], color='white')
        # plt.scatter(sons, unattested[1], color='white')
        # plt.plot(sons, lr2.predict(np.array(sons).reshape(-1,1)), color='red')

        # plt.annotate('$r=${:.3f}'.format(son_r), xy = (2, -32.5))

        # plt.show()

    return(r, p, sr, sp)

if __name__ == '__main__':
    import sys

    tau, taup, sr, sp = evaluate(sys.argv[1], sys.argv[2])

    print(f"\tTau: {tau:.4f} ({taup:.4f}). Pearson: {sr:.4f} ({sp:.4f})")

