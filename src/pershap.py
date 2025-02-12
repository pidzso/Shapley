import numpy as np
import scipy as sc
import math
import os
import sys
from sklearn.metrics import mean_squared_error
from itertools import product
import matplotlib.pyplot as plt


def shapley(clients, groups, acc):
    '''
    compute the Shapley Value
        (input)  clients: client number
        (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
        (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
        (output) score:   Shapley value of the players
    '''
    scores = np.zeros(clients)
    for i in range(clients):
        tmp = 0
        for j, subset in enumerate(groups):
            if subset[i] == 0:
                continue
            subset_without_i = np.copy(subset)
            subset_without_i[i] = 0
            subset_index = j
            idx = [k for k, l in enumerate(groups) if all(x == y for x, y in zip(l, subset_without_i))]
            subset_without_i_index = idx[0]
            marginal_contribution = acc[subset_index] - acc[subset_without_i_index]
            subset_size = np.sum(subset) - 1
            weight = (math.factorial(subset_size) * math.factorial(clients - subset_size - 1)) / math.factorial(clients)
            tmp += weight * marginal_contribution
        scores[i] = tmp
    return scores


def ppce(clients, groups, acc):
    '''
    compute the privacy-preserving contribution scores
        (input)  clients: client number
        (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
        (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
        (output) i1i:     include-one-in
        (output) l1o:     leave-one-out
        (output) ieei:    include-everybody-else-in
        (output) leeo:    leave-everybody-else-out
    '''
    i1i = np.zeros(clients)
    l1o = np.zeros(clients)
    ieei = np.zeros(clients)
    leeo = np.zeros(clients)
    include = np.zeros(clients)
    leave = np.zeros(clients)
    for i, subset in enumerate(groups):
        if np.sum(subset) == 0:
            null = acc[i]
        if np.sum(subset) == clients:
            grand = acc[i]
        if np.sum(subset) == 1:
            tmp = [j for j, k in enumerate(subset) if k == 1]
            include[tmp[0]] = acc[i]
        if np.sum(subset) == clients - 1:
            tmp = [j for j, k in enumerate(subset) if k == 0]
            leave[tmp[0]] = acc[i]
    for i in range(clients):
        i1i[i] = include[i] - null
        l1o[i] = grand - leave[i]
        for j in range(clients):
            ieei[i] += leave[j] - null
            leeo[i] += grand - include[j]
        ieei[i] -= leave[i] - null
        leeo[i] -= grand - include[i]
    return i1i, l1o, ieei / (clients - 1) ** 2, leeo / (clients - 1) ** 2


def optim(clients, type, mc):
    '''
    computing the optimal linear combination to approximate the Shapley
    (input) clients: number of clients
    (input) type:    assigning random / size-based / client-based noise to the coalitions
        'rand' - assign uniform random scores to the coalitions
        'size' - assign cardinality-based Gauss noise to the coalitions
        'cli'  - assign uniform random integers to the clients and add Gauss noise to the coalitions
    (input) mc:      number of MonteCarlo simulations
    (output) se:     optimal weights for self evaluation - i1i & l1o
    (output) ee:     optimal weights for everybody else  - ieei & leeo
    (output) total:  optimal weights for total combination - i1i & l1o & ieei & leeo
    '''
    weights_self = [0, 0]
    weights_else = [0, 0]
    weights_total = [0, 0, 0, 0]
    for i in range(mc):
        groups = [list(seq) for seq in product([0, 1], repeat=clients)]
        if type == 'rand':
            acc = np.random.rand(2 ** clients)
        elif type == 'size':
            acc = [np.random.normal(np.sum(k), 1) for k in groups]
        elif type == 'cli':
            acc = np.zeros(2 ** clients)
            cli = [np.random.randint(clients) for j in range(clients)]
            for j, l in enumerate(acc):
                for k in range(clients):
                    if groups[j][k] == 1:
                        acc[j] += np.random.normal(cli[k], 1)
        SV = shapley(clients, groups, acc)
        i1i, l1o, ieei, leeo = ppce(clients, groups, acc)
        X_self = np.transpose([i1i, l1o])
        weights_self += np.linalg.lstsq(X_self, SV, rcond=None)[0]
        X_else = np.transpose([ieei, leeo])
        weights_else += np.linalg.lstsq(X_else, SV, rcond=None)[0]
        X_total = np.transpose([i1i, l1o, ieei, leeo])
        weights_total += np.linalg.lstsq(X_total, SV, rcond=None)[0]
    return np.round(weights_self / mc, 3), np.round(weights_else / mc, 3), np.round(weights_total / mc, 3)


def avg_approx(w_total, w_ee, w_se, clients, type, mc):
    '''
    computing the least square error and the Spearman coeficient for the various SV approximations
    (input) w_total: optimal weights for i1i & l1o & ieei & leeo combination
    (input) w_ee:    optimal weights for ieei & leeo combination
    (input) w_se:    optimal weights for i1i & l1o combination
    (input) clients: number of clients
    (input) type:    assigning random / size-based / client-based noise to the coalitions
        'rand' - assign uniform random scores to the coalitions
        'size' - assign cardinality-based Gauss noise to the coalitions
        'cli'  - assign uniform random integers to the clients and add Gauss noise to the coalitions
    (input) mc:      number of MonteCarlo simulations
    (output) {method: LeastSquareError, SpearmanCoefficient}
        'i1i' - 'l1o' - 'ieei' - 'leeo' - 'se' - 'ee' - 'total'
    '''
    e_i1i, s_i1i, s_l1o, e_l1o, s_ieei, e_ieei, s_leeo, e_leeo, s_se, e_se, s_ee, e_ee, s_total, e_total = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(mc):
        groups = [list(seq) for seq in product([0, 1], repeat=clients)]
        if type == 'rand':
            acc = np.random.rand(2 ** clients)
        elif type == 'size':
            acc = [np.random.normal(np.sum(k), 1) for k in groups]
        elif type == 'cli':
            acc = np.zeros(2 ** clients)
            cli = [np.random.randint(clients) for j in range(clients)]
            for j, l in enumerate(acc):
                for k in range(clients):
                    if groups[j][k] == 1:
                        acc[j] += np.random.normal(cli[k], 1)
        SV = shapley(clients, groups, acc)
        i1i, l1o, ieei, leeo = ppce(clients, groups, acc)
        se = np.add(w_se[0] * i1i, w_se[1] * l1o)
        ee = np.add(w_ee[0] * ieei, w_ee[1] * leeo)
        total = np.add(np.add(w_total[0] * i1i, w_total[1] * l1o),
                       np.add(w_total[2] * ieei, w_total[3] * leeo))
        e_i1i += mean_squared_error(SV, i1i, squared=False)
        e_l1o += mean_squared_error(SV, l1o, squared=False)
        e_ieei += mean_squared_error(SV, ieei, squared=False)
        e_leeo += mean_squared_error(SV, leeo, squared=False)
        e_se += mean_squared_error(SV, se, squared=False)
        e_ee += mean_squared_error(SV, ee, squared=False)
        e_total += mean_squared_error(SV, total, squared=False)
        s_i1i += sc.stats.spearmanr(SV, i1i)[0]
        s_l1o += sc.stats.spearmanr(SV, l1o)[0]
        s_ieei += sc.stats.spearmanr(SV, ieei)[0]
        s_leeo += sc.stats.spearmanr(SV, leeo)[0]
        s_se += sc.stats.spearmanr(SV, se)[0]
        s_ee += sc.stats.spearmanr(SV, ee)[0]
        s_total += sc.stats.spearmanr(SV, total)[0]
    return {'i1i': [np.round(e_i1i / mc, 3), np.round(s_i1i / mc, 3)],
            'l1o': [np.round(e_l1o / mc, 3), np.round(s_l1o / mc, 3)],
            'ieei': [np.round(e_ieei / mc, 3), np.round(s_ieei / mc, 3)],
            'leeo': [np.round(e_leeo / mc, 3), np.round(s_leeo / mc, 3)],
            'se': [np.round(e_se / mc, 3), np.round(s_se / mc, 3)],
            'ee': [np.round(e_ee / mc, 3), np.round(s_ee / mc, 3)],
            'total': [np.round(e_total / mc, 3), np.round(s_total / mc, 3)]}


def weightplot(num, mc):
    '''
    plot the weights (client number)-wise for the 3 combinations and for 3 games
        (input) num: set of client numbers
        (input) mc:  number of MonteCarlo simulations
        (output) save: the weight used in SE, EE, and total for rand, size, and cli game
                 form: plots/ + self|else|total + _ + rand|size|cli + .png
    '''
    param = {}
    for c in num:
        param[c] = {'rand': {}, 'size': {}, 'cli': {}}
        for s in ['rand', 'size', 'cli']:
            param[c][s] = {'self': [], 'else': [], 'total': []}
        for s in ['rand', 'size', 'cli']:
            param[c][s]['self'], param[c][s]['else'], param[c][s]['total'] = optim(c, s, mc)
    for t in ['self', 'else', 'total']:
        if t == 'self':
            label1, label2 = 'I1i', 'L1O'
        if t == 'else':
            label1, label2 = 'IEEi', 'LEEO'
        if t == 'total':
            label1, label2, label3, label4 = 'I1i', 'L1O', 'IEEi', 'LEEO'
        for s in ['rand', 'size', 'cli']:
            data = {label1: [param[c][s][t][0] for c in num], label2: [param[c][s][t][1] for c in num]}
            if t == 'total':
                data[label3] = [param[c][s][t][2] for c in num]
                data[label4] = [param[c][s][t][3] for c in num]
            x = np.arange(len(num))
            width = 0.25 / (len(data.keys())/2)
            multiplier = 0
            fig, ax = plt.subplots(layout='constrained')
            for method, weights in data.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, weights, width, label=method)
                ax.bar_label(rects, padding=3)
                multiplier += 1
            ax.set_ylabel('Weights', fontsize='x-large')
            ax.set_title(t + '-based evaluation for the ' + s + '-game', fontsize='x-large')
            ax.set_xticks(x + (len(data.keys()) - 1) * width / 2, num)
            ax.legend(loc='upper center', fontsize='x-large', ncols=len(data.keys()))
            #ax.set_ylim(-1, 2)
            #plt.show()
            plt.savefig(os.path.abspath(sys.argv[0])[:-14] + "\\plots\\weights_" + t + '_' + s + ".png", dpi=300, bbox_inches='tight')
            plt.close()
    return 0


weightplot([3, 4, 5, 6], 10000)

'''
w_s_r3, w_e_r3, w_t_r3 = optim(3, 'rand', 1000)
w_s_s3, w_e_s3, w_t_s3 = optim(3, 'size', 1000)
w_s_c3, w_e_c3, w_t_c3 = optim(3, 'cli',  1000)
w_s_r4, w_e_r4, w_t_r4 = optim(4, 'rand', 1000)
w_s_s4, w_e_s4, w_t_s4 = optim(4, 'size', 1000)
w_s_c4, w_e_c4, w_t_c4 = optim(4, 'cli',  1000)
w_s_r5, w_e_r5, w_t_r5 = optim(5, 'rand', 1000)
w_s_s5, w_e_s5, w_t_s5 = optim(5, 'size', 1000)
w_s_c5, w_e_c5, w_t_c5 = optim(5, 'cli',  1000)

print('3 clients')
print('weights - se / ee / total')
print('rand\t',  w_s_r3, w_e_r3, w_t_r3)
print('size\t',  w_s_s3, w_e_s3, w_t_s3)
print('cli\t\t', w_s_c3, w_e_c3, w_t_c3)
print('i1i, l1i, ieei, leeo, se / ee / total - Error / Order')
print(avg_approx(w_t_r3, w_e_r3, w_s_r3, 3, 'rand', 1000))
print(avg_approx(w_t_s3, w_e_s3, w_s_s3, 3, 'size', 1000))
print(avg_approx(w_t_c3, w_e_c3, w_s_c3, 3, 'cli', 1000))
print('4 clients')
print('weights - se / ee / total')
print('rand\t',  w_s_r4, w_e_r4, w_t_r4)
print('size\t',  w_s_s4, w_e_s4, w_t_s4)
print('cli\t\t', w_s_c4, w_e_c4, w_t_c4)
print('i1i, l1i, ieei, leeo, se / ee / total - Error / Order')
print(avg_approx(w_t_r4, w_e_r4, w_s_r4, 4, 'rand', 1000))
print(avg_approx(w_t_s4, w_e_s4, w_s_s4, 4, 'size', 1000))
print(avg_approx(w_t_c4, w_e_c4, w_s_c4, 4, 'cli', 1000))
print('4 clients')
print('weights - se / ee / total')
print('rand\t',  w_s_r5, w_e_r5, w_t_r5)
print('size\t',  w_s_s5, w_e_s5, w_t_s5)
print('cli\t\t',  w_s_c5, w_e_c5, w_t_c5)
print('i1i, l1i, ieei, leeo, se / ee / total - Error / Order')
print(avg_approx(w_t_r5, w_e_r5, w_s_r5, 5, 'rand', 1000))
print(avg_approx(w_t_s5, w_e_s5, w_s_s5, 5, 'size', 1000))
print(avg_approx(w_t_c5, w_e_c5, w_s_c5, 5, 'cli', 1000))
'''