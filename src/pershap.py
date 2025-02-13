import numpy as np
import scipy as sc
import math, os, sys
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


def pri_ce(clients, groups, acc):
    '''
    compute the privacy-preserving contribution scores
    (input) clients: client number
    (input) groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input) acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (output) i1i:  include-one-in
    (output) l1o:  leave-one-out
    (output) ieei: include-everybody-else-in
    (output) leeo: leave-everybody-else-out
    (output) se:   self evaluation (i1i+l1o)
    (output) ee:   everybody else (ieei+leeo)
    (output) ppce: privacy-preserving contribution evaluation ((i1i+l1o+ieei+leeo)
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
    ieei = ieei / (clients - 1) ** 2
    leeo = leeo / (clients - 1) ** 2
    se = i1i + l1o
    ee = ieei + leeo
    ppce = se + ee
    return i1i, l1o, ieei, leeo, se, ee, ppce

def simulate_game(type, clients):
    '''
    generating coalitions and assigning to them scores according to one of the three rules
    (input) clients: number of clients
    (input) type:    assigning random / size-based / client-based noise to the coalitions
        'rand' - assign uniform random scores to the coalitions
        'size' - assign cardinality-based Gauss noise to the coalitions
        'cli'  - assign uniform random integers to the clients and add Gauss noise to the coalitions
    (output) groups: binary assignment matrix determining all coalitions
    (output) acc:    accuracies corresponding to the coalitions
    '''
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
    return groups, acc


def combine(clients, groups, acc):
    '''
    computing the optimal linear combination to approximate the Shapley
    (input)  clients: client number
    (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (output) weights: optimal weights
    - se:   optimal weights for self evaluation (i1i & l1o)
    - ee:   optimal weights for everybody else (ieei & leeo)
     -ppce: optimal weights for total combination (i1i & l1o & ieei & leeo)
    (output) values: contribution scores
    - sv, i1i, l1o, ieei, leeo, se, ee, ppce
    '''
    SV = shapley(clients, groups, acc)
    i1i, l1o, ieei, leeo = pri_ce(clients, groups, acc)
    X_self = np.transpose([i1i, l1o])
    weights_se = np.linalg.lstsq(X_self, SV, rcond=None)[0]
    X_else = np.transpose([ieei, leeo])
    weights_ee = np.linalg.lstsq(X_else, SV, rcond=None)[0]
    X_total = np.transpose([i1i, l1o, ieei, leeo])
    weights_ppce = np.linalg.lstsq(X_total, SV, rcond=None)[0]
    return {'weights': {'se': weights_se, 'ee': weights_ee, 'ppce': weights_ppce},
            'values': {'sv': SV, 'i1i': i1i, 'l1o': l1o, 'ieei': ieei, 'leeo': leeo,
                       'se': (weights_se[0]) * i1i + (weights_se[1]) * l1o,
                       'ee': (weights_ee[0]) * ieei + (weights_ee[1]) * leeo,
                        'ppce': (weights_ppce[0]) * i1i + (weights_ppce[1]) * l1o +
                                (weights_ppce[2]) * ieei +(weights_ppce[3]) * leeo}}


def error_comp(clients, groups, acc, w_ppce=None, w_ee=None, w_se=None):
    '''
    computing the least square error and the Spearman coefficient for the various SV approximations
    (input) clients: number of clients
    (input) type:    how to assign score to the coalitions
        'rand|size|cli' - use random distribution
        'cust'          - use given groups & accuracies
    (input) w_ppce:  weights for i1i & l1o & ieei & leeo combination
    (input) w_ee:    weights for ieei & leeo combination
    (input) w_se:    weights for i1i & l1o combination
                     None means equal weighting
    (output) LeastSquareError: smaller the better, min 0
    - sv, i1i, l1o, ieei, leeo, se, ee, ppce
    (output) SpearmanCoefficient: higher the better, max 1
    - sv, i1i, l1o, ieei, leeo, se, ee, ppce
    '''
    sv = shapley(clients, groups, acc)
    i1i, l1o, ieei, leeo , se, ee, ppce = pri_ce(clients, groups, acc)
    if w_se is not None:
        se = np.add(w_se[0] * i1i, w_se[1] * l1o)
    if w_ee is not None:
        ee = np.add(w_ee[0] * ieei, w_ee[1] * leeo)
    if w_ppce is not None:
        ppce = np.add(np.add(w_ppce[0] * i1i, w_ppce[1] * l1o),
                      np.add(w_ppce[2] * ieei, w_ppce[3] * leeo))
    return {'lse': {'il1':  mean_squared_error(sv, i1i,  squared=False), 'l1o':  mean_squared_error(sv, l1o,  squared=False),
                    'ieei': mean_squared_error(sv, ieei, squared=False), 'leeo': mean_squared_error(sv, leeo, squared=False),
                    'se':   mean_squared_error(sv, se,   squared=False), 'ee':   mean_squared_error(sv, ee,   squared=False),
                    'ppce': mean_squared_error(sv, ppce, squared=False)},
            'sc':  {'il1':  sc.stats.spearmanr(sv, i1i)[0],  'l1o':  sc.stats.spearmanr(sv, l1o)[0],
                    'ieei': sc.stats.spearmanr(sv, ieei)[0], 'leeo': sc.stats.spearmanr(sv, leeo)[0],
                    'se':   sc.stats.spearmanr(sv, se)[0],   'ee':   sc.stats.spearmanr(sv, ee)[0],
                    'ppce': sc.stats.spearmanr(sv, ppce)[0]}}


def plot_pri_ce(clients, groups, acc):
    '''
    plotting i1i, l1o, ieei, leeo, and the sv for the given game
    (input)  clients: client number
    (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (output) plt:     the contribution values method-wise for each client
    '''
    sv = shapley(clients, groups, acc)
    i1i, l1o, ieei, leeo, se, ee, ppce = pri_ce(clients, groups, acc)
    values = {'i1i': i1i, 'l1o': l1o, 'ieei': ieei, 'leeo': leeo, 'sv': sv}
    tmp = {}
    for n in range(clients):
        tmp[n] = [value[n] for key, value in values.items()]
    x = np.arange(clients)
    width = 0.15
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, value in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=method)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Values')
    ax.set_title('Contribution scores')
    ax.set_xticks(x + 2 * width, range(clients))
    ax.legend(loc='upper center', ncols=5)
    return plt


def plot_pri_comb(clients, groups, acc, opt=None):
    '''
    plotting i1i, l1o, ieei, leeo, se, ee, ppce, and the sv for the given game
    (input)  clients: client number
    (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (input)  opt:     use for optimal weights for se, ee, and ppce
    (output) plt:     the contribution values method-wise for each client
    '''
    sv = shapley(clients, groups, acc)
    i1i, l1o, ieei, leeo, se, ee, ppce = pri_ce(clients, groups, acc)
    if opt is not None:
        tmp = combine(clients, groups, acc)
        values = {'sv': sv, 'i1i': i1i, 'l1o': l1o, 'ieei': ieei, 'leeo': leeo,
                  'se': tmp['values']['se'], 'ee': tmp['values']['ee'], 'ppce': tmp['values']['ppce']}
    else:
        values = {'sv': sv, 'i1i': i1i, 'l1o': l1o, 'ieei': ieei, 'leeo': leeo,
                  'se': se, 'ee': ee, 'ppce': ppce}
    x = np.arange(clients)
    width = 0.1
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, value in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=method)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Values')
    ax.set_title('Contribution scores')
    ax.set_xticks(x + 3.5 * width, range(clients))
    ax.legend(loc='upper center', ncols=4)
    return plt


def plot_pri_ce_perf(clients, groups, acc):
    '''
    plotting the Least Square Error and the Spearman coefficient of i1i, l1o, ieei, leeo for the given game
    (input)  clients: client number
    (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (output) plt:     the approximation accuracy for the contribution scores
    '''
    errors = error_comp(clients, groups, acc)
    x = np.arange(len(errors['lse']))
    width = 0.2
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, error in errors.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [error[key] for key in error], width, label=method)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Error & Correlation')
    ax.set_title('Approximation accuracy')
    ax.set_xticks(x + 0.5 * width, {key: value for key, value in errors['lse'].items() if key != 'sv'})
    ax.legend(loc='upper center', ncols=len(errors['lse']))
    return plt


def plot_pri_comb_perf(clients, groups, acc, opt=None):
    '''
    plotting the Least Square Error and the Spearman coefficient of i1i, l1o, ieei, leeo for the given game
    (input)  clients: client number
    (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
    (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
    (input)  opt:     use for optimal weights for se, ee, and ppce
    (output) plt:     the approximation accuracy for the contribution scores
    '''
    if opt is not None:
        tmp = combine(clients, groups, acc)
        errors = error_comp(clients, groups, acc, tmp['weights']['ppce'], tmp['weights']['ee'], tmp['weights']['se'])
    else:
        errors = error_comp(clients, groups, acc)
    x = np.arange(len(errors['lse']))
    width = 0.2
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for method, error in errors.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [error[key] for key in error], width, label=method)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Error & Correlation')
    ax.set_title('Approximation accuracy')
    ax.set_xticks(x + 0.5 * width, {key: value for key, value in errors['lse'].items() if key != 'sv'})
    ax.legend(loc='upper center', ncols=len(errors['lse']))
    return plt


clients = 3
type = 'cli'
groups, acc = simulate_game(type, clients)
plt = plot_pri_ce_perf(clients, groups, acc)
plt.show()
#plt.savefig(os.path.abspath(sys.argv[0])[:-14] + "\\plots\\" + X + ".png", dpi=300, bbox_inches='tight')
plt.close()
