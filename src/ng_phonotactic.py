from itertools import combinations, permutations
from random import sample
from phono_util import convert_to_ng, get_unattested, read_phone_file
from utils import arguments
import numpy as np
import torch
import sys
import regex as re

class constraint():
    def __init__(self, n, sets, name):
        self.n = n
        self.sets = sets
        full_re = []

        for s in sets:
            as_re = '|'.join([x for x in list(s)])
            full_re.append('(' + as_re + ')')

        self.regex = '[\\.]'.join(full_re)
        self.name = name
        self.eval = re.compile(self.regex)
    def evaluate(self, word):
        ''' return number of violations of a word '''
        return len(re.findall(self.eval, word, overlapped=True))

def get_data(df, word_boundaries=False, syll_boundaries=False):
    all_forms = []
    all_freqs = []
    for i, line in enumerate(df):
        line = line.strip().split('\t')
        line = line[0].split()
        if word_boundaries:
            if syll_boundaries:
                line = ['!' if x=='.' else x for x in line] #already using . as a special character
                all_forms.append('#.' + '.'.join(line) + '.' + '#.')
            else:
                all_forms.append('#.' + '.'.join(line) + '.' + '#.')
        else:
            all_forms.append('.'.join(line) + '.')
        all_freqs.append(1)

    return all_forms, all_freqs

def get_test_data(df, boundaries):
    all_forms = []
    for i, line in enumerate(df):
        line = line.strip().split()
        if '.'.join(line) + '.' not in all_forms:
            if boundaries:
                all_forms.append('#.!.' + '.'.join(line) + '.!.#.')
            else:
                all_forms.append('.'.join(line) + '.')
    return all_forms

# def LL_Loss(pred, targs):
#     lls = torch.nn.KLDivLoss()
#     return lls(pred, targs)

def log_likelihood(obs, pred):
    p = pred/np.sum(pred)
    p = np.clip(p, a_min=1e-20, a_max=1)

    log_probs = (-np.log(p) * obs)
    return(np.sum(log_probs))


# def get_probs(w, tabs, obs):
#     weighted = w*tabs
#     harmonies = torch.sum(weighted, dim=1)
#     hstar = torch.exp(harmonies)
#     denoms = torch.sum(hstar, dim=0)
#     probs = torch.log(hstar/denoms)

#     return probs

def get_probs(w, tabs, obs):
    weighted = w*tabs
    harmonies = np.sum(weighted, axis=1)
    hstar = np.exp(harmonies)
    denoms = (np.sum(hstar, axis=0))
    probs = hstar/denoms

    return probs

def get_harmony(w, tabs):
    weighted = w*tabs
    harmonies = np.sum(weighted, axis=1)

    return harmonies

def loss(w, tabs, obs, lm):
    pred = get_probs(w, tabs, obs)
    ll = log_likelihood(obs, pred) 
    ll += lm * np.sum(w**2)
    return ll

def backward(w, tabs, obs, lr=0.05, lamb=0.0, plx=None):
    weighted = w*tabs
    harmonies = np.sum(weighted, axis=1)

    # print(harmonies)

    hstar = np.exp(harmonies)
    denoms = (np.sum(hstar, axis=0))
    pred = hstar/denoms

    expected = np.sum(pred*tabs.transpose(), axis=1)
    observed = np.sum(obs.reshape(-1)*tabs.transpose(), axis=1)


    # print(expected[plx], observed[plx])
    # print("Delta: ", observed[plx] - expected[plx])

    grad = observed - expected
    # print(grad[plx])
    update = lr * (grad - lamb) 
    # print(update[plx])


    w += update


    w[w<0] = 0
    return w
    
def build_cons(class_file_path, phones, max_n, singles, 
    word_boundaries, syll_boundaries, use_comp=True):
    class_f = open(class_file_path, 'r')


    classes = []
    class_names = []

    #First add all classes specified in the file
    for l in class_f:
        line = l.strip().split('\t')
        curr_class = frozenset(line[0].split())
        curr_name = line[-1]


        if curr_class not in classes and len(curr_class) != 0: #add the class if its new and nonzero
            classes.append(curr_class)
            class_names.append(curr_name)

    #Then add all class complements, done second so that a complement that is characterizable
    #as a class is not described as a compliment
    if use_comp: 
        class_f.seek(0)
        for l in class_f:
            line = l.strip().split('\t')
            curr_class = frozenset(line[0].split())
            curr_name = line[-1]        
            curr_comp = frozenset(phones) - curr_class
            if curr_comp not in classes and len(curr_comp) != 0:
                classes.append(curr_comp)
                class_names.append('!' + curr_name)

    if singles: #add all singletons as classes regardless of whether or not they are uniqely specifiable
        classes += [frozenset([x]) for x in list(phones)]
        class_names += [x for x in phones]
    if word_boundaries: #word boundaries are a class
        classes += [frozenset(['#'])]
        class_names.append('word')
    if syll_boundaries: #syllable boundaries are a class
        classes += [frozenset(['!'])]
        class_names.append('syllable')


    assert len(classes) == len(class_names), "Classes and names do not align"
    
    #Build all n gram constraints from the specified classes
    cons = [] 
    class_ixs = [i for i in range(len(classes))]
    for n in range(1, max_n+1):
        all_ngrams = permutations(class_ixs, n)
        for sequence in all_ngrams:
            # print(sequence)
            # print(classes)
            curr_classes = [classes[x] for x in sequence]
            curr_name = ''.join(['[' + class_names[x] + ']' for x in sequence])

            cons.append(constraint(n, curr_classes, curr_name))
        
        # #add in constraints penalizing self combinations
        for ix in class_ixs:
            cons.append(constraint(n, [classes[ix] for x in range(n)], ''.join(['[' + class_names[ix] + ']' for x in range(n)])))
        
        # for sequence in [[c]*n for c in class_ixs]:
        #     cons.append(constraint(n, seq))

    class_f.close()

    return cons

def assign_vios(cands, con, ix2cluster):
    tableaux = np.zeros((len(cands), len(con)))

    for row in range(len(ix2cluster.keys())):
        # if row%1000 == 0:
        #     print('\tassigning violations to candidate {}...'.format(row))
        for col, c in enumerate(con):
            tableaux[row,col] -= c.evaluate(ix2cluster[row])

    print('\t\tAssigned violations to {} candidates'.format(row))


    return tableaux


def fit_phonotactic(data_file_path, natural_class_file, test_file_path, phones, 
                    out_dir, args, suffix=""):

    data_file = open(data_file_path, "r")
    attested, _ = get_data(data_file, args.word_boundaries, args.syll_boundaries)
    max_con_len = args.N


    attested, frequencies = convert_to_ng(max_con_len, attested)
    frequencies = np.array(frequencies)
    

    con = build_cons(natural_class_file, phones, max_con_len, args.singletons, args.word_boundaries, args.syll_boundaries)

    if args.syll_boundaries:
        phones.append('!')
    if args.word_boundaries:
        phones.append('#')

    unattested = get_unattested(attested, phones, args.N)

    #some duplicates may get added by singleton or boundary, just cutting them here for now
    #TO DO: Don't add duplicates to begin with
    dup_checks = []
    fixed_con = []
    for c in con:
        if c.regex not in dup_checks:
            dup_checks.append(c.regex)
            fixed_con.append(c)
    con = fixed_con

    print("\tConstructed a grammar with {} constraints".format(len(con)))

    all_candidates = attested + unattested
    candidate2ix = {v:k for k,v in enumerate(all_candidates)}
    ix2candidate = {k:v for v,k in candidate2ix.items()}

    ### subsets during development, building tableaux takes 136s with full trigram con ###
    # all_candidates = all_candidates[:1000] #subset for developmen

    tableaux = assign_vios(all_candidates, con, ix2candidate)

    # grammar = np.random.random((len(con),))
    grammar = np.zeros((len(con),))
    observed_counts = np.concatenate([frequencies, np.zeros(len(unattested),)])
    observed = observed_counts/np.sum(observed_counts)

    if args.type_frequency:
        observed = (observed > 0.0).astype(float)

    if args.opt == 'LBFGS':
        from scipy.optimize import minimize
        solution = minimize(loss, grammar, 
            method="L-BFGS-B", 
            args=(tableaux, observed, lam), 
            bounds=[(0.0, 200) for x in grammar])

        print("\tFinal loss {}".format(solution.fun))
        grammar = solution.x

    else:
        ll = 1e9
        for i in range(args.n_iter):
            new_ll = loss(grammar, tableaux, observed, lm=args.lam)
            grammar = backward(grammar, tableaux, observed, lr=args.lr, lamb=args.lam, plx=0)
            
            if args.non_negative:
                grammar = grammar * (grammar>0)

            if i%50 == 0:
                print("\t\tLoss at epoch {}: {:.3f}".format(i, new_ll))

                if ll - new_ll <=0.0000001:
                    print(ll, new_ll)
                    print('Stop early reached at {} iterations'.format(i))
                    break

            ll = new_ll

    grammar_path = out_path + "Grammars/" + suffix
    print(f"\tWriting grammar to {grammar_path}")

    oot_grammar = open(grammar_path, 'w', encoding='utf-8')
    cons_with_weights = []
    for k, c in enumerate(con):
        cons_with_weights.append((c.regex, grammar[k]))

    cons_with_weights.sort(key = lambda x: x[1], reverse=True) 
    for cw in cons_with_weights:
        if cw[1] > 0.000:
            oot_grammar.write(cw[0] + '\t{:.3f}'.format(cw[1]) + '\n')
    oot_grammar.close()

    print(f"\tTesting on the forms in {test_file_path}")

    test_file = open(test_file_path, "r")
    test_data, _ = get_data(test_file, args.word_boundaries, args.syll_boundaries)
    test_file.close()

    ix2test = {k:v for k,v in enumerate(test_data)}
    test_tableaux = assign_vios(test_data, con, ix2test)
    harmonies = get_harmony(grammar, test_tableaux)


    out_file = out_path + "/Judgements/" + suffix
    print(f"\tWriting harmony scores to {out_file}\n")
    oot = open(out_file, 'w', encoding='utf-8')
    for ix in ix2test.keys():
        oot.write(ix2test[ix].replace('#.','').replace('.', ' ') + '\t' + '{:.3f}'.format(harmonies[ix]) + '\n')
    oot.close()



if __name__ == "__main__":
    import sys

    training_data = sys.argv[1]
    class_file = sys.argv[2]
    test_file = sys.argv[3]
    phone_file = sys.argv[4]
    out_path = sys.argv[5]
    suffix = sys.argv[6]

    args = arguments(sys.argv[7], "phonotactic")

    phones = read_phone_file(phone_file)

    fit_phonotactic(training_data, class_file, test_file, phones, out_path, args, suffix=suffix)



