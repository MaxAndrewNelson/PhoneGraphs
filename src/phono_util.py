from collections import defaultdict
from itertools import permutations, product

def get_projections(proj_file):

    if proj_file == None:
        return None

    else:

        f = open(proj_file, "r")

        tiers = {}

        for line in f:
            line = line.strip().split('\t')
            name = line[0]

            definition = line[1].split(':')
            trigger = definition[0].strip()
            features = [x.strip() for x in definition[1].split(',')]

            tiers[name] = {trigger:features}

        f.close()

        return tiers

def read_phone_file(phone_file):
    '''
    returns a list of all phonemes in a phone file
    '''
    f = open(phone_file, 'r')

    return f.readline().strip().split()

def get_phones(feature_file):
    '''
    returns a list of all phonemes in a feature file
    '''
    f = open(feature_file, "r")
    header = f.readline()

    phones = []
    for line in f:
        phones.append(line.split('\t')[0])

    f.close()

    return phones

def convert_to_ng(n, forms):
    '''
    Inputs

        forms : a list of words
        n : an ngram size

    Returns
    
        attested : list of attested n grams
        freqs : unnormalized frequencies of those n grams

    '''

    gram = defaultdict(int)

    for form in forms:
        form = [x for x in form.split('.') if x != '']

        for i in range(len(form)-n+1):
            curr_gram = '.'.join(form[i:i+n])
            gram[curr_gram] += 1

    attested = []
    freqs = []
    for k in gram:
        attested.append(k)
        freqs.append(gram[k])

    return attested, freqs

def get_unattested(attested, phones, n):
    '''
    Inputs

        attested : list of all attested ngrams
        phones : list of all phonemes in the language

    Returns

        unattested : list of all unattested ngrams
    '''

    unattested = []
    ngrams = [p for p in product(phones, repeat=n)]
    # ngrams = permutations(phones, n)
    for ng in ngrams:
        ng = '.'.join(ng)
        if ng not in attested:
            unattested.append(ng)

    return unattested


