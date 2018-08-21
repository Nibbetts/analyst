# Fasttext, Glove, DEPS, BNC
from tqdm import tqdm
import numpy as np
import re
import pickle as pkl
import sys


"""
    We take the ordered words from Fasttext, Glove, and DEPS embedding spaces,
(Fasttext and Glove are certainly in frequency order, and DEPS appears obviously
to be), as well as the words from Adam Kilgarriff's word frequency list from the
British National Corpus, World Edition, available at:
http://www.kilgarriff.co.uk/bnc-readme.html

Then, on each of these extracted word lists, we:
    1. Remove comments and any padding whitespace around words
    2. Convert all to lower case, keeping the first position in the ordering
        if multiple are found
    # 3. Remove any backslash characters from words, because otherwise some may
    #     filter through the regex later (ex: \' is treated as ' and gets through)
    # 4. On words of more than one character, if it starts or ends with a period
    #     or a slash, we remove those characters only. If the word is bookended
    #     with single or double quotes we also remove these.
    # 5. Throw out any words that are not composed only of lower-case letters,
    #     dashes, and apostrophies, and do not have at least one letter.
    #     Also remove any with -- or '', but keep '- or -'
    #     Note: we make the assumption that any valuable apostrophies were already
    #     converted from unicode, and we do not keep words containing unicode
    #     characters. We also drop words containing digits.
    ALTERNATIVE:
        Remove all tokens containing non-lower-cased letter characters.
        (including backslashes, checked for separately.)
    # Now each list's new indeces should be on comparable scales, since unicode
        and non-words have mostly been removed.
    6. Throw out words not found in 3 or more of our 4 lists, compiling a master
        list.
    7. Assign each word in the master list a value that is the average of its
        index in each list it is found in (with no weighting).
        Sort the words by this value, (We use default behavior for resolving
        conflicts,) and we use this list's words and ordering to select which
        words belong in each analyst.

    Some of these decisions on what to keep and what to throw out were made
because of our use case, which is focused on analogies, and analogical reasoning
applications in artificial agent navigation and decision-making.
"""


MAX_LINES = None#1000000
REQUIRED_IN = 3

#//////////////////////////////////////////////////////////////////////////////#

def read_text_table(path, firstline=True, limit_lines=None):
    lines = open(path, 'r', errors='ignore').readlines()
    if firstline:
        numvecs, dim = map(int, lines[0].split(" "))
    else:
        numvecs = len(lines) if limit_lines == None \
            else min(len(lines), limit_lines)
        dim = len(lines[0].split(" ")) - 1
    strings = []
    #embeddings = np.empty(shape=(numvecs, dim))
    for i in tqdm(range(numvecs), desc="Reading " + path):
        row = lines[i + firstline].split(" ")
        strings.append(row[0])#str(row[0]))
        #embeddings[i] = row[1:]
    return strings#, embeddings

def read_lexicon(path, limit_lines=None):
    lines = open(path, 'r', errors='ignore').readlines()[1:] # Remove header.
    if limit_lines is not None:
        lines = lines[:limit_lines]
    strings = []
    for l in tqdm(lines, desc="Reading " + path):
        row = [t.strip() for t in l.split()]
        if len(row) > 1:
            strings.append(row[1])
    return strings

#//////////////////////////////////////////////////////////////////////////////#

def clean_list(l):
    #p = re.compile("\A[a-z\-\']*[a-z]+[a-z\-\']*\Z") # Allow a-z, -, '
                                                     # with at least one letter.
    #q = re.compile("(\'\')|(\-\-)") # Throw out any with -- or ''
    p = re.compile("\A[a-z]+\Z")
    s = set()
    result = []
    for word in l:
        w = word.lower().strip() # For caps and spaces
        w = w.replace('\\','') # No special characters
        if len(w) > 1:
            if w[-1] == '.' or w[-1]  == '/': w = w[:-1]  # For abreviations
            if w[0]  == '.' or w[0]   == '/': w = w[1:]   # For websites
        if len(w) > 2:
            if w[0]  == '"' and w[-1] == '"': w = w[1:-1] # For quotes
            if w[0]  == "'" and w[-1] == "'": w = w[1:-1] # For quotes
        #if p.match(w) and w not in s and not q.match(w):
        if p.match(w) and w not in s:
            s.add(w)
            result.append(w)
    return result

#//////////////////////////////////////////////////////////////////////////////#

def fasttext_strings():
    import pickle as pkl

    with open("/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
            "fasttext.en.py2.pkl", 'rb') as f:
        data_ft = pkl.load(f)
        str_f = data_ft['tokens'][:MAX_LINES]
        return list(map(str, str_f))

def glove_strings():
    return read_text_table(
        "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
        "glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)

def deps_strings():
    return read_text_table(
        "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
        "dependency_based_word_embeddings/deps.words",
        firstline=False, limit_lines=MAX_LINES)

def bnc_strings():
    return read_lexicon(
        "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
        "british_national_corpus.num",
        limit_lines=MAX_LINES)

#//////////////////////////////////////////////////////////////////////////////#

functions = [
    fasttext_strings,
    glove_strings,
    deps_strings,
    bnc_strings,
]


if __name__ == "__main__":

    if len(sys.argv) > 1:
        REQUIRED_IN = int(sys.argv[1])

    print("Reading and Cleaning files")
    lists = [clean_list(f()) for f in functions]
    # sets = [set(l) for l in lists]
    orderings = {}
    freq = {}
    dropped = []
    for l in tqdm(lists, desc="Compiling Dictionary"):
        for i, w in enumerate(l):
            if w in orderings:
                orderings[w].append(i)
            else:
                orderings[w] = [i]
    for w, f in tqdm(orderings.items(), desc="Culling Dictionary"):
        if len(f) >= REQUIRED_IN: # Only keep ones that are common across
                                  # multiple lists
            freq[w] = np.mean(f) # Finding average position
        else:
            dropped.append((w, f))
    print("Zipping Items")
    words, infrequencies = zip(*freq.items())
    print("Sorting Words")
    words = np.array(words)[np.argsort(infrequencies)]
    if MAX_LINES is not None:
        words = words[:MAX_LINES]

    # Write files:
    print("Writing txt File")
    with open("/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
            "word_frequencies/clean_words_ordered_ALPHA_{}.txt".format(REQUIRED_IN), 'w') as f:
        for w in words:
            f.write(str(w) + '\n')
    print("Writing pkl File")
    pkl.dump(words, open("/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
        "word_frequencies/clean_words_ordered_ALPHA_{}.pkl".format(REQUIRED_IN), 'wb'))
    
    print(dropped)
    print()
    print("Kept:", len(words), "Dropped:", len(dropped))
