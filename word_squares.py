import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import copy

from multiprocessing import Process, Queue, Manager
import time
import sys
import re


def add_word_to_tree(tree, word):
    if len(word) > 0:
        try:
            sub_tree = tree[word[0]]
        except:
            sub_tree = defaultdict(lambda: defaultdict())
        tree[word[0]] = add_word_to_tree(sub_tree, word[1:])
        return tree
    else:
        return defaultdict(lambda: defaultdict())


def get_loc_from_index(i, n):
    x = i % n
    y = int(i / n)
    return [x, y]

def get_index_from_loc(loc, n):
    x, y = loc
    return y * n + x

def get_possible_chars(sq, loc, char_tree):
    x, y = loc
    partial_word1 = sq[x, :y]
    partial_word2 = sq[:x, y]

    options1 = get_possible_chars_from_partial_word(partial_word1, char_tree)
    options2 = get_possible_chars_from_partial_word(partial_word2, char_tree)
    return  options1.intersection(options2)

def get_possible_chars_from_partial_word(partial_word, char_tree):
    t = copy.copy(char_tree)
    for char in partial_word:
        if char in t.keys():
            t = t[char]
        else:
            t = {}
    return set(t.keys())

def get_char_tree(n):
    char_tree = defaultdict(lambda: defaultdict())
    count = 0
    for word in words:
        if len(word) == n:
            count += 1
            char_tree = add_word_to_tree(char_tree, word)
    print(f'Made char tree with depth {n} from {count} words')
    char_tree['info'] = {'num_words':count, 'depth':n}
    return char_tree


class Square():
    def __init__(self, sq):
        self.sq = copy.copy(sq)
        self.symmetry_score = self.get_symmetry()
    
    def __eq__(self, other): 
        if np.all(self.sq == other.sq) or np.all(self.sq == other.sq.T): 
            return True
        else: 
            return False
    
    def __ge__(self, other):
        return (not __lt__(self, other) and not __eq__(self, other))

    def __lt__(self, other):
        return np.mean(self.sq < other.sq) > 0.5
    
    def __hash__(self):
        sq = self.sq if np.mean(self.sq.tostring() > self.sq.T.tostring()) > 0.5 else self.sq.T
        return hash(sq.tostring())
    
    def __str__(self):
        array_string = ''
        for line in self.sq:
            for char in line:
                array_string += f'{char} '
            array_string += f'\n'
        return array_string
    
    def get_symmetry(self):
        return np.mean(self.sq == self.sq.T)

def get_partial_squares(sq, limit=None):

    sq = copy.copy(sq)
    n = sq.shape[0]
    i = np.sum(sq != '')

    if limit is None:
        limit = n**2 - 1

    loc = get_loc_from_index(i, n)
    x, y = loc
    possible_chars = get_possible_chars(sq, loc, char_tree)

    sqs = []

    if len(possible_chars) == 0:
        return []

    for char in possible_chars:

        sq[x, y] = char

        if i < limit:
            sqs.extend(get_partial_squares(sq))
        else:
            sqs.extend([Square(sq)])
    
    return sqs


def get_squares(n):
    
    global char_tree
    char_tree = get_char_tree(n)
    
    sq = np.chararray((n, n))
    sq.fill('')
    sq = sq.astype('<U1')

    _start = time.time()

    sqs = get_partial_squares(sq)

    print("Took {0} seconds".format((time.time() - _start)))

    return sqs, char_tree

def get_unique_sqs(sqs):
    return list(set(sqs))

def print_sqs(sqs):
    for sq in sqs:
        print(sq)


def get_parallel_partial_squares(sq, done_list):

    sq = copy.copy(sq)
    n = sq.shape[0]
    i = np.sum(sq != '')

    loc = get_loc_from_index(i, n)
    x, y = loc
    possible_chars = get_possible_chars(sq, loc, char_tree)

    sqs = []

    if len(possible_chars) == 0:
        return []

    for char in possible_chars:
        sq[x, y] = char

        sqs = get_partial_squares(sq)
        done_list.extend(sqs)
    
    return done_list

def get_squares_parallel(n, max_processes = 80):
    with Manager() as manager:

        done_list = manager.list() 

        global char_tree
        char_tree = get_char_tree(n)
        
        sq = np.chararray((n, n))
        sq.fill('')
        sq = sq.astype('<U1')

        processes_level = 0
        possible_chars = get_possible_chars(sq, [0,0], char_tree)
        partial_sqs = get_partial_squares(sq, limit=processes_level)
        
        procs = []
        _start = time.time()

        print(f'Starting {len(partial_sqs)} processes')
        for sq in partial_sqs:
          
            # print(name)
            proc = Process(target=get_parallel_partial_squares, args=(sq.sq, done_list))
            proc.daemon = True
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

        print("Took {0} seconds".format((time.time() - _start)))
    
        return list(done_list), char_tree

def sqs2txt(sqs):
    txt = ''
    for i, sq in enumerate(sqs):
        txt += f'\n{i}:\n'
        txt += str(sq)
    return txt

def txt2sq(txt):
    rows = txt.split('\n')

    clean_rows = []
    for row in rows:
        if len(row) > 0:
            clean_row = row.split(' ')[:-1]
            if len(clean_row) > 0:
                clean_rows.append(clean_row)

    return Square(np.array(clean_rows))
    
def get_path_from_vars(sqs, words, n, note=''):
    if len(note) > 0:
        note = f'_note:{note}'
    name = f'repos/word_squares/squares/dict-len={len(words)}_n={n}_uniq={len(sqs)}{note}.txt'
    path = Path(Path.cwd(), name)
    return path

def save_sqs_from_path(sqs, path):
    print(f'Writting {len(sqs)} squares to: {path}')
    with path.open("w") as text_file:
        n = text_file.write(sqs2txt(sqs))

def load_sqs_from_file(path):
    with path.open("r") as text_file:
        sqs_string = text_file.read()

    loaded_sqs = []
    for sq_string in re.split('\n*.:', sqs_string):
        if len(sq_string) > 0:
            loaded_sqs.append(txt2sq(sq_string))
    print(f'Loaded {len(loaded_sqs)} Squares from: {path}')
    return loaded_sqs

def save_with_load_check(sqs_to_write, words, n, note=''):
    path = get_path_from_vars(sqs_to_write, words, n, note=note)
    save_sqs_from_path(sqs_to_write, path)
    loaded_sqs = load_sqs_from_file(path)

    assert(loaded_sqs == sqs_to_write)
    print('Load check succesful!')

# def get_partial_squares(sq):
#     sq = copy.copy(sq)
#     i = np.sum(sq != '')
#     n = sq.shape[0]
#     loc = get_loc_from_index(i, n)
#     x, y = loc
#     possible_chars = get_possible_chars(sq, loc, char_tree)

#     sqs = []

#     if len(possible_chars) == 0:
#         return []

#     for char in possible_chars:
#         sq[x, y] = char

#         if i < n**2 - 1:
#             sqs.extend(get_partial_squares(sq))
#         else:
#             sqs.extend([Square(sq)])

#     return sqs
