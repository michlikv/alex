#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import pprint
from collections import Counter, defaultdict
from itertools import tee, islice
from TrainedStruct import TrainedStructure
from alex.components.slu.da import DialogueAct
try:
    import cPickle as pickle
except:
    import pickle


class NgramsTrained(TrainedStructure):

    def __init__(self, n):
        #TrainedStructure.__init__(self)
        self._structure_unigrams = defaultdict(int)
        self._structure = defaultdict(lambda: defaultdict(int))
        self._ngram_n = n
        self._prefix = ['<s>']*(n-2) if (n-2) > 0 else []
        self._pp = pprint.PrettyPrinter(indent=4)

    def train_counts(self, acts_list, class_type):
        bigr = list(NgramsTrained._ngrams(self._prefix+acts_list, self._ngram_n))
        bigr = Counter([gram for pos, gram in enumerate(bigr) if pos % 2 == 0])
        # print bigr
        # ('cond')->'it'->count
        for ngram, count in bigr.iteritems():
            # print "|",ngram,"|",count,"|"
            self._structure[ngram[:-1]][ngram[-1]] += count
            self._structure_unigrams[ngram[-1]] += count

    # returns list of possible reactions with its counts and total count as tuple
    # or null if history is unknown
    def get_possible_reactions(self, hist):
        if self._structure.get(hist, None):
            vals = self._structure[hist].values()
            return self._structure[hist].keys(), vals, sum(vals)
        else:
            return None

    def print_table_bigrams(self):
        #print self.pp.pprint(self.structure.keys())
        print self._pp.pprint(dict(self._structure[(unicode(DialogueAct('hello()')),)], sort_keys=False, indent=2))
        print 'asking for ', (unicode(DialogueAct('hello()')),)

    def print_table_unigrams(self):
        print self._pp.pprint(dict(self._structure_unigrams, sort_keys=False, indent=2))

    # returns list of possible reactions by unigram prob
    # or none if empty
    def get_possible_unigrams(self):
        if len(self._structure_unigrams.items()) != 0:
            vals = self._structure_unigrams.values()
            return self._structure_unigrams.keys(), vals, sum(vals)
        else:
            return None

    @staticmethod
    def _ngrams(lst, n):
        tlst = lst
        while True:
            a, b = tee(tlst)
            l = tuple(islice(a, n))
            if len(l) == n:
                yield l
                next(b)
                tlst = b
            else:
                break

    def save(self, filename):
        """Saves object to file"""
        self._structure = dict(self._structure)
        out = open(filename, 'wb')
        pickle.dump(self._structure_unigrams, out)
        pickle.dump(self._structure, out)
        pickle.dump(self._ngram_n, out)
        pickle.dump(self._prefix, out)
        out.close()

    @staticmethod
    def load(filename):
        """Returns the instance of TrainedStructure from the pickle string"""
        input = open(filename, 'rb')
        obj = NgramsTrained(2)
        obj._structure_unigrams = pickle.load(input)
        structure = pickle.load(input)
        obj._ngram_n = pickle.load(input)
        obj._prefix = pickle.load(input)
        input.close()
        obj._structure = defaultdict(lambda: defaultdict(int))
        obj._structure.update(structure)

        return obj
