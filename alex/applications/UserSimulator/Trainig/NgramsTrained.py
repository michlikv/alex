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
    """
    Implementation of n-gram model
    """

    def __init__(self, n):
        self._structure_unigrams = defaultdict(int)
        self._structure = defaultdict(lambda: defaultdict(int))
        self._ngram_n = n
        self._prefix = ['<s>']*(n-2) if (n-2) > 0 else []
        self._pp = pprint.PrettyPrinter(indent=4)

    def train_counts(self, dialogue):
        """Add counts of n-grams from the dialogue to structures

        :param dialogue: list of system and user actions
        """
        bigr = list(NgramsTrained._ngrams(self._prefix+dialogue, self._ngram_n))
        bigr = Counter([gram for pos, gram in enumerate(bigr) if pos % 2 == 0])
        # print bigr
        # ('cond')->'it'->count
        for ngram, count in bigr.iteritems():
            self._structure[ngram[:-1]][ngram[-1]] += count
            self._structure_unigrams[ngram[-1]] += count

    def get_possible_reactions(self, hist):
        """Find history in the structure and return possible reactions with counts

        :param hist: history
        :return: list of possible reactions with its counts and total count, or None if history is unknown.
        """
        if self._structure.get(hist, None):
            keys = []
            vals = []
            for k, v in self._structure[hist].iteritems():
                keys.append(k)
                vals.append(v)
            return keys, vals, sum(vals)
        else:
            return None, [], []

    def print_table_bigrams(self):
        print self._pp.pprint(dict(self._structure[(unicode(DialogueAct('hello()')),)], sort_keys=False, indent=2))
        print 'asking for ', (unicode(DialogueAct('hello()')),)

    def print_table_unigrams(self):
        print self._pp.pprint(dict(self._structure_unigrams, sort_keys=False, indent=2))

    def get_possible_unigrams(self):
        """ Return all unigrams with counts

        :return: list of possible reactions with unigram counts
        """
        if len(self._structure_unigrams.items()) != 0:
            keys = []
            vals = []
            for k, v in self._structure_unigrams.iteritems():
                keys.append(k)
                vals.append(v)
            return keys, vals, sum(vals)
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
        """Saves object to file

        :param filename: name of file
        """

        self._structure = dict(self._structure)
        out = open(filename, 'wb')
        pickle.dump(self._structure_unigrams, out)
        pickle.dump(self._structure, out)
        pickle.dump(self._ngram_n, out)
        pickle.dump(self._prefix, out)
        out.close()

    @staticmethod
    def load(filename):
        """Load object from file

        :param filename: name of file
        """
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
