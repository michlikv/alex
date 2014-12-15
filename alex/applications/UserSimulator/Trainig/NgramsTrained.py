#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import pprint
from collections import Counter, defaultdict
from itertools import tee, islice
from TrainedStruct import TrainedStructure
from alex.components.slu.da import DialogueAct

class NgramsTrained(TrainedStructure):

    def __init__(self, n):
        self.structure_unigrams = defaultdict(int)
        self.structure = defaultdict(self.defdictint)
        self.ngram_n = n
        self.prefix = ['<s>']*(n-2) if (n-2) > 0 else []
        self.pp = pprint.PrettyPrinter(indent=4)

    def defdictint(self):
        return defaultdict(int)

    def train_counts(self, acts_list, class_type):
        bigr = list(NgramsTrained._ngrams(self.prefix+acts_list, self.ngram_n))
        bigr = Counter([gram for gram, pos in zip(bigr, range(0, len(bigr))) if pos % 2 == 0])
        # print bigr
        # ('cond')->'it'->count
        for ngram, count in bigr.iteritems():
            # print "|",ngram,"|",count,"|"
            self.structure[ngram[:-1]][ngram[-1]] += count
            # if not isinstance(ngram[0], class_type) or not isinstance(ngram[-1], class_type):
            #     print ngram[-1],"IS NOT DA!!!"
            # if ngram[-1] == u'tram' or ngram[0] == u'tram':
            #     print "TADY TADY TADY", ngram

            self.structure_unigrams[ngram[-1]] += count

    # returns list of possible reactions with its counts and total count as tuple
    # or null if history is unknown
    def get_possible_reactions(self, hist):
        if self.structure.get(hist, None):
            vals = self.structure[hist].values()
            return self.structure[hist].keys(), vals, sum(vals)
        else:
            return None

    def print_table_bigrams(self):
        #print self.pp.pprint(self.structure.keys())
        print self.pp.pprint(dict(self.structure[(unicode(DialogueAct('hello()')),)], sort_keys=False, indent=2))
        print 'asking for ', (unicode(DialogueAct('hello()')),)

    def print_table_unigrams(self):
        print self.pp.pprint(dict(self.structure_unigrams, sort_keys=False, indent=2))

    # returns list of possible reactions by unigram prob
    # or none if empty
    def get_possible_unigrams(self):
        if len(self.structure_unigrams.items()) != 0:
            vals = self.structure_unigrams.values()
            return self.structure_unigrams.keys(), vals, sum(vals)
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

#     # lst - list to count n-grams on
#     # @returns dictionary with frequencies
#     @staticmethod
#     def _bigrams(lst):
#         pairs = [(system,user) for system,user,pos in zip( lst,lst[1:], range(0,len(lst)) ) if pos%2==0 ]
#         # make counts from bigrams
#         return Counter(pairs)
# #        return { t:pairs.count(t) for t in set(pairs)}
