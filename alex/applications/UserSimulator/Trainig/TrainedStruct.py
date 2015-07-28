#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import abc
try:
    import cPickle as pickle
except:
    import pickle

class TrainedStructure(object):
    """Abstract class for a trained structure)."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_possible_reactions(self, hist):
        """Find history in the structure and return possible reactions with counts

        :param hist: history
        :return: list of possible reactions with its counts and total count, or None if history is unknown.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_possible_unigrams(self):
        """ Return all unigrams with counts

        :return: list of possible reactions with unigram counts
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_counts(self, acts_list):
        """Add counts of n-grams from the dialogue to structures

        :param dialogue: list of system and user actions
        """
        raise NotImplementedError()

    def save(self, filename):
        """Saves object to file

        :param filename: name of file
        """
        out = open(filename, 'wb')
        pickle.dump(self, out)
        out.close()

    @staticmethod
    def load(filename):
        """Load object from file

        :param filename: name of file
        """
        input = open(filename, 'rb')
        obj = pickle.load(input)
        input.close()
        return obj


