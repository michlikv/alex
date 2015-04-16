#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import abc
try:
    import cPickle as pickle
except:
    import pickle

class TrainedStructure(object):
    """Abstract class for user simulator."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_possible_reactions(self, hist):
        raise NotImplementedError

    @abc.abstractmethod
    def get_possible_unigrams(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_counts(self, acts_list):
        """Train structure from list of consecutive user and system dialogue acts"""
        raise NotImplementedError()

    def save(self, filename):
        """Saves object to file"""
        out = open(filename, 'wb')
        pickle.dump(self, out)
        out.close()

    @staticmethod
    def load(filename):
        """Returns the instance of TrainedStructure from the pickle string"""
        input = open(filename, 'rb')
        obj = pickle.load(input)
        input.close()
        return obj


