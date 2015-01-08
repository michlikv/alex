#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import abc
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
        """Train structure from list of dialogue acts where
        user act and system act ...
        cond->it->prob"""
        #todo prubezne se stridaji.
        raise NotImplementedError()

    def save(self, filename):
        """Returns the pickle serialization of the object"""
        return pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        """Returns the instance of TrainedStructure from the pickle string"""
        return pickle.load(open(filename, 'rb'))


