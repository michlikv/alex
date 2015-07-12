#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import abc


class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class ErrorModel(object):
    """Abstract class for user simulator."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, cfg):
        """Train structures."""
        raise NotImplementedError()

    @abc.abstractmethod
    def change_da(self, user_da):
        """Add errors to user DA, return n-best list.

            system_da: alex.components.slu.da.DialogueAct
            n-best list: alex.components.slu.da.DialogueActNBList
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, cfg):
        """Save trained model to file specified in cfg"""
        raise NotImplementedError()

    @abstractstatic
    def load(cfg):
        """  Load model from files specified in cfg"""
        raise NotImplementedError()
