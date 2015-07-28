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
    """Abstract class for error models."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, cfg):
        """Train structures.

           :param cfg: training configuration
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def change_da(self, user_da):
        """Add errors to user DA, return n-best list.

            :param user_da: intended user response
            :type user_da: alex.components.slu.da.DialogueAct
            :rtype: alex.components.slu.da.DialogueActNBList
            :returns: n-best list of acts with added noise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, cfg):
        """Save trained model to file specified in cfg

           :param cfg: configuration
        """
        raise NotImplementedError()

    @abstractstatic
    def load(cfg):
        """  Load model from files specified in cfg

           :param cfg: configuration
        """
        raise NotImplementedError()
