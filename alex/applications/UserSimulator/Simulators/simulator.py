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

class Simulator(object):
    """Abstract class for user simulator."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train_simulator(self, cfg):
        """
        Train simulator using setting from configuration file

        :param cfg: configuration
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def new_dialogue(self):
        """Start a new dialogue.
           Clean structures that depend on dialogue history.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_response_from_history(self, history):
        """Generate response from history of a dialogue in a form of n-best list.

        :param history: history of a dialogue
        :return: response
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_response(self, system_da):
        """Generate response to the system dialogue act in a form of n-best list.

        :param system_da: system action
        :return: response
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, cfg):
        """Save data of a simulator to file.

        :param cfg: configuration
        """
        raise NotImplementedError()

    @abstractstatic
    def load(cfg):
        """  Load simulator data from files specified in cfg

        :param cfg: configuration
        """
        raise NotImplementedError()
