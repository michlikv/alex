#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import abc

class Simulator(object):
    """Abstract class for user simulator."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def new_dialogue(self):
        """Start a new dialogue.
           Include cleaning of the structures (dialogue state) that depend on dialogue history.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_response_from_history(self, history):
        """Generate response to the system dialogue act in a form of n-best list.
           History of the dialogue is passed as parameter

            history: list of alex.components.slu.da.DialogueAct
            n-best list: alex.components.slu.da.DialogueActNBList
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_response(self, system_da):
        """Generate response to the system dialogue act in a form of n-best list.

            system_da: alex.components.slu.da.DialogueAct
            n-best list: alex.components.slu.da.DialogueActNBList
        """
        raise NotImplementedError()
