#!/usr/bin/env python
# encoding: utf8

import abc

class Simulator(object):
    """Abstract class for user simulator."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_response(self, system_da):
        """Generate response to the system dialogue act in a form of n-best list.

            system_da: alex.components.slu.da.DialogueAct
            n-best list: alex.components.slu.da.DialogueActNBList
        """
        raise NotImplementedError()
