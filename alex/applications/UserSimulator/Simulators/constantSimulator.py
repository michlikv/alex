#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import random
from simulator import Simulator
from alex.components.slu.da import DialogueAct, DialogueActNBList

class ConstantSimulator(Simulator):

    def __init__(self, cfg):
        random.seed(19910604)

    @staticmethod
    def load(cfg):
        return ConstantSimulator(cfg)

    def generate_response(self, system_da):
        nblist = DialogueActNBList()

        # randomly generate silence or hangup :)
        if random.randint(0, 4) == 0:
            nblist.add(1.0, DialogueAct('hangup()'))
        else:
            nblist.add(1.0, DialogueAct('silence()'))

        nblist.merge()
        nblist.scale()

   #     print (unicode(nblist.get_confnet().get_best_da()))

        return nblist.get_confnet()

    def new_dialogue(self):
        pass

    def generate_response_from_history(self, history):
        pass

    def train_simulator(self, cfg):
        pass

    def save(self, cfg):
        pass
