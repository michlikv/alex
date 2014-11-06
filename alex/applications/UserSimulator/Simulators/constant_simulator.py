#!/usr/bin/env python
# encoding: utf8

import random

from simulator import Simulator
from alex.components.slu.da import DialogueAct, DialogueActNBList

class Constant_simulator(Simulator):

    def __init__(self):
        random.seed(19910604)

    def generate_response(self, system_da):
        nblist = DialogueActNBList()
        nblist.add(1.0, DialogueAct('silence()'))

        # randomly generate silence or hangup :)
        if random.randint(0, 4) == 0:
            nblist.add(1.0, DialogueAct('hangup()'))
        else:
            nblist.add(1.0, DialogueAct('silence()'))

        nblist.merge()
        nblist.scale()

   #     print (unicode(nblist.get_confnet().get_best_da()))

        return nblist.get_confnet()


