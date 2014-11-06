#!/usr/bin/env python
# encoding: utf8

from simulator import Simulator
from alex.components.slu.da import DialogueAct, DialogueActNBList


class Simple_bigram_simulator(Simulator):




    def generate_response(self, system_da):
        #TODO this should be in user simulators
        nblist = DialogueActNBList()
        nblist.add(prob, da)

        #TODO find wth that means :-O
        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


