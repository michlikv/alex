#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from alex.components.slu.da import DialogueAct, DialogueActNBList
from copy import deepcopy

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.NgramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class NgramSimulatorFilterSlots(Simulator):

    def __init__(self, cfg):
        # todo N will be in  CFG?.
        self.cfg = cfg
        self.n = 2
        self.simulator = NgramsTrained(self.n)
        self.slotvals = NgramsTrained(2)

    def train_simulator(self, filename_filelist):
        list_of_files = FileReader.read_file(filename_filelist)
        self.simulator = NgramsTrained(self.n)

        for f in list_of_files:
            self.cfg['Logging']['system_logger'].info('Processing file: '+f)

            try:
                # read file to list
                dialogue = FileReader.read_file(f)
                if dialogue:
                    # create alternating user and system turns
                    dialogue = Preprocessing.prepare_conversations(dialogue,
                                                                   Preprocessing.create_act_from_stack_use_last,
                                                                   Preprocessing.create_act_from_stack_use_last)

                    #dialogue = Preprocessing.convert_string_to_dialogue_acts("")
                    dialogue = [DialogueAct(d) for d in dialogue]

                    Preprocessing.add_end_da(dialogue)
                    # save slot values
                    slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue)
                    # remove slot values
                    Preprocessing.remove_slot_values_from_dialogue(dialogue)

                    dialogue = [Preprocessing.shorten_connection_info(a) for a in dialogue]

                    self.simulator.train_counts(dialogue, DialogueAct)

                    self.slotvals.train_counts(slot_values, unicode)
                    # self.simulator.print_table_bigrams()
            except:
                self.cfg['Logging']['system_logger'].info('Error: '+f)

    def load_simulator(self, filename):
        self.simulator = NgramsTrained.load(filename)

    def save_simulator(self, filename):
        self.simulator.save(filename)

    def generate_response(self, system_da):
        # preprocess DA
        da_unicode = unicode(system_da)
        # da_unicode = Preprocessing.shorten_connection_info(da_unicode)
        Preprocessing.remove_slot_values(system_da)
        filtered = Preprocessing.shorten_connection_info(system_da)

        hist = (filtered,)
        nblist = DialogueActNBList()

        # print "generating:", hist
        reactions = self.simulator.get_possible_reactions(hist)

        # print "Possible reactions:", reactions
        if not reactions:
            reactions = self.simulator.get_possible_unigrams()
        else:
            print "DA found in history :-)", filtered

        response = RandomGenerator.generate_random_response(reactions[0], reactions[1], reactions[2])

        response = deepcopy(response)

        for dai in response.dais:
            if dai.value:
                possible_values = self.slotvals.get_possible_reactions((dai.name,))
                if not possible_values:
                    possible_values = self.slotvals.get_possible_unigrams()

                selected = RandomGenerator.generate_random_response(possible_values[0],
                                                                    possible_values[1],
                                                                    possible_values[2])
                dai.value = selected

        #response = DialogueAct(response)
        nblist.add(1.0, response)

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


