#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.components.slu.common import slu_factory
import re

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.NgramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class NgramSimulatorFilterSlots(Simulator):

    def __init__(self, cfg):
        #todo N will be in  CFG?.
        self.n = 2
        self.simulator = NgramsTrained(self.n)
        self.slotvals = NgramsTrained(2)
  #      self.slu = slu_factory(cfg)

    def train_simulator(self, filename_filelist):
        list_of_files = FileReader.read_file(filename_filelist)
        self.simulator = NgramsTrained(self.n)

        for file in list_of_files:
            print "processing file", file

            # read file to list
            dialogue = FileReader.read_file(file)
            # create alternating user and system turns
            dialogue = Preprocessing.filter_acts_one_only(dialogue)

            #todo convert every system turn do dialogue act using SLU module
            #dialogue = Preprocessing.convert_string_to_dialogue_acts("")

            # save slot values
            slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue)
            # remove slot values
            dialogue = Preprocessing.remove_slot_values_from_dialogue(dialogue)
            #todo not implemented yet, merge connection info to one dialogue act givingConnectionInfo()
            #todo bacha tohle bere jenom jeden DA, musi se tim protahnout cely dialog
            #dialogue = Preprocessing.shorten_connection_info(dialogue)

            self.simulator.train_counts(dialogue)
            self.slotvals.train_counts(slot_values)
        # self.simulator.print_table_bigrams()

    def load_simulator(self, filename):
        self.simulator = NgramsTrained.load(filename)

    def save_simulator(self, filename):
        self.simulator.save(filename)

    def generate_response(self, system_da):
        # preprocess DA
        da_unicode = unicode(system_da)
        #da_unicode = Preprocessing.shorten_connection_info(da_unicode)
        da_unicode = Preprocessing.remove_slot_values(da_unicode)

        hist = (da_unicode,)
        nblist = DialogueActNBList()

        # print "generating:", hist
        reactions = self.simulator.get_possible_reactions(hist)

        # print "Possible reactions:", reactions
        if not reactions:
            reactions = self.simulator.get_possible_unigrams()

        response = RandomGenerator.generate_random_response(
            reactions[0], reactions[1], reactions[2])

        da_response = DialogueAct(response)
        slots = da_response.get_slots_and_values()

        #todo add slot values check this.
        #todo could add according to a GOAL
        for name, value in slots:
            possible_values = self.slotvals.get_possible_reactions( (name,) )
            if not possible_values:
                # TODO what IF there IS NO VALUE FOR THE SLOT
                possible_values = self.slotvals.get_possible_unigrams()
            selected = RandomGenerator.generate_random_response(possible_values[0],possible_values[1],possible_values[2])
            response = re.sub(name+'=""', name+'="'+selected+'"', response)

        response = DialogueAct(response)
        nblist.add(1.0, response)

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


