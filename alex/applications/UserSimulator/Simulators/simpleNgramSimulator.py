#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import random
from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.components.slu.common import slu_factory

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.NgramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class SimpleNgramSimulator(Simulator):

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 2
        self.simulator = NgramsTrained(self.n)
        #self.slu = slu_factory(cfg)

    def train_simulator(self, filename_filelist):
        list_of_files = FileReader.read_file(filename_filelist)
        self.simulator = NgramsTrained(self.n)

        for file in list_of_files:
            print "processing file", file
            try:
                dialogue = Preprocessing.prepare_conversations(FileReader.read_file(file),
                                                               Preprocessing.create_act_from_stack_use_last,
                                                               Preprocessing.create_act_from_stack_use_last)
                Preprocessing.add_end_string(dialogue)
            except:
                self.cfg['Logging']['system_logger'].info('Error: '+file)
            self.simulator.train_counts(dialogue)


    # def train_simulator_using_slu(self,filename_filelist, slu):
    #     list_of_files = FileReader.read_file(filename_filelist)
    #     self.simulator = NgramsTrained(2)
    #
    #     for file in list_of_files:
    #         print "processing file", file
    #         dialogue = Preprocessing.prepare_conversations(FileReader.read_file(file))
    #         self.simulator.train_counts(dialogue)
    #     # self.simulator.print_table_bigrams()

    def load_simulator(self, filename):
        self.simulator = NgramsTrained.load(filename)

    def save_simulator(self, filename):
        self.simulator.save(filename)

    def generate_response(self, system_da):
        hist = (unicode(system_da),)
        nblist = DialogueActNBList()

        # print "generating:", hist
        reactions = self.simulator.get_possible_reactions(hist)
        # print "Possible reactions:", reactions

        if not reactions:
            reactions = self.simulator.get_possible_unigrams()

        response = DialogueAct(RandomGenerator.generate_random_response(
                               reactions[0], reactions[1], reactions[2]))
        nblist.add(1.0, response)

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


