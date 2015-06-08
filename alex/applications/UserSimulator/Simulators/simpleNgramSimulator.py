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

    def new_dialogue(self):
        pass

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 2
        self.simulator = NgramsTrained(self.n)
        #self.slu = slu_factory(cfg)

        self.uniform_counter = 0
        self.found_counter = 0

    def train_simulator(self, cfg):
        list_of_files = FileReader.read_file(cfg['UserSimulation']['files']['training-data'])
        self.simulator = NgramsTrained(self.n)

        for file in list_of_files:
            self.cfg['Logging']['system_logger'].info("processing file: "+ file)
            try:
                # read file to list
                dialogue = FileReader.read_file(file)
                if dialogue:
                    dialogue = Preprocessing.prepare_conversations(dialogue,
                                                                   Preprocessing.create_act_from_stack_use_last,
                                                                   Preprocessing.create_act_from_stack_use_last)
                    Preprocessing.add_end_string(dialogue)
                    Preprocessing.clear_numerics(dialogue)
                    self.simulator.train_counts(dialogue)
            except:
                self.cfg['Logging']['system_logger'].info('Error: '+file)
                raise


    # def train_simulator_using_slu(self,filename_filelist, slu):
    #     list_of_files = FileReader.read_file(filename_filelist)
    #     self.simulator = NgramsTrained(2)
    #
    #     for file in list_of_files:
    #         print "processing file", file
    #         dialogue = Preprocessing.prepare_conversations(FileReader.read_file(file))
    #         self.simulator.train_counts(dialogue)
    #     # self.simulator.print_table_bigrams()

    @staticmethod
    def load(cfg):
        sim = SimpleNgramSimulator(cfg)
        sim.simulator = NgramsTrained.load(cfg['UserSimulation']['files']['model'])
        return sim

    def save(self, cfg):
        self.simulator.save(cfg['UserSimulation']['files']['model'])

    def get_oov(self):
        return self.uniform_counter/(self.found_counter+self.uniform_counter+0.0)

    def generate_response_from_history(self, history):
        return self.generate_response(history[-1])

    def generate_response(self, system_da):
        hist = (unicode(system_da),)
        nblist = DialogueActNBList()

        # print "generating:", hist
        reactions = self.simulator.get_possible_reactions(hist)
        # print "Possible reactions:", reactions

        if not reactions[0]:
            reactions = self.simulator.get_possible_unigrams()
            self.uniform_counter += 1
        else:
            self.found_counter += 1

        response = DialogueAct(RandomGenerator.generate_random_response(
                               reactions[0], reactions[1], reactions[2]))
        nblist.add(1.0, response)

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


