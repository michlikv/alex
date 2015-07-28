#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from alex.components.slu.da import DialogueAct, DialogueActNBList

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.NgramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class RandomSimulator(Simulator):
    """
    Implementation of random simulator.
    It generates a response from the training dataset with uniform probability.
    """

    def new_dialogue(self):
        pass

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 2
        self.simulator = NgramsTrained(self.n)

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


    def get_oov(self):
        return 0.0

    @staticmethod
    def load(cfg):
        sim = RandomSimulator(cfg)
        sim.simulator = NgramsTrained.load(cfg['UserSimulation']['files']['model'])
        return sim

    def save(self, cfg):
        self.simulator.save(cfg['UserSimulation']['files']['model'])

    def generate_response_from_history(self, history):
        return self.generate_response(history[-1])

    def generate_response(self, system_da):
        hist = (unicode(system_da),)
        nblist = DialogueActNBList()

        reactions = self.simulator.get_possible_unigrams()

        response = DialogueAct(RandomGenerator.generate_random_response_uniform(reactions[0]))
        nblist.add(1.0, response)

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


