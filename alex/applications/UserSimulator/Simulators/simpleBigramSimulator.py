#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import random
from alex.components.slu.da import DialogueAct, DialogueActNBList

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.BigramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class SimpleBigramSimulator(Simulator):

    def __init__(self):
        random.seed(19910604)
        self.simulator = NgramsTrained(2)

    def train_simulator(self, filename_filelist):
        list_of_files = FileReader.read_file(filename_filelist)
        self.simulator = NgramsTrained(2)

        for file in list_of_files:
            print "processing file", file
            dialogue = Preprocessing.filter_acts_one_only(FileReader.read_file(file))
            self.simulator.train_counts(dialogue)

        # self.simulator.print_table_bigrams()

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
        nblist.add(1.0,response)
      #  nblist.add(0.9, DialogueAct('hangup()'))

        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


