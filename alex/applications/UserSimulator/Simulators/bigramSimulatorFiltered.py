#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from alex.components.slu.da import DialogueAct, DialogueActNBList
from copy import deepcopy
from alex.components.dm import Ontology

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Trainig.NgramsTrained import NgramsTrained
from Generators.randomGenerator import RandomGenerator


class BigramSimulatorFiltered(Simulator):
    """Implementation of filtered bigram simulator.
       The simulator shortens connection information in system actions and
       uses frame dialogue acts: it generates the response without values
       and the values are generated separately.
    """

    def new_dialogue(self):
        pass

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 2
        self.simulator = NgramsTrained(self.n)
        self.slotvals = NgramsTrained(2)
        self.ontology = Ontology(cfg['UserSimulation']['ontology'])

        self.uniform_counter = 0
        self.found_counter = 0

    def train_simulator(self, cfg):
        list_of_files = FileReader.read_file(cfg['UserSimulation']['files']['training-data'])
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
                    Preprocessing.clear_numerics(dialogue)
                    dialogue = [DialogueAct(d) for d in dialogue]
                    Preprocessing.add_end_da(dialogue)
                    # save slot values
                    slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue,
                                                                                         ignore_values=['none', '*'])

                    for i,da in enumerate(dialogue):
                        for dai in da:
                            if dai.name and dai.name == 'vehicle':
                                break
                            if dai.name and dai.name == 'walk_to':
                                pass

                    # remove slot values
                    Preprocessing.remove_slot_values_from_dialogue(dialogue)
                    dialogue = [Preprocessing.shorten_connection_info(a) if i % 2 == 0 else a for i, a in enumerate(dialogue)]

                    self.simulator.train_counts(dialogue)
                    self.slotvals.train_counts(slot_values)
            except:
                self.cfg['Logging']['system_logger'].info('Error: '+f)
                raise

    @staticmethod
    def load(cfg):
        sim = BigramSimulatorFiltered(cfg)
        sim.simulator = NgramsTrained.load(cfg['UserSimulation']['files']['model'])
        sim.slotvals = NgramsTrained.load(cfg['UserSimulation']['files']['slotvals'])
        return sim

    def save(self, cfg):
        self.simulator.save(cfg['UserSimulation']['files']['model'])
        self.slotvals.save(cfg['UserSimulation']['files']['slotvals'])

    def get_oov(self):
        """Return percentage of out of domain actions in a generation process.

           :return: percentage of ood
           :rtype: float
        """
        return self.uniform_counter/(self.found_counter+self.uniform_counter+0.0)

    def generate_response_from_history(self, history):
        return self.generate_response(history[-1])

    def generate_response(self, system_da):
        # preprocess DA
        da_unicode = unicode(system_da)
        Preprocessing.remove_slot_values(system_da)
        filtered = Preprocessing.shorten_connection_info(system_da)

        hist = (filtered,)
        nblist = DialogueActNBList()

        reactions = self.simulator.get_possible_reactions(hist)

        if not reactions[0]:
            reactions = self.simulator.get_possible_unigrams()
            self.uniform_counter += 1
        else:
            self.found_counter += 1

        response = RandomGenerator.generate_random_response(reactions[0], reactions[1], reactions[2])

        response = deepcopy(response)
        new_resp = DialogueAct()

        for dai in response.dais:
            if dai.value:
                if "city" in dai.name or "stop" in dai.name:
                    possible_values = list(self.ontology['slots'][dai.name])
                    if not possible_values:
                        #possible_values = self.slotvals.get_possible_unigrams()
                        print "No SLOT VALUE FOR SLOT NAME:", dai.name
                        raise
                    else:
                        selected = RandomGenerator.generate_random_response_uniform(possible_values)
                else:
                    possible_values = self.slotvals.get_possible_reactions((dai.name,))
                    if not possible_values:
                        #possible_values = self.slotvals.get_possible_unigrams()
                        print "No SLOT VALUE FOR SLOT NAME:", dai.name
                        raise
                    else:
                        selected = RandomGenerator.generate_random_response(possible_values[0],
                                                                        possible_values[1],
                                                                        possible_values[2])
                dai.value = selected
            new_resp.append(dai)
        if len(new_resp) == 0:
            new_resp = DialogueAct('null()')
        response = new_resp

        #response = DialogueAct(response)
        nblist.add(1.0, response)
        nblist.merge()
        nblist.scale()
        return nblist.get_confnet()


