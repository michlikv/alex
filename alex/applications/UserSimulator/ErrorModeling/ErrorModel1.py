#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

from collections import defaultdict
from copy import deepcopy
import os
import re
try:
    import cPickle as pickle
except:
    import pickle

from alex.components.dm import Ontology
from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActItem

from errormodel import ErrorModel
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from Generators.randomGenerator import RandomGenerator
from Trainig.NgramsTrained import NgramsTrained


class ErrorModel1(ErrorModel):

    def __init__(self, cfg):
        self.cfg = cfg
        self.ontology = Ontology(cfg['ErrorModel']['ontology'])
        self.present_dai_struct = defaultdict(lambda: defaultdict(int))
        self.absent_dai_struct = defaultdict(lambda: defaultdict(int))
        self.changed_val_dai_struct = defaultdict(lambda: defaultdict(int))
        self.scores = []

        self.slotvals = NgramsTrained(2)
        self.turns = 0
        self.slot_frames = set()

    def train(self, cfg):
        list_of_files = FileReader.read_file(cfg['ErrorModel']['files']['training-data'])

        for filename in list_of_files:
            self.cfg['Logging']['system_logger'].info("processing file: " + filename)
            try:
                #split
                file_real, file_messed = filename.split("\t",2)

                # read dialogues to list
                dialogue_real = self._get_dialogue_from_file(file_real)
                dialogue_messed = self._get_dialogue_from_file(file_messed)

                if len(dialogue_messed) == len(dialogue_real):
                    #learn slot values from real dialogue (there will be better values)
                    slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue_real,
                                                                                         ignore_values=['none', '*'])
                    self.slotvals.train_counts(slot_values)
                    # learn edits
                    for i, (da_r, da_m) in enumerate(zip(dialogue_real, dialogue_messed)):
                        if i % 2 == 1:
                            self._compare_das(da_r, da_m)
                            self.turns += 1
                else:
                    raise

            except Exception, e:
                self.cfg['Logging']['system_logger'].info('Error: '+filename)
                self.cfg['Logging']['system_logger'].info(e)
                raise e
        #update "missing" DAIS structures
        for name, val in self.absent_dai_struct.iteritems():
            self.absent_dai_struct[name]['N'] = self.turns - val['Y']

        self.scores = FileReader.read_file(cfg['ErrorModel']['files']['training-nums'])

        self.slot_frames.update(self.changed_val_dai_struct.keys())
        self.slot_frames.update(self.absent_dai_struct.keys())
        self.slot_frames.update(self.present_dai_struct.keys())

    def _compare_das(self, real_da, messed_da, exclude=['null()', 'other()', 'silence()', 'None()']):
        """Matching of the DAIs, updating trained structures.

        :param real_da: real intended DA
        :type real_da: alex.components.slu.da.DialogueAct
        :param messed_da: an incorrect DA
        :type messed_da: alex.components.slu.da.DialogueAct
        :param exclude: list of DAIs to exclude
        :type exclude: list(str)
        """

        copy_messed = set()
        for dai in messed_da:
            copy_messed.add(dai)

        for dai_r in real_da.dais:
            found = False
            if unicode(dai_r) in exclude:
                break
            for dai_m in copy_messed:
                # act share same type
                if dai_r.dat == dai_m.dat:
                    #act share same slot name --> found match!
                    if dai_m.name and dai_r.name and dai_m.name == dai_r.name:
                        if dai_r.value and dai_m.value:
                            val = dai_m.value
                            dai_m.value = "&"
                            self.present_dai_struct[unicode(dai_m)]['Y'] += 1
                            if dai_r.value and val and dai_r.value.lower() == val.lower():
                                self.changed_val_dai_struct[unicode(dai_m)]['N'] += 1
                            else:
                                self.changed_val_dai_struct[unicode(dai_m)]['Y'] += 1
                            dai_m.value = val
                        else:
                            self.present_dai_struct[unicode(dai_m)]['Y'] += 1
                        found = True
                        copy_messed.remove(dai_m)
                        break
                    # if messed has same value but has no slot name, we suppose it matched
                    elif not dai_m.name and dai_r.value and dai_m.value and dai_r.value == dai_m.value:
                        val = dai_r.value
                        dai_r.value = "&"
                        self.present_dai_struct[unicode(dai_r)]['Y'] += 1
                        self.changed_val_dai_struct[unicode(dai_r)]['N'] += 1
                        dai_r.value = val
                        found = True
                        copy_messed.remove(dai_m)
                        break
                    # if has no value but same name, or no value and no name
                    elif ((not dai_r.name and not dai_m.name and not dai_r.value and not dai_m.value) or
                         (not dai_r.value and not dai_m.value and dai_r.name and dai_m.name and dai_r.name == dai_m.name)):
                        self.present_dai_struct[unicode(dai_m)]['Y'] += 1
                        found = True
                        copy_messed.remove(dai_m)
                        break
            if not found:
                if dai_r.value:
                    val = dai_r.value
                    dai_r.value = "&"
                self.present_dai_struct[unicode(dai_r)]['N'] += 1
                if dai_r.value:
                    dai_r.value = val
        for dai_m in copy_messed:
            # we exclude variant dat(=value)
            if not (not dai_m.name and dai_m.value) and unicode(dai_m) not in exclude:
                if dai_m.value:
                    dai_m.value = '&'
                self.absent_dai_struct[unicode(dai_m)]['Y'] += 1

    def _get_dialogue_from_file(self, filename):
        """ Read and preprocess dialogue from file

           :param filename: name of a file with dialogue
           :type filename: str
           :return: list of alsterning system and user DAs
        """
        dialogue = FileReader.read_file(filename)
        if dialogue:
            dialogue = Preprocessing.prepare_conversations(dialogue,
                                                           Preprocessing.create_act_from_stack_use_last,
                                                           Preprocessing.create_act_from_stack_use_last)
            Preprocessing.clear_numerics(dialogue)
            dialogue = [re.sub(r"\)([^&])", r")&\1", a) for a in dialogue]
        return [DialogueAct(a) for a in dialogue]

    @staticmethod
    def load(cfg):
        em = ErrorModel1(cfg)
        filename = cfg['ErrorModel']['files']['model']
        input = open(filename, 'rb')
        present = pickle.load(input)
        absent = pickle.load(input)
        changed = pickle.load(input)
        em.slot_frames = pickle.load(input)
        em.scores = pickle.load(input)
        em.turns = pickle.load(input)
        input.close()

        em.present_dai_struct = defaultdict(lambda: defaultdict(int))
        em.present_dai_struct.update(present)
        em.absent_dai_struct = defaultdict(lambda: defaultdict(int))
        em.absent_dai_struct.update(absent)
        em.changed_val_dai_struct = defaultdict(lambda: defaultdict(int))
        em.changed_val_dai_struct.update(changed)

        em.slotvals = NgramsTrained.load(cfg['ErrorModel']['files']['slotvals'])
        return em

    def save(self, cfg):
        filename = cfg['ErrorModel']['files']['model']
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.present_dai_struct = dict(self.present_dai_struct)
        self.absent_dai_struct = dict(self.absent_dai_struct)
        self.changed_val_dai_struct = dict(self.changed_val_dai_struct)

        out = open(filename, 'wb')
        pickle.dump(self.present_dai_struct, out)
        pickle.dump(self.absent_dai_struct, out)
        pickle.dump(self.changed_val_dai_struct, out)
        pickle.dump(self.slot_frames, out)
        pickle.dump(self.scores, out)
        pickle.dump(self.turns, out)
        out.close()

        self.slotvals.save(cfg['ErrorModel']['files']['slotvals'])
        print "."

    def _fill_random_value(self, dai):
        """ fill in the DAI a new random value.

           :param dai: DAI from new noisy DA
           :type dai: alex.components.slu.da.DialogueActItem
        """

        if dai.value and dai.value == "&":
            #generate uniform from compatible values
            if "city" in dai.name or "stop" in dai.name:
                possible_values = list(self.ontology['slots'][dai.name])
            else:
                possible_values, v, s = self.slotvals.get_possible_reactions((dai.name,))
                if not possible_values:
                    raise ValueError('No slot value for slot name:', dai.name)

            selected = RandomGenerator.generate_random_response_uniform(possible_values)
            dai.value = selected

    def _fill_correct_value(self, real_da, new_dai):
        """ fill in the new DAI the value from real DA.

           :param real_da: real intended user DA
           :type real_da: alex.components.slu.da.DialogueAct
           :param new_dai: DAI from new noisy DA
           :type new_dai: alex.components.slu.da.DialogueActItem
        """

        found = False
        for dai in real_da:
            if dai.dat == new_dai.dat and dai.name == new_dai.name and dai.value:
                new_dai.value = dai.value
                found = True
        if not found:
            self._fill_random_value(new_dai)

    def change_da(self, user_da):
        u_da = user_da.get_best_da()
        #get best da!
        copy_da = deepcopy(u_da)
        Preprocessing.remove_slot_values(copy_da)
        new_da = DialogueAct()

        #introduce errors from modeled frames
        for dai in self.slot_frames:
            if dai in copy_da:
                if unicode(dai) in self.present_dai_struct:
                    sampler = [self.present_dai_struct[unicode(dai)]["Y"], self.present_dai_struct[unicode(dai)]["N"]]
                    res = RandomGenerator.generate_random_response([True, False], sampler, sum(sampler))
                    if res:
                        d = DialogueAct(dai)
                        new_da.append(d.dais[0])
                else:
                    d = DialogueAct(dai)
                    new_da.append(d.dais[0])
            else:
                if unicode(dai) in self.absent_dai_struct:
                    sampler = [self.absent_dai_struct[unicode(dai)]["Y"], self.absent_dai_struct[unicode(dai)]["N"]]
                    res = RandomGenerator.generate_random_response([True, False], sampler, sum(sampler))
                    if res:
                        d = DialogueAct(dai)
                        new_da.append(d.dais[0])

        #restore dais that are not trained
        for dai in copy_da:
            if dai not in self.slot_frames:
                new_da.append(dai)

        if len(new_da) == 0:
            new_da.append(DialogueActItem('null'))

        #fill in values
        for dai in new_da:
            if dai.value and unicode(dai) in self.changed_val_dai_struct:
                sampler = [self.changed_val_dai_struct[unicode(dai)]["Y"], self.changed_val_dai_struct[unicode(dai)]["N"]]
                res = RandomGenerator.generate_random_response([True, False], sampler, sum(sampler))
                if res:
                    self._fill_random_value(dai)
                else:
                    self._fill_correct_value(u_da, dai)
            elif dai.value:
                self._fill_correct_value(u_da, dai)

        prob = float(RandomGenerator.generate_random_response_uniform(self.scores))

        if 'hangup' in unicode(new_da):
            prob = 1.0

        nblist = DialogueActNBList()
        nblist.add(prob, new_da)
        nblist.add(1-prob, DialogueAct('null()'))
        nblist.merge()
        nblist.scale()

        return nblist.get_confnet()


