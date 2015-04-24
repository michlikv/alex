#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from gi.overrides.GLib import MainLoop
import random
import numpy
import codecs
import pickle
from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActItem
from alex.components.slu.common import slu_factory
from collections import defaultdict
import pylab as pl

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
from StateTracking import Tracker
from Generators.randomGenerator import RandomGenerator
from Trainig.NgramsTrained import NgramsTrained

from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer

class MLsimulator(Simulator):

    def __init__(self, cfg):
        self.cfg = cfg
        self.vectorizer = None
        self.classifiers = None
        self.slotvals = NgramsTrained(2)
        self.luda = DialogueAct('silence()')
        self.tracker = Tracker.Tracker(self.cfg)
        RandomGenerator()

    def new_dialogue(self):
        self.luda = DialogueAct('silence()')
        self.tracker = Tracker.Tracker(self.cfg)

    def _print_lines_to_file(self, filename, lines):
        f = codecs.open(filename, "w", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()

    def _create_feature_vectors(self, filename_list):
        list_of_files = FileReader.read_file(filename_list)
        feature_vects = []
        responses = []

        for file in list_of_files:
            self.cfg['Logging']['system_logger'].info("processing file" + file)
            dialogue = FileReader.read_file(file)
            if dialogue:
                dialogue = Preprocessing.prepare_conversations(dialogue,
                    Preprocessing.create_act_from_stack_use_last,
                    Preprocessing.create_act_from_stack_use_last)

                Preprocessing.add_end_string(dialogue)
                Preprocessing.clear_numerics(dialogue)
                dialogue = ['silence()'] + dialogue

                self.cfg['Logging']['system_logger'].info(dialogue)
                self.cfg['Logging']['system_logger'].info(len(dialogue))

                dialogue = [DialogueAct(d) for d in dialogue]

                # save slot values
                slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue)
                self.slotvals.train_counts(slot_values, unicode)

                self.tracker = Tracker.Tracker(self.cfg)
                while len(dialogue) > 1:
                    self.tracker.update_state(dialogue[0], dialogue[1])
                    self.tracker.log_state()

                    dialogue = dialogue[2:]
                    if len(dialogue) >= 1:
                        feature_vects.append(self.tracker.get_featurized_hash())
                        responses.append(dialogue[0])

                    #except:
                    #   self.cfg['Logging']['system_logger'].info('Error: '+file)

        # add negative used features
        #1. find all names
        names_used = defaultdict(str)
        for elem in feature_vects:
            for name, value in elem.iteritems():
                if name.endswith("_in"):
                    names_used[name] = 1

        #2. add all negatives
        for elem in feature_vects:
            for name, value in names_used.iteritems():
                if name not in elem:
                    elem[name]="not_used"

        # training and testing data are 90% and 10% of vectors
        cutindex =int(len(feature_vects)*0.1)
        vectors_train = feature_vects[:-cutindex]
        vectors_test = feature_vects[-cutindex:]
        responses_train = responses[:-cutindex]
        responses_test = responses[-cutindex:]

        # transform vectors - fit by training data only!
        self.vectorizer = DictVectorizer(sparse=True)
        vectors_train = self.vectorizer.fit_transform(vectors_train)
        vectors_test = self.vectorizer.transform(vectors_test)

        # save objects to files
        if 'feature_vects_train' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_train'], vectors_train)
        if 'feature_vects_test' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_test'], vectors_test)

        # save feature names to file :)
        if 'feature_names' in self.cfg['UserSimulation']['files']:
            self._print_lines_to_file(self.cfg['UserSimulation']['files']['feature_names'], self.vectorizer.get_feature_names())
        if 'vectorizer' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['vectorizer'], self.vectorizer)
        if 'slotvals' in self.cfg['UserSimulation']['files']:
            self.slotvals.save(self.cfg['UserSimulation']['files']['slotvals'])


        # build classes from testing and training lines.
        classes_train = self._create_classes(responses_train)
        classes_test = self._create_classes(responses_test)
        if 'classes_train' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classes_train'], classes_train)
        if 'classes_test' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classes_test'], classes_test)

        #todo ladici - velikosti class
        sizes = []
        for name, cl in classes_train.iteritems():
            sizes.append(name + ' ' + str(sum(cl)))
        self._print_lines_to_file('data/classes-sizes.txt', sizes)

        self.cfg['Logging']['system_logger'].info("training data: " + str(vectors_train.get_shape()) + " - " + str(
            len(responses_train))+";")
        self.cfg['Logging']['system_logger'].info("testing data: " + str(vectors_test.get_shape()) + " - " + str(
            len(responses_test))+".")

        return vectors_train, classes_train, vectors_test, classes_test

    def _create_classes(self, responses):
        size = len(responses)
        classes = defaultdict(str)
        for i, response in enumerate(responses):
            Preprocessing.remove_slot_values(response)
            for dai in response.dais:
                if dai.dat != 'null':
                    classes.setdefault(unicode(dai), numpy.zeros(size, dtype=int))[i] = 1
        return classes

    def train_simulator(self, filename_filelist, create_vectors=False):
        if create_vectors:
            self.vectors_train, self.classes_train, self.vectors_test, self.classes_test = self._create_feature_vectors(filename_filelist)
        else:
            if ('feature_vects_train' in self.cfg['UserSimulation']['files'] and
                'feature_vects_test' in self.cfg['UserSimulation']['files'] and
                'classes_train' in self.cfg['UserSimulation']['files'] and
                'classes_test' in self.cfg['UserSimulation']['files']):
                self.classes_train = self.load_obj(self.cfg['UserSimulation']['files']['classes_train'])
                self.classes_test = self.load_obj(self.cfg['UserSimulation']['files']['classes_test'])
                self.vectors_train = self.load_obj(self.cfg['UserSimulation']['files']['feature_vects_train'])
                self.vectors_test = self.load_obj(self.cfg['UserSimulation']['files']['feature_vects_test'])
                if 'vectorizer' in self.cfg['UserSimulation']['files']:
                    self.vectorizer = self.load_obj(self.cfg['UserSimulation']['files']['vectorizer'])
                if 'slotvals' in self.cfg['UserSimulation']['files']:
                    self.slotvals = self.load_obj(self.cfg['UserSimulation']['files']['slotvals'])

        self.classifiers = defaultdict(str)
        for name, vector in self.classes_train.iteritems():
            #todo training with some parameters :)
            self.classifiers[name] = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
            #some search for params -- self.classifier.set_params(C=c)
            self.classifiers[name].fit(self.vectors_train, vector)
            predict = self.classifiers[name].predict_proba(self.vectors_train)[:, 1]
            predict_noprob = self.classifiers[name].predict(self.vectors_train)
            print "| ones in classes:", str(sum(vector))
            print "| ones in predicted:", str(sum(predict_noprob))

            print str(set(predict))

            print "train score for", name, "-", str(self.classifiers[name].score(self.vectors_train, vector))
            if name in self.classes_test:
                print "test score for", name, "-", str(self.classifiers[name].score(self.vectors_test, self.classes_test[name]))
            else:
                print "no test vector"

        if 'classifiers' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classifiers'], self.classifiers)

        #precision, recall, null = self.count_precision_recall_null(self.vectors_train, self.classes_train, 0.5)
        #print "train data precision:", precision, "recall", recall, "nulls", null

    def make_stats_all(self, prefixfile):

        thresholds = numpy.linspace(0.0, 1.0, 11)
        self.make_stats(self.vectors_train, self.classes_train, prefixfile+"-train-", thresholds)
        #thresholds = numpy.linspace(0.1, 0.3, 11)
        #self.make_stats(self.vectors_train, self.classes_train, prefixfile+"-0203-", thresholds)

        tops = []
        for name, classifier in self.classifiers.iteritems():
            tops.append(self._get_top10(self.vectorizer, classifier, name))
        self._print_lines_to_file('data/top-weight-features.txt', tops)
        print "."
        #todo debug proc to pada!
       # self.make_stats(self.vectors_test, self.classes_test, prefixfile+"-test-")

    def make_stats(self, vectors, classes, prefixfile, thresholds):
        #for threshold in thresholds:
          #  precision, recall, null = self.count_precision_recall_null(vectors, classes, threshold)
          #  print prefixfile, "threshold:", threshold, "precision:", precision, "recall", recall, "nulls", null
          #  precs.append(precision)
          #  recs.append(recall)
          #  nulls.append(null)
        precs, recs, nulls = self.count_precision_recall_null(vectors, classes, thresholds)
        print "Thresholds:", thresholds
        print "Precisions:", precs
        print "Recalls:", recs
        print "Nulls:", nulls

        # make a plot
        p, = pl.plot(thresholds, precs, "ro-", label="Precision")
        r, = pl.plot(thresholds, recs, "bs-", label="Recall")
        n, = pl.plot(thresholds, nulls, "g^-", label="nulls")
        pl.axis([min(thresholds), max(thresholds), 0.0, 1.0])
        pl.xlabel('Thresholds')
        pl.ylabel('Probability')
        pl.legend(handles=[p, r, n])
        pl.savefig("data/"+prefixfile+'-prec-rec.png')

    def _get_top10(self, vectorizer, clf, class_label, n=10):
        """Prints features with the highest coefficient values, per class"""
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        s = class_label+"\n"
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            s += "\t%.4f\t%-15s\t\t%.4f\t%-15s \n" % (coef_1, fn_1, coef_2, fn_2)
            #return "%s: %s" % ('class_label', " ".join(feature_names[j] for j in top10))
        return s

    def count_precision_recall_null(self, vectors, classes, thresholds):
        size = len(thresholds)
        none_gen = numpy.array([0.0]*size)
        true_pos = numpy.array([0.0]*size)
        false_pos = numpy.array([0.0]*size)
        total_pos = 0.0
        for i, vector in enumerate(vectors):
            pom = numpy.array([0]*size)
            for name, classifier in self.classifiers.iteritems():
                prob = classifier.predict_proba(vector)[0, 1]
                probs = [prob]*size
                probs = numpy.array([0 if p<=t else 1 for p, t in zip(probs, thresholds)])
                pom += probs

                if classes[name][i] == 1:
                    total_pos += 1
                    true_pos += probs
                else:
                    false_pos += probs
            none_gen = [ n + 1 if pm>0 else n for n,pm in zip(none_gen, pom)]
            #if not hasone: none_gen += 1

        none_gen = numpy.array(none_gen)
        #return precision, recall, nulls
        #print true_pos
        #print false_pos
        #print total_pos
        #print none_gen
        return true_pos/(true_pos+false_pos), true_pos/total_pos, 1 - none_gen/vectors.get_shape()[1]



    @staticmethod
    def load_obj(filename):
        input = open(filename, 'rb')
        obj = pickle.load(input)
        input.close()
        return obj

    @staticmethod
    def load(cfg):
        obj = MLsimulator(cfg)
        if 'vectorizer' in cfg['UserSimulation']['files']:
            obj.vectorizer = obj.load_obj(cfg['UserSimulation']['files']['vectorizer'])
        if 'slotvals' in cfg['UserSimulation']['files']:
            obj.slotvals = NgramsTrained.load(cfg['UserSimulation']['files']['slotvals'])
        if 'classifiers' in cfg['UserSimulation']['files']:
            obj.classifiers = obj.load_obj(obj.cfg['UserSimulation']['files']['classifiers'])
        if ('feature_vects_train' in obj.cfg['UserSimulation']['files'] and
            'feature_vects_test' in obj.cfg['UserSimulation']['files'] and
            'classes_train' in obj.cfg['UserSimulation']['files'] and
            'classes_test' in obj.cfg['UserSimulation']['files']):
            obj.classes_train = obj.load_obj(obj.cfg['UserSimulation']['files']['classes_train'])
            obj.classes_test = obj.load_obj(obj.cfg['UserSimulation']['files']['classes_test'])
            obj.vectors_train = obj.load_obj(obj.cfg['UserSimulation']['files']['feature_vects_train'])
            obj.vectors_test = obj.load_obj(obj.cfg['UserSimulation']['files']['feature_vects_test'])
        return obj

    def save(self):
        if 'vectorizer' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['vectorizer'], self.vectorizer)
        print "."
        if 'slotvals' in self.cfg['UserSimulation']['files']:
            self.slotvals.save(self.cfg['UserSimulation']['files']['slotvals'])
        print "."
        if 'classifiers' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classifiers'], self.classifiers)
        print "."
        if ('feature_vects_train' in self.cfg['UserSimulation']['files'] and
            'feature_vects_test' in self.cfg['UserSimulation']['files'] and
            'classes_train' in self.cfg['UserSimulation']['files'] and
            'classes_test' in self.cfg['UserSimulation']['files']):
            self.save_obj(self.cfg['UserSimulation']['files']['classes_train'], self.classes_train)
            print "."
            self.save_obj(self.cfg['UserSimulation']['files']['classes_test'], self.classes_test)
            print "."
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_train'], self.vectors_train)
            print "."
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_test'], self.vectors_test)
            print "."

    def save_obj(self, filename, obj):
        out = open(filename, 'wb')
        pickle.dump(obj, out)
        out.close()

    @staticmethod
    def load_simulator(cfg):
        simulator = MLsimulator(cfg)
        if ('vectorizer' in cfg['UserSimulation']['files'] and
            'classifiers' in cfg['UserSimulation']['files']):
            simulator.vectorizer = MLsimulator.load_obj(cfg['UserSimulation']['files']['vectorizer'])
            simulator.classifiers = MLsimulator.load_obj(cfg['UserSimulation']['files']['classifiers'])
        return simulator

    def save_simulator(self):
        if ('vectorizer' in self.cfg['UserSimulation']['files'] and
            'classifiers' in self.cfg['UserSimulation']['files']):
            MLsimulator.save_obj(self.cfg['UserSimulation']['files']['vectorizer'], self.vectorizer)
            MLsimulator.save_obj(self.cfg['UserSimulation']['files']['classifiers'], self.classifiers)


    def _sample_response(self, vector):
        response = []
        # classify vector, sample response
        for name, classifier in self.classifiers.iteritems():
            prob = classifier.predict_proba(vector)[0, 1]
            if RandomGenerator.is_generated(prob):
                response.append(name)
        if len(response) > 0:
            return DialogueAct("&".join(response))
        else:
            return None

    def _sample_response_sec(self, vector):
        response = []
        # classify vector, sample response
        for name, classifier in self.classifiers.iteritems():
            prob = classifier.predict(vector)[0]
            if prob:
                response.append(name)
        if len(response) > 0:
            return DialogueAct("&".join(response))
        else:
            return None

    def generate_response(self, system_da):
        # update state and log it
        self.tracker.update_state(self.luda, system_da)
        self.tracker.log_state()

        # create vector from state
        vector = self.vectorizer.transform(self.tracker.get_featurized_hash())
        response = self._sample_response(vector)

        if response is not None:
            for dai in response.dais:
                if dai.value:
                    # deny what system said, confirm and inform what user previously said.
                    # todo je mozne pracovat s nespolupracujicim uzivatelem
                    if dai.dat == "deny":
                        from_state = self.tracker.get_value_said_system(dai.name)
                    else:
                        from_state = self.tracker.get_value_said_user(dai.name)

                    if from_state is not None:
                        selected = from_state

                    else: #generate uniform
                        possible_values = self.slotvals.get_possible_reactions((dai.name,))
                        if not possible_values:
                            possible_values = self.slotvals.get_possible_unigrams()
                        selected = RandomGenerator.generate_random_response_uniform(possible_values[0])
                    dai.value = selected
        else:
            response = DialogueAct('null()')

        self.luda = response

        nblist = DialogueActNBList()
        nblist.add(1.0, response)
        nblist.merge()
        nblist.scale()
        return nblist.get_confnet()


