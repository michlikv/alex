#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import random
import numpy
import codecs
import pickle
import math
from copy import deepcopy
from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActItem
from alex.components.slu.common import slu_factory
from alex.components.dm import Ontology
from collections import defaultdict
import pylab as pl

from simulator import Simulator
from Readers.FileReader import FileReader
from Readers.FileWriter import FileWriter
from Readers.Preprocessing import Preprocessing
from StateTracking import Tracker
from Generators.randomGenerator import RandomGenerator
from Trainig.NgramsTrained import NgramsTrained

from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer

class MLsimulator(Simulator):

    def __init__(self, cfg):
        self.cfg = cfg
        self.ontology = Ontology(cfg['UserSimulation']['ontology'])
        self.vectorizer = None
        self.classifiers = None
        self.slotvals = NgramsTrained(2)
        self.luda = DialogueAct('silence()')
        self.tracker = Tracker.Tracker(self.cfg)

        self._do_error_model = 'ErrorModel' in self.cfg['UserSimulation']
        self._mess_dai_rate = 0.0
        self._mess_slotval_rate = 0.0
        if self._do_error_model:
            self._mess_dai_rate = self.cfg['UserSimulation']['ErrorModel']['daiRate']
            self._mess_slotval_rate = self.cfg['UserSimulation']['ErrorModel']['slotValRate']

        RandomGenerator()

    def get_luda_nblist(self):
        nblist = DialogueActNBList()
        nblist.add(1.0, self.luda)
        nblist.merge()
        nblist.scale()
        return nblist

    def get_state(self):
        return self.tracker

    def new_dialogue(self):
        self.luda = DialogueAct('silence()')
        self.tracker = Tracker.Tracker(self.cfg)

    def _read_dialogue(self, filename):
        dialogue = FileReader.read_file(filename)
        if dialogue:
            dialogue = Preprocessing.prepare_conversations(dialogue,
                Preprocessing.create_act_from_stack_use_last,
                Preprocessing.create_act_from_stack_use_last)

            Preprocessing.add_end_string(dialogue)
            Preprocessing.clear_numerics(dialogue)
            return dialogue
        else:
            return None

    def _get_feature_names_and_responses(self, filelist):
        feature_vects = []
        responses = []
        for file_name in filelist:
            self.cfg['Logging']['system_logger'].info("processing file" + file_name)
            dialogue = self._read_dialogue(file_name)
            if dialogue:
                dialogue = ['silence()'] + dialogue
                self.cfg['Logging']['system_logger'].info(dialogue)
                self.cfg['Logging']['system_logger'].info(len(dialogue))

                # save slot values
                dialogue = [DialogueAct(d) for d in dialogue]
                slot_values = Preprocessing.get_slot_names_plus_values_from_dialogue(dialogue,
                                                                                         ignore_values=['none', '*'])
                self.slotvals.train_counts(slot_values)

                # track through dialogue, extract feature hash
                self.tracker.new_dialogue()
                while len(dialogue) > 1:
                    self.tracker.update_state(dialogue[0], dialogue[1])
                    self.tracker.log_state()

                    dialogue = dialogue[2:]
                    if len(dialogue) >= 1:
                        feature_vects.append(self.tracker.get_featurized_hash())
                        responses.append(dialogue[0])
        return feature_vects, responses

    def _create_classes(self, responses):
        size = len(responses)
        classes = defaultdict(str)
        for i, response in enumerate(responses):
            Preprocessing.remove_slot_values(response, ['task'])
            for dai in response.dais:
                if dai.dat != 'null':
                    classes.setdefault(unicode(dai), numpy.zeros(size, dtype=int))[i] = 1
        return classes

    def _create_feature_vectors(self, cfg):
        list_of_files_train = FileReader.read_file(cfg['UserSimulation']['files']['training-data'])
        list_of_files_test = FileReader.read_file(cfg['UserSimulation']['files']['testing-data'])

        vectors_train, responses_train = self._get_feature_names_and_responses(list_of_files_train)
        vectors_test, responses_test = self._get_feature_names_and_responses(list_of_files_test)

        # # add negative used features
        # #1. find all names
        # names_used = defaultdict(str)
        # for elem in feature_vects:
        #     for name, value in elem.iteritems():
        #         if name.endswith("_in"):
        #             names_used[name] = 1
        #
        # #2. add all negatives
        # for elem in feature_vects:
        #     for name, value in names_used.iteritems():
        #         if name not in elem:
        #             elem[name]="not_used"

        # transform vectors - fit by training data only!
        self.vectorizer = DictVectorizer(sparse=True)
        vectors_train = self.vectorizer.fit_transform(vectors_train)
        vectors_test = self.vectorizer.transform(vectors_test)
        self._save_features_vects_to_file(vectors_train, vectors_test)

        # build classes from testing and training lines.
        classes_train = self._create_classes(responses_train)
        classes_test = self._create_classes(responses_test)
        self._save_classes_to_file(classes_train, classes_test)

        self.cfg['Logging']['system_logger'].info("training data: " + str(vectors_train.get_shape()) + " - " + str(
            len(responses_train))+";")
        self.cfg['Logging']['system_logger'].info("testing data: " + str(vectors_test.get_shape()) + " - " + str(
            len(responses_test))+".")

        return vectors_train, classes_train, vectors_test, classes_test

    def train_simulator(self, cfg, create_vectors=False):
        if create_vectors:
            self.vectors_train, self.classes_train, self.vectors_test, self.classes_test = self._create_feature_vectors(cfg)
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
            #todo training with some parameters :) ???
            self.classifiers[name] = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
            #todo some search for params -- self.classifier.set_params(C=c)

            self.classifiers[name].fit(self.vectors_train, vector)
            predict = self.classifiers[name].predict_proba(self.vectors_train)[:, 1]
            predict_noprob = self.classifiers[name].predict(self.vectors_train)

            print "train score for", name, "-", str(self.classifiers[name].score(self.vectors_train, vector))
            print "> ones in classes:", str(sum(vector))
            print "> ones in predicted:", str(sum(predict_noprob))
            print str(set(predict))
            if name in self.classes_test:
                print "test score for", name, "-", str(self.classifiers[name].score(self.vectors_test, self.classes_test[name]))
            else:
                print "no test vector"

        if 'classifiers' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classifiers'], self.classifiers)

        self.make_stats_all(self.cfg['UserSimulation']['files']['stats'])

    def make_stats_all(self, prefixfile):
        thresholds = numpy.linspace(0.0, 1.0, 11)
        #training data stats
        self.make_stats(self.vectors_train, self.classes_train, prefixfile+"-train-", thresholds)
        #testing data stats
        self.make_stats(self.vectors_test, self.classes_test, prefixfile+"-test-", thresholds)
        #thresholds = numpy.linspace(0.1, 0.3, 11)
        #self.make_stats(self.vectors_train, self.classes_train, prefixfile+"-0203-", thresholds)

        tops = []
        for name, classifier in self.classifiers.iteritems():
            tops.append(eval.get_topN_features(self.vectorizer, classifier, name, n=10))
        FileWriter.write_file(prefixfile+'top-weight-features.txt', tops)
        self.cfg['Logging']['system_logger'].info("Stats finished")

    def make_stats(self, vectors, classes, prefixfile, thresholds):
        #for threshold in thresholds:
          #  precision, recall, null = self.count_precision_recall_null(vectors, classes, threshold)
          #  print prefixfile, "threshold:", threshold, "precision:", precision, "recall", recall, "nulls", null
          #  precs.append(precision)
          #  recs.append(recall)
          #  nulls.append(null)
        #factored, mean = eval.count_precision_recall_accuracy_null(vectors, classes, thresholds, self.classifiers)
        mean_prec, mean_rec = eval.count_precision_recall_by_turns(vectors, classes, thresholds, self.classifiers)
        # this counts mean precision and mean recall across classifiers
        eval.make_plot_pr(prefixfile+"means", thresholds, mean_prec, mean_rec)

        # this makes precision and recall plot for each classifier
        # i = 100
        # for name, val in factored.iteritems():
        #     eval.make_plot_pra(str(i)+prefixfile+name, thresholds, val['precision'], val['recall'], val['accuracy'])
        #     i += 1


    @staticmethod
    def load_obj(filename):
        input = open(filename, 'rb')
        obj = pickle.load(input)
        input.close()
        return obj

    @staticmethod
    def load(cfg):
        obj = MLsimulator(cfg)
        if 'vectorizer' in obj.cfg['UserSimulation']['files']:
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

    def save(self, cfg):
        if 'vectorizer' in cfg['UserSimulation']['files']:
            self.save_obj(cfg['UserSimulation']['files']['vectorizer'], self.vectorizer)
        if 'slotvals' in cfg['UserSimulation']['files']:
            self.slotvals.save(cfg['UserSimulation']['files']['slotvals'])
        if 'classifiers' in cfg['UserSimulation']['files']:
            self.save_obj(cfg['UserSimulation']['files']['classifiers'], self.classifiers)
        self._save_features_vects_to_file(self.vectors_train, self.vectors_test)
        self._save_classes_to_file(self.classes_train, self.classes_test)
        print "."

    def save_obj(self, filename, obj):
        out = open(filename, 'wb')
        pickle.dump(obj, out)
        out.close()

    def _save_features_vects_to_file(self, vectors_train, vectors_test):
        # save objects to files
        if 'feature_vects_train' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_train'], vectors_train)
        if 'feature_vects_test' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['feature_vects_test'], vectors_test)

        # save feature names to file :)
        if 'feature_names' in self.cfg['UserSimulation']['files']:
            FileWriter.write_file(self.cfg['UserSimulation']['files']['feature_names'], self.vectorizer.get_feature_names())
        if 'vectorizer' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['vectorizer'], self.vectorizer)
        if 'slotvals' in self.cfg['UserSimulation']['files']:
            self.slotvals.save(self.cfg['UserSimulation']['files']['slotvals'])

    def _save_classes_to_file(self, classes_train, classes_test):
        if 'classes_train' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classes_train'], classes_train)
        if 'classes_test' in self.cfg['UserSimulation']['files']:
            self.save_obj(self.cfg['UserSimulation']['files']['classes_test'], classes_test)
        # write names of classes with its sizes
        sizes = []
        for name, cl in classes_train.iteritems():
            sizes.append(name + ' ' + str(sum(cl)))
        FileWriter.write_file('data/classes-sizes.txt', sizes)

    def _sample_response(self, vector):
        """
        Sample response for given vector using probability from classifiers.
        :rtype : tuple of DA - real DA response and messed DA according to error model
        :param vector: feature vector
        :return: sampled DA
        """
        response_real = []

        # classify vector, sample response
        for name, classifier in self.classifiers.iteritems():
            prob = classifier.predict_proba(vector)[0, 1]

            is_generated = RandomGenerator.is_generated(prob)
            if is_generated:
                response_real.append(name)

        realDA = DialogueAct("&".join(response_real)) if len(response_real) > 0 else None
        return realDA

    def _sample_response_sec(self, vector):
        """
        Build response for given vector using hard 1 and 0 from classifiers.
        :param vector: feature vector
        :return: DA
        """
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

    def _fill_in_slot_values(self, response, system_da):
        #substitute "city" and "stop" with correct context-dependent value
        cop_da = deepcopy(response)
        for dai in response:
            if dai.name and dai.value and dai.name == "city" or dai.name == "stop" and dai.dat == 'inform':
                dai_system = None
                # find first request with correct substring
                for sdai in system_da:
                    if sdai.name and dai.name in sdai.name and sdai.dat == "request":
                        dai_system = sdai
                        break
                # #if there is no request, try any slot name with substring
                # if dai_system is None:
                #     for sdai in system_da:
                #         if sdai.name is not None and dai.name in sdai.name:
                #             dai_system = sdai
                #             break
                if dai_system:
                    dai.name = dai_system.name
                else:
                    dai.name = "from_"+dai.name

        for dai in response.dais:
            if dai.value and dai.value == "&":
                # deny what system said, confirm and inform what user previously said.
                if dai.dat == "deny":
                    from_state = self.tracker.get_value_said_system(dai.name)
                    # use brand new value for deny if system and user share value
                    if from_state == self.tracker.get_value_said_user(dai.name):
                        from_state = None
                else:
                    from_state = self.tracker.get_value_said_user(dai.name)

                if from_state is not None and (dai.name in self.ontology['fixed_goal'] or dai.dat == "deny"):
                    selected = from_state
                else: #generate uniform from compatible values
                    possible_values = None
                    if ("city" in dai.name or "stop" in dai.name) and dai.dat != "deny":
                        possible_values = self.tracker.get_compatible_values(dai, response)
                        if possible_values is None or len(possible_values) == 0:
                            possible_values = None

                    if possible_values is None:
                        possible_values, v, s = self.slotvals.get_possible_reactions((dai.name,))

                        if not possible_values:
                            #possible_values, v, s = self.slotvals.get_possible_unigrams()
                            print "No SLOT VALUE FOR SLOT NAME:", dai.name
                            raise
                    selected = RandomGenerator.generate_random_response_uniform(possible_values)
                dai.value = selected

        #substitute "city" and "stop" back
        for daiR, daiO in zip(response, cop_da):
            if daiO.name and daiR.name and daiO.name != daiR.name:
                daiR.name = daiO.name

    # def _messup_slot_values(self, response):
    #     mess = deepcopy(response)
    #
    #     for dai in mess.dais:
    #         if dai.value and RandomGenerator.is_generated(self._mess_dai_rate):
    #             # if there is a value and we decide to alter values, choose uniformly different value
    #             possible_values, v, s = self.slotvals.get_possible_reactions((dai.name,))
    #             selected = RandomGenerator.generate_random_response_uniform(possible_values)
    #             dai.value = selected
    #     return mess

    def get_oov(self):
        return 0.0

    def generate_response_from_history(self, history):
        #start a new dialogue
        self.new_dialogue()

        #track history
        d = [DialogueAct('silence()')] + history
        while len(d) > 1:
            self.tracker.update_state(d[0], d[1])
            self.tracker.log_state()
            d = d[2:]

        # create vector from state
        vector = self.vectorizer.transform(self.tracker.get_featurized_hash())
        #sample response from vector
        response = self._sample_response(vector)

        # fill in slot values or set to null act
        if response is not None:
            self._fill_in_slot_values(response)
        else:
            response = DialogueAct('null()')

        nblist = DialogueActNBList()
        nblist.add(1.0, response)
        nblist.merge()
        nblist.scale()
        return nblist.get_confnet()

    def generate_response(self, system_da):
        # update state and log it
        self.tracker.update_state(self.luda, system_da)
        self.tracker.log_state()

        # create vector from state
        vector = self.vectorizer.transform(self.tracker.get_featurized_hash())
        response = self._sample_response(vector)

        if response is None:
            response = DialogueAct('null()')
        self._fill_in_slot_values(response, system_da)

        self.luda = response

        nblist = DialogueActNBList()
        nblist.add(1.0, response)
        nblist.merge()
        nblist.scale()
        return nblist.get_confnet()

class eval:

    @staticmethod
    def count_precision_recall_null_bag_of(vectors, classes, thresholds, classifiers):
        # counts precision, recall, accuracy as "bag of" representation of zeros and ones
        # across all classifiers
        size = len(thresholds)
        none_gen = numpy.array([0.0]*size)
        true_pos = numpy.array([0.0]*size)
        false_pos = numpy.array([0.0]*size)
        true_neg = numpy.array([0.0]*size)
        total_pos = 0.0
        for i, vector in enumerate(vectors):
            pom = numpy.array([0]*size)
            for name, classifier in classifiers.iteritems():
                prob = classifier.predict_proba(vector)[0, 1]
                probs = [prob]*size
                probs = numpy.array([0 if p <= t else 1 for p, t in zip(probs, thresholds)])
                pom += probs

                if classes[name][i] == 1:
                    total_pos += 1
                    true_pos += probs
                else:
                    false_pos += probs
                    true_neg += numpy.array([0 if p == 1 else 1 for p in probs])
            none_gen = [n + 1 if pm > 0 else n for n, pm in zip(none_gen, pom)]
        none_gen = numpy.array(none_gen)

        #return precision, recall, accuracy, nulls
        return true_pos/(true_pos+false_pos), true_pos/total_pos, \
               (true_pos+true_neg)/(vectors.get_shape()[0]*len(classes)), \
               1-none_gen/vectors.get_shape()[0]

    @staticmethod
    def count_precision_recall_by_turns(vectors, classes, thresholds, classifiers):
        # counts precision, recall, accuracy as "bag of" representation of zeros and ones
        # across all classifiers

        size = len(thresholds)
        precisions = []
        recalls = []

        # for each vector = for each turn
        for i, vector in enumerate(vectors):
            true_pos = numpy.array([0.0]*size)  # number of correct predictions
            false_pos = numpy.array([0.0]*size) # number of incorrect predictions
            total_pos = 0.0                     # total number of true ones in test vector

            pom = numpy.array([0]*size)
            for name, classifier in classifiers.iteritems():
                prob = classifier.predict_proba(vector)[0, 1]
                probs = [prob]*size
                probs = numpy.array([0 if p <= t else 1 for p, t in zip(probs, thresholds)])
                pom += probs
                if name in classes and classes[name][i] == 1:
                    total_pos += 1
                    true_pos += probs
                else:
                    false_pos += probs

            # if user said null, predicted null is ok -- add one act to truly positives and one to total positives
            # we must not divide by zero!
            if total_pos < 1.0:
                true_pos = numpy.array([1.0 if tp+fp < 1.0 else tp for tp, fp in zip(true_pos, false_pos)])
                total_pos += 1
            else:
                false_pos = numpy.array([1.0 if tp+fp < 1.0 else fp for tp, fp in zip(true_pos, false_pos)])

            #count precision and recall, add to list.
            precisions.append(true_pos/(true_pos+false_pos))
            recalls.append(true_pos/total_pos)

        def avg(vec):
            av = numpy.array([0.0]*size)
            for v in precisions:
                av += v
            av /= len(precisions)
            return av

        #return average precision, recall
        return avg(precisions), avg(recalls)


    @staticmethod
    def count_precision_recall_accuracy_null_by_classifiers(vectors, classes, thresholds, classifiers):
        #counts precision, recall, accuracy by classifiers, adds mean + nulls
        size = len(thresholds)

        none_gen = numpy.array([0.0]*size)

        class_stats = defaultdict(lambda: defaultdict(str))
        class_result = defaultdict(lambda: defaultdict(str))
        class_result_mean = defaultdict(str)

        for i, vector in enumerate(vectors):
            pom = numpy.array([0]*size)
            for name, classifier in classifiers.iteritems():
                prob = classifier.predict_proba(vector)[0, 1]
                probs = [prob]*size
                probs = numpy.array([0 if p <= t else 1 for p, t in zip(probs, thresholds)])
                pom += probs

                if classes[name][i] == 1:
                    class_stats[name]['true_pos'] = class_stats[name].get('true_pos', numpy.array([0.0]*size)) + probs
                    class_stats[name]['total_pos'] = class_stats[name].get('total_pos', 0.0) + 1
                else:
                    class_stats[name]['false_pos'] = class_stats[name].get('false_pos', numpy.array([0.0]*size)) + probs
                    class_stats[name]['true_neg'] = class_stats[name].get('true_neg',
                                               numpy.array([0.0]*size)) + numpy.array([0 if p==1 else 1 for p in probs])

            none_gen = [n + 1 if pm > 0 else n for n, pm in zip(none_gen, pom)]
        none_gen = numpy.array(none_gen)

        for name, h in class_stats.iteritems():
            class_result[name]['precision'] = class_stats[name].get('true_pos', numpy.array([0.0]*size))/(class_stats[name].get('true_pos', numpy.array([0.0]*size))+class_stats[name].get('false_pos', numpy.array([0.0]*size)))
            class_result[name]['precision'] = numpy.nan_to_num(class_result[name]['precision'])
            class_result[name]['recall'] = class_stats[name].get('true_pos', numpy.array([0.0]*size))/class_stats[name].get('total_pos', 0.0)
            class_result[name]['accuracy'] = (class_stats[name].get('true_pos', numpy.array([0.0]*size))+class_stats[name].get('true_neg', numpy.array([0.0]*size)))/(vectors.get_shape()[0])

            #if clas_result[name]['precision'] > 0.0:
            class_result_mean['mean_precision'] = class_result_mean.get('mean_precision', numpy.array([0.0]*size)) + class_result[name]['precision']
            #if clas_result[name]['recall'] > 0.0:
            class_result_mean['mean_recall'] = class_result_mean.get('mean_recall',numpy.array([0.0]*size)) + class_result[name]['recall']
            #if clas_result[name]['accuracy'] > 0.0:
            class_result_mean['mean_accuracy'] = class_result_mean.get('mean_accuracy', numpy.array([0.0]*size)) + class_result[name]['accuracy']

        class_result_mean['mean_precision'] /= len(classes)
        class_result_mean['mean_recall'] /= len(classes)
        class_result_mean['mean_accuracy'] /= len(classes)
        class_result_mean['nulls'] = 1 - none_gen/vectors.get_shape()[0]
        return class_result, class_result_mean

    @staticmethod
    def get_topN_features(vectorizer, clf, class_label, n=10):
        """Prints features with the highest coefficient values, per class"""
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        s = class_label+"\n"
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            s += "\t%.4f\t%-15s\t\t%.4f\t%-15s \n" % (coef_1, fn_1, coef_2, fn_2)
            #return "%s: %s" % ('class_label', " ".join(feature_names[j] for j in top10))
        return s

    @staticmethod
    def make_plot_pr(filename, thresholds, precs, recs):
        l = ["\t".join([str(t) for t in thresholds]),
             "\t".join([str(t) for t in precs]),
             "\t".join([str(t) for t in recs])]
        FileWriter.write_file(filename+".txt", l)

        # make a plot
        p, = pl.plot(thresholds, precs, "ro-", label="Precision")
        r, = pl.plot(thresholds, recs, "bs-", label="Recall")
        pl.axis([min(thresholds), max(thresholds), 0.0, 1.0])
        pl.xlabel('Thresholds')
        pl.ylabel('Value')
        pl.legend(handles=[p, r])
        pl.savefig(filename+".png")
        pl.close()


    @staticmethod
    def make_plot_pran(filename, thresholds, precs, recs, accs, nulls):
        l = ["\t".join([str(t) for t in thresholds]),
             "\t".join([str(t) for t in precs]),
             "\t".join([str(t) for t in recs]),
             "\t".join([str(t) for t in accs])]
        FileWriter.write_file("data/"+filename+".txt", l)

        # make a plot
        p, = pl.plot(thresholds, precs, "ro-", label="Precision")
        r, = pl.plot(thresholds, recs, "bs-", label="Recall")
        a, = pl.plot(thresholds, accs, "g^-", label="Accuracy")
        n, = pl.plot(thresholds, nulls, "k+-", label="Nulls")
        pl.axis([min(thresholds), max(thresholds), 0.0, 1.0])
        pl.xlabel('Thresholds')
        pl.ylabel('val')
        pl.legend(handles=[p, r, a, n])
        pl.savefig("data/"+filename+".png")
        pl.close()

    @staticmethod
    def make_plot_pra(filename, thresholds, precs, recs, accs):
        l = ["\t".join([str(t) for t in thresholds]),
             "\t".join([str(t) for t in precs]),
             "\t".join([str(t) for t in recs]),
             "\t".join([str(t) for t in accs])]
        FileWriter.write_file("data/"+filename+".txt", l)

        # make a plot
        p, = pl.plot(thresholds, precs, "ro-", label="Precision")
        r, = pl.plot(thresholds, recs, "bs-", label="Recall")
        a, = pl.plot(thresholds, accs, "g^-", label="Accuracy")
        pl.axis([min(thresholds), max(thresholds), 0.0, 1.0])
        pl.xlabel('Thresholds')
        pl.ylabel('val')
        pl.legend(handles=[p, r, a])
        pl.savefig("data/"+filename+".png")
        pl.close()

