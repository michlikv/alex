#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals, print_function

import time
import datetime
from collections import defaultdict
import autopath
import os.path
import argparse
import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActConfusionNetwork
from alex.utils.config import Config
from alex.components.dm.common import dm_factory, get_dm_type
import codecs
from Simulators import constantSimulator, simpleNgramSimulator, NgramSimulatorFiltered, MLsimulator
from Generators.randomGenerator import RandomGenerator
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing

class Eval:

    def __init__(self, cfg, dirname):
        self.cfg = cfg
        self.simulator = None
        self.dirname = dirname
        #self.ml_init(cfg)

        dm_type = get_dm_type(cfg)
        self.dm = dm_factory(dm_type, cfg)
        #TODO config user simulators from config :-O

        self.stats = Dialogue_stats()

        #self.bigram_init(cfg)
        RandomGenerator()


    # def generate20(self):
    #     r = []
    #     for i in range(20):
    #         r.append(RandomGenerator.generate_random_response([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1], 22))
    #     return r

    def bigram_init(self, cfg):
        self.simulator = simpleNgramSimulator.SimpleNgramSimulator(cfg)
        self.simulator.train_simulator('data-lists/03-slu-500.txt')

    def bigram_filtered_init(self, cfg):
        self.simulator = NgramSimulatorFiltered.NgramSimulatorFilterSlots(cfg)
        self.simulator.train_simulator('data-lists/03-slu-500.txt')

    def ml_init(self, cfg):
        #simulator = MLsimulator.MLsimulator(self.cfg)
        #simulator.train_simulator(self.cfg['UserSimulation']['files']['source'], False)
        self.simulator = MLsimulator.MLsimulator.load(cfg)

    def run(self, test_list, sim_list):
        self.cfg['Logging']['system_logger'].info("Preparing structures")
        listT = FileReader.read_file(test_list)
        for filename in listT:
            dialogue = self.get_dialogue_from_file(filename)
            self.stats.update_structure(dialogue)

        dial_len = self.stats.dialogue_lengths
        turn_len = self.stats.turn_lengths

        Draw_plots.count_length_stats({"real": dial_len}, self.dirname+"/dialogue-length.txt")
        Draw_plots.count_length_stats({"real": turn_len}, self.dirname+"/turn-length.txt")
        Draw_plots.count_length_stats(self.stats.unique_acts, self.dirname+"/Real-avg-acts.txt")

        Draw_plots.plot_mean_lengths_stats({"real1": dial_len, "real2": dial_len, "real3": dial_len}, self.dirname + "/dialogue-lengths.png", "Dialogue Lengths")
        Draw_plots.plot_mean_lengths_stats({"real1": turn_len, "real2": turn_len, "real3": turn_len}, self.dirname + "/turn-lengths.png", "Turn Lengths")
        Draw_plots.plot_histograms({"real": dial_len}, self.dirname + "/dial-lengths-hist.png", "Dialogue Lengths", 15)
        Draw_plots.plot_histograms({"real": turn_len}, self.dirname + "/turn-lengths-hist.png", "Turn Lengths", 15)

        system = self.stats.system_acts_count + 0.0
        user = self.stats.user_acts_count + 0.0
        Draw_plots.plot_stacked_bar_system_user({"real1": [system/(system+user), user/(system+user)],
                                                 "real2": [(system+500)/(system+user), (user-500)/(system+user)],
                                                 "real3": [(system-500)/(system+user), (user+500)/(system+user)]},
                                                self.dirname+"/system-user-actions.png", "System and User Actions")

        Draw_plots.plot_bar_next_to_one_another({"madeup1": {"a":1, "b":5, "c":6, 'd':2 },
                                                 "madeup2": {"a":2, "b":3, "c":5, 'd':2 },
                                                 "madeup3": {"a":3, "b":1, "c":4, 'd':2 },},
                                                ["blue", "green", "red", "yellow"],
                                                "Bars next to one another",
                                                self.dirname+"/picture.png")

    def get_dialogue_from_file(self, filename):
        dialogue = FileReader.read_file(filename)
        if dialogue:
            dialogue = Preprocessing.prepare_conversations(dialogue,
                Preprocessing.create_act_from_stack_use_last,
                Preprocessing.create_act_from_stack_use_last)
            Preprocessing.add_end_string(dialogue)
            Preprocessing.clear_numerics(dialogue)
            dialogue = [DialogueAct(d) for d in dialogue]
        return dialogue


class Dialogue_stats:

    def __init__(self):
        # Structures for dialogue statistics
        self.dialogue_lengths = numpy.array([], dtype=int)
        self.turn_lengths = numpy.array([], dtype=int)
        self.unique_acts = defaultdict(str)
        self.system_acts_count = 0
        self.user_acts_count = 0

    def update_structure(self, dialogue):
        self._append_dialogue_length(dialogue)
        self._append_turn_length(dialogue)
        self._append_unique_speech_acts(dialogue)
        self._append_system_user_acts(dialogue)

    def _append_system_user_acts(self, dialogue):
        d = dialogue
        while len(d) > 1:
            self.system_acts_count += len(d[0])
            self.user_acts_count += len(d[1])
            d = d[2:]

    def _append_unique_speech_acts(self, dialogue):
        uniq_local = defaultdict(str)
        for da in dialogue:
            for dai in da:
                if dai.value:
                    val = dai.value
                    dai.value = "&"
                uniq_local[unicode(dai)] = uniq_local.get(unicode(dai), 0)+1
                if dai.value:
                    dai.value = val

        for name, value in uniq_local.iteritems():
            n = self.unique_acts.get(name, numpy.array([]))
            self.unique_acts[name] = numpy.append(n, value)

    def _append_dialogue_length(self, dialogue):
        self.dialogue_lengths = numpy.append(self.dialogue_lengths, len(dialogue)/2)

    def _append_turn_length(self, dialogue):
        d = dialogue
        while len(d) > 1:
            self.turn_lengths = numpy.append(self.turn_lengths, len(d[0]) + len(d[1]))
            d = d[2:]


class Draw_plots:

    @staticmethod
    def _print_lines_to_file(filename, lines):
        f = codecs.open(filename, "w", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()

    @staticmethod
    def count_length_stats(numbers, filename):
        lines = ["name\tmean\tmedian\tstd_dev"]
        for name, nums in numbers.iteritems():
            lines.append(name+"\t"+str(numpy.mean(nums))+"\t"+str(numpy.median(nums))+"\t"+str(numpy.std(nums)))
        Draw_plots._print_lines_to_file(filename, lines)

    @staticmethod
    def plot_histograms(numbers, filename, title="", b=15):
        for name, nums in numbers.iteritems():
            plt.hist(nums, bins=b, histtype='step', label=name)

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_histograms_pointy(numbers, filename, title="", b=15):
        for name, nums in numbers.iteritems():
            y,bin = numpy.histogram(nums, bins=b)
            bincenters = bin[:-1] + (bin[1] - bin[0])/2
            plt.plot(bincenters, y, "-", label=name)

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_bar_next_to_one_another(nums, colors, title, filename):
        # simulator : data : number
        names = []
        data = defaultdict(str)

        ax = plt.subplot(111)
        for sim_name, data_n in nums.iteritems():
            names.append(sim_name)

            for n, d in data_n.iteritems():
                data[n] = data.get(n, [])
                data[n].append(d)
        i = 0
        w = 0.2
        N = len(names)
        ind = numpy.arange(N)
        for name, d in data.iteritems():
            ax.bar(ind+i*w, d, width=w, color=colors[0], label=name)
            i += 1
            colors = colors[1:]

        plt.xticks(ind+w*N/2., names)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_stacked_bar_system_user(numbers, filename, title=""):
        nums_system = []
        nums_user = []
        xlabels = []

        #tdo then many bars nex to each other
        for name, nums in numbers.iteritems():
            nums_system.append(nums[0])
            nums_user.append(nums[1])
            xlabels.append(name)

        N = len(numbers)
        ind = numpy.arange(N)    # the x locations for the groups
        width = 0.15       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, nums_system, width, color='r', align='center', label="System")#, yerr=womenStd)
        p2 = plt.bar(ind, nums_user, width, color='y', align='center', label="User", bottom=nums_system)#, yerr=menStd)

        plt.xticks(ind+width/2., xlabels)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_mean_lengths_stats(numbers, filename, title=""):
        means = []
        std_dev = []
        xlabels = []
        for name, nums in numbers.iteritems():
            means.append(numpy.mean(nums))
            std_dev.append(numpy.std(nums))
            xlabels.append(name)

        N = len(numbers)
        ind = numpy.arange(N)  # the x locations for the groups
        width = 0.10       # the width of the bar

        plt.bar(ind, means, width, align='center', color='r')#, yerr=std_dev)
        # add some text for labels, title and axes ticks
        plt.ylabel('Means')
        plt.title(title)
        plt.xticks(ind+width, xlabels)

        #ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )
        # def autolabel(rects):
        #     # attach some text labels
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
        #             ha='center', va='bottom')
        #
        # autolabel(rects1)
        # autolabel(rects2)

        plt.grid(True)
        plt.savefig(filename)
        plt.show()
        plt.close()

    # def count_dialogue_length_stats(self):
    #     #print histogram
    #     a = numpy.array(self.dialogue_lengths)
    #     print("Dialogue length stats:")
    #     print("min:", min(self.dialogue_lengths),
    #          "median:", numpy.median(self.dialogue_lengths),
    #          "mean:", numpy.mean(self.dialogue_lengths),
    #          "std dev:", numpy.std(self.dialogue_lengths),
    #          "max:", max(self.dialogue_lengths))
    #
    #     s = sorted(self.dialogue_lengths)
    #     q = len(self.dialogue_lengths) / 4
    #     h = len(self.dialogue_lengths) / 10
    #     print(0.25, s[1*q], 0.5, s[2*q], 0.75, s[3*q], 0.9, s[int(9*h)])
    #
    #     # the histogram of the data
    #     n, bins, patches = plt.hist(self.dialogue_lengths, 50, normed=1, facecolor='b')# , alpha=0.75)
    #
    #     plt.xlabel('Lengths')
    #     plt.ylabel('Amount')
    #     plt.title('Histogram of dialogue lengths')
    #     plt.axis([0, 50, 0, 0.15])
    #     plt.grid(True)
    #     plt.show()
    #     plt.savefig(self.dirname+"/dialogue-length-hist.png")
    #     plt.close()


#########################################################################
#########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        The program reads the default config in the resources directory ('../resources/default.cfg') config
        in the current directory.

        In addition, it reads all config file passed as an argument of a '-c'.
        The additional config files overwrites any default or previous values.""")

    parser.add_argument('-c', "--configs", nargs='+',
                        help='additional configuration files')
    parser.add_argument('-s', "--sim", type=str, required=True,
                        help='list of files with simulated dialogues')
    parser.add_argument('-t', "--test", type=str, required=True,
                        help='list of files with test dialogues')
    args = parser.parse_args()

    sim_list = args.sim
    test_list = args.test
    if not os.path.isfile(sim_list) or not os.path.isfile(test_list):
        print("WARNING: File list", "'"+sim_list+"'", "or", "'"+test_list+"'", "could not be found.", file=sys.stderr)
        parser.print_help()
        exit()

    cfg = Config.load_configs(args.configs)

    # #########################################################################
    cfg['Logging']['system_logger'].info("Dialogue evaluation\n" + "=" * 120)
    cfg['Logging']['system_logger'].session_start("localhost")
    cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))

    cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    cfg['Logging']['session_logger'].input_source("dialogue acts")

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    dirname = "eval/"+st+"eval"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    evaluator = Eval(cfg, dirname)
    evaluator.run(test_list, sim_list)