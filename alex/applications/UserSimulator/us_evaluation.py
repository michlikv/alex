#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals, print_function
from distutils.command.config import config

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
from copy import deepcopy

from alex.components.slu.da import DialogueAct
from alex.utils.config import Config
from alex.components.dm.common import dm_factory, get_dm_type

from Simulators.factory import simulator_factory_load
from Generators.randomGenerator import RandomGenerator
from Readers.FileReader import FileReader
from Readers.FileWriter import FileWriter
from Readers.Preprocessing import Preprocessing
from StateTracking.Tracker import Tracker

class Eval:

    def __init__(self, cfg, dirname, simulator_configs):
        self.cfg = cfg
        self.dirname = dirname
        #load simulator
        self.simulators = defaultdict(lambda: defaultdict(str))
        for name, data in simulator_configs.iteritems():
            self.simulators[name]['stats'] = DialogueStats(cfg, data[0])
            self.simulators[name]['data'] = data[1]

        self.stats = DialogueStats(cfg)
        self.test_dialogues = []

        #load DM
        dm_type = get_dm_type(cfg)
        self.dm = dm_factory(dm_type, cfg)
        #set random seed
        RandomGenerator()

    def _prepare_test_dialogues(self, test_list):
        self.test_dialogues = []
        listT = FileReader.read_file(test_list)
        for filename in listT:
            print('Processing:', filename)
            if os.path.isfile(filename):
                self.test_dialogues.append(self.get_dialogue_from_file(filename))
            else:
                print('Is not file:', filename)

    def _count_stats(self, list_dialogues, statsl):
        listT = FileReader.read_file(list_dialogues)
        for filename in listT:
            print('Processing:', filename)
            if os.path.isfile(filename):
                dialogue = self.get_dialogue_from_file(filename)
                statsl.update_structure(dialogue)
            else:
                print('Is not file:', filename)

    def run_prec_rec(self, test_list):
        # prepare test file
        self.cfg['Logging']['system_logger'].info("Counting test statistics")
        self._prepare_test_dialogues(test_list)
        for dialogue in self.test_dialogues:
            self.stats.update_structure(dialogue)

        #precision and recall finding stable number
        for i in range(0, 100):
            for name, simulator in self.simulators.iteritems():
                simulator['stats'].precisions = []
                simulator['stats'].recalls = []
                for dialogue in self.test_dialogues:
                    simulator['stats'].count_precision_recall(dialogue)
            print(i)
            #if i % 5 == 0:
            Draw_plots.print_mean_prec_rec_to_file(self.simulators, self.dirname+"/dialogue-prec-rec.txt")

    def run(self, test_list):
        # prepare test file
        self.cfg['Logging']['system_logger'].info("Counting test statistics")
        self._prepare_test_dialogues(test_list)
        for dialogue in self.test_dialogues:
            self.stats.update_structure(dialogue)

        dial_lengths = defaultdict(str)
        dial_lengths['TEST'] = self.stats.dialogue_lengths

        turn_lengths = defaultdict(str)
        turn_lengths['TEST'] = self.stats.turn_lengths

        user_turn_lengths = defaultdict(str)
        user_turn_lengths['TEST'] = self.stats.user_turn_lengths

        short_turn_lengths = defaultdict(str)
        short_turn_lengths['TEST'] = self.stats.shortened_turn_lengths

        num_con_info = defaultdict(str)
        num_con_info['TEST'] = self.stats.nums_con_info

        num_uniq_con_info = defaultdict(str)
        num_uniq_con_info['TEST'] = self.stats.nums_uniq_con_info

        num_apology = defaultdict(str)
        num_apology['TEST'] = self.stats.nums_apology

        num_correct = defaultdict(str)
        num_correct['TEST'] = self.stats.nums_correct

        num_incorrect = defaultdict(str)
        num_incorrect['TEST'] = self.stats.nums_incorrect


        system_user_ratio = defaultdict(str)
        system = self.stats.system_acts_count + 0.0
        user = self.stats.user_acts_count + 0.0
        system_user_ratio['TEST'] = [system/(system+user), user/(system+user)]

        # Draw_plots.print_lengths_tofile(self.stats.system_acts, self.stats.i_dial_count, self.dirname+"/TEST_system_act_count.txt")
        # Draw_plots.print_lengths_tofile(self.stats.user_acts, self.stats.i_dial_count, self.dirname+"/TEST_user_act_count.txt")


        for name, simulator in self.simulators.iteritems():
            filename_sim = simulator['data']
            stats_sim = simulator['stats']
            self._count_stats(filename_sim, stats_sim)

            dial_lengths[name] = stats_sim.dialogue_lengths
            turn_lengths[name] = stats_sim.turn_lengths
            user_turn_lengths[name] = stats_sim.user_turn_lengths
            short_turn_lengths[name] = stats_sim.shortened_turn_lengths
            num_con_info[name] = stats_sim.nums_con_info
            num_uniq_con_info[name] = stats_sim.nums_uniq_con_info
            num_apology[name] = stats_sim.nums_apology
            num_correct[name] = stats_sim.nums_correct
            num_incorrect[name] = stats_sim.nums_incorrect

            system = stats_sim.system_acts_count + 0.0
            user = stats_sim.user_acts_count + 0.0
            system_user_ratio[name] = [system/(system+user), user/(system+user)]

            # Draw_plots.print_lengths_tofile(stats_sim.system_acts, stats_sim.i_dial_count, self.dirname+"/"+ name+"_system_act_count.txt")
            # Draw_plots.print_lengths_tofile(stats_sim.user_acts, stats_sim.i_dial_count, self.dirname+"/"+name+"_user_act_count.txt")

        # dialogue lengths
        Draw_plots.count_length_stats(dial_lengths, self.dirname+"/dialogue-length.txt")
        Draw_plots.plot_mean_lengths_stats(dial_lengths, self.dirname + "/dialogue-lengths.png", "Dialogue Lengths")
        Draw_plots.plot_histogram_lines(dial_lengths, self.dirname + "/dial-lengths-hist.png", title="Dialogue Lengths",
                                        xlabel="Turns", ylabel="Amount of dialogues")
        #Draw_plots.plot_histograms_pointy(dial_lengths, self.dirname + "/dial-lengths-hist.png", "Dialogue Lengths", 15)

        # turn lengths
        Draw_plots.count_length_stats(turn_lengths, self.dirname+"/turn-length.txt")
        Draw_plots.plot_mean_lengths_stats(turn_lengths, self.dirname + "/turn-lengths.png", "Turn Lengths")
        Draw_plots.plot_histogram_lines(turn_lengths, self.dirname + "/turn-lengths-hist.png", title="Turn Lengths",
                                        xlabel="Dialogue act items", ylabel="Dialogues (%)")
        #Draw_plots.plot_histograms_pointy(turn_lengths, self.dirname + "/turn-lengths-hist.png", "Turn Lengths", 15)

        Draw_plots.count_length_stats(user_turn_lengths, self.dirname+"/user-turn-length.txt")
        Draw_plots.plot_mean_lengths_stats(user_turn_lengths, self.dirname + "/user-turn-lengths.png", "User Turn Lengths")
        Draw_plots.plot_histogram_lines(user_turn_lengths, self.dirname + "/user-turn-lengths-hist.png", title="User Turn Lengths",
                                        xlabel="Dialogue act items", ylabel="Dialogues (%)")

        Draw_plots.count_length_stats(short_turn_lengths, self.dirname+"/short-turn-length.txt")
        Draw_plots.plot_mean_lengths_stats(short_turn_lengths, self.dirname + "/short-turn-lengths.png", "Shortened Turn Lengths")
        Draw_plots.plot_histogram_lines(short_turn_lengths, self.dirname + "/short-turn-lengths-hist.png", title="Shortened Turn Lengths",
                                        xlabel="Dialogue act items", ylabel="Dialogues (%)")

        Draw_plots.count_length_stats_nonzero(num_con_info, self.dirname+"/num_con_info.txt")
        Draw_plots.plot_mean_lengths_stats_nonzero(num_con_info, self.dirname + "/num_con_info_bars.png", "Number of connection info per dialogue")
        Draw_plots.plot_histogram_lines(num_con_info, self.dirname + "/num_con_info.png", title="Number of connection info per dialogue",
                                        xlabel="Connection information", ylabel="Dialogues (%)")

        Draw_plots.count_length_stats_nonzero(num_uniq_con_info, self.dirname+"/num_uniq_con_info.txt")
        Draw_plots.plot_mean_lengths_stats_nonzero(num_uniq_con_info, self.dirname + "/num_uniq_con_info_bars.png", "Number of unique connection info per dialogue")
        Draw_plots.plot_histogram_lines(num_uniq_con_info, self.dirname + "/num_uniq_con_info.png", title="Number of unique connection info per dialogue",
                                        xlabel="Connection information", ylabel="Dialogues (%)")

        Draw_plots.count_freq_stats_nonzero(num_con_info, num_apology, self.dirname+"/num_con_apo_info.txt")

        Draw_plots.count_freq_stats_nonzero(num_correct, num_incorrect, self.dirname+"/num_con_corr_incorr.txt")

        #Draw_plots.count_length_stats(self.stats.unique_acts, self.dirname+"/Real-avg-acts.txt")

        #Draw_plots.plot_stacked_bar_system_user(system_user_ratio, self.dirname+"/system-user-actions.png",
        #                                        "System and User Actions")

        #Draw_plots.plot_bar_next_to_one_another({"madeup1": {"a": 1, "b": 5, "c": 6, 'd': 2 },
         #                                        "madeup2": {"a": 2, "b": 3, "c": 5, 'd': 2 },
          #                                       "madeup3": {"a": 3, "b": 1, "c": 4, 'd': 2 },},
           #                                     ["blue", "green", "red", "yellow"],
            #                                    "Bars next to each other",
             #                                   self.dirname+"/picture.png")

    def get_dialogue_from_file(self, filename):
        dialogue = FileReader.read_file(filename)
        if dialogue:
            dialogue = Preprocessing.prepare_conversations(dialogue,
                Preprocessing.create_act_from_stack_use_last,
                Preprocessing.create_act_from_stack_use_last)
            Preprocessing.clear_numerics(dialogue)
            dialogue = [DialogueAct(d) for d in dialogue]
        return dialogue


class DialogueStats:

    def __init__(self, cfg, simulator=None):
        # Structures for dialogue statistics
        self.dialogue_lengths = numpy.array([], dtype=int)

        self.turn_lengths = numpy.array([], dtype=int)
        self.user_turn_lengths = numpy.array([], dtype=int)
        self.shortened_turn_lengths = numpy.array([], dtype=int)

        self.nums_con_info = numpy.array([], dtype=int)
        self.nums_uniq_con_info = numpy.array([], dtype=int)
        self.nums_apology = numpy.array([], dtype=int)
        self.nums_correct = numpy.array([], dtype=int)
        self.nums_incorrect = numpy.array([], dtype=int)

        self.unique_acts = defaultdict(str)
        self.system_acts_count = 0
        self.system_acts = defaultdict(list)
        self.user_acts_count = 0
        self.user_acts = defaultdict(list)
        self.i_dial_count = 0

        self.precisions = []
        self.recalls = []

        self.simulator = simulator

        #load state tracker -- is needed for evaluation of correct connection information
        self.tracker = Tracker(cfg)

    def update_structure(self, dialogue):
        self._append_dialogue_length(dialogue)
        self._append_turn_length(dialogue)
        self._append_unique_speech_acts(dialogue)
        self._count_num_of_acts(dialogue)
        self.analyze_con_infos(dialogue)

    def analyze_con_infos(self, dialogue):
        d = [DialogueAct('silence()')] + dialogue
        num_con_info = 0
        uniq_con_info = set()
        apology = 0
        correct = 0
        incorrect = 0

        self.tracker.new_dialogue()

        while len(d) > 1:
            self.tracker.update_state(d[0], d[1])

            a = Preprocessing.shorten_connection_info(d[1])

            if unicode(d[1]) != unicode(a):
                num_con_info += 1
                if self.tracker.is_connection_correct():
                    correct += 1
                else:
                    incorrect += 1

                prefix = unicode(a)[:-16]
                if len(prefix) == 0 or unicode(d[1]).startswith(prefix):
                    con_inf = unicode(d[1])[len(prefix):]
                    uniq_con_info.add(con_inf)
                else:
                    raise

            if 'apology' in unicode(d[1]):
                apology += 1
            d = d[2:]

        self.nums_con_info = numpy.append(self.nums_con_info, num_con_info)
        self.nums_uniq_con_info = numpy.append(self.nums_uniq_con_info, len(uniq_con_info))
        self.nums_apology = numpy.append(self.nums_apology, apology)
        self.nums_correct = numpy.append(self.nums_correct, correct)
        self.nums_incorrect = numpy.append(self.nums_incorrect, incorrect)


    def count_precision_recall(self, dialogue):
        self._append_dialogue_precision_recall(dialogue)

    def _append_dialogue_precision_recall(self, dialogue):
        if self.simulator is not None:
            for i in range(1, len(dialogue), 2):
                sim_resp = self.simulator.generate_response_from_history(dialogue[:i])
                real_resp = dialogue[i]
                p, r = self._count_prec_rec(real_resp, sim_resp.get_best_da())
                self.precisions.append(p)
                self.recalls.append(r)

    def _count_prec_rec(self, real_da, sim_da):
        correct = 0
        cp_real = DialogueAct(unicode(real_da))
        cp_sim = DialogueAct(unicode(sim_da))
        Preprocessing.remove_slot_values(cp_real)
        Preprocessing.remove_slot_values(cp_sim)

        for dai in cp_sim:
            if dai in cp_real:
                correct += 1
        return correct/(len(sim_da)+0.0), correct/(len(real_da)+0.0),
        pass

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

    def _count_num_of_acts(self, dialogue):
        d = dialogue

        user_acts = defaultdict(int)
        system_acts = defaultdict(int)
        # count user acts, system acts -- to count average number of acts in dialogues
        while len(d) > 1:
            self.system_acts_count += len(Preprocessing.shorten_connection_info(d[0]))
            self.user_acts_count += len(d[1])
            a = Preprocessing.shorten_connection_info(d[0])

            for dai in a.dais:
                system_acts[dai.dat] += 1

            for dai in d[1].dais:
                user_acts[dai.dat] += 1

            d = d[2:]

        for name, val in system_acts.iteritems():
            self.system_acts[name].append(val)

        for name, val in user_acts.iteritems():
            self.user_acts[name].append(val)
        self.i_dial_count += 1

    def _append_turn_length(self, dialogue):
        d = dialogue

        while len(d) > 1:
            self.turn_lengths = numpy.append(self.turn_lengths, len(d[0]) + len(d[1]))
            self.user_turn_lengths = numpy.append(self.user_turn_lengths, len(d[1]))

            a = Preprocessing.shorten_connection_info(d[0])
            self.shortened_turn_lengths = numpy.append(self.shortened_turn_lengths, len(a)+len(d[1]))
            d = d[2:]

class Draw_plots:

    @staticmethod
    def plot_histogram_lines(hash_data, filename, styles=["ro-", "bs-", "g^-", "k+-"], title="", xlabel="", ylabel=""):
        dataX = defaultdict(str)
        dataY = defaultdict(str)

        for name, freqs in hash_data.iteritems():
            # count freq
            m = min(freqs)
            x = range(m, max(freqs)+1)
            counts = numpy.array([0.0]*len(x))
            for f in freqs:
                counts[f-m] += 1
            dataX[name] = x
            dataY[name] = counts/sum(counts)

        Draw_plots.plot_lines(filename, dataX, dataY, styles, title, xlabel, ylabel)

    @staticmethod
    def plot_lines(filename, hash_dataX, hash_data_y, styles=["ro-", "bs-", "g^-", "k+-"], title="", xlabel="", ylabel=""):
        plots = []
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0
        for name, x in hash_dataX.iteritems():
            y = hash_data_y[name]
            p, = plt.plot(x, y, styles[0], label=name)
            plots.append(p)
            minX = min([minX, min(x)])
            maxX = max([maxX, max(x)])
            minY = min([minY, min(y)])
            maxY = max([maxY, max(y)])
            styles = styles[1:]

        #plt.title(title)
        plt.axis([minX, maxX, minY, maxY])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(handles=plots)
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def print_mean_prec_rec_to_file(simulators, filename):
        lines = [] #["name\tprecision\trecall"]
        for name, simuls in simulators.iteritems():
            precs = simuls['stats'].precisions
            recs = simuls['stats'].recalls
            lines.append(name+"\t"+str(numpy.mean(precs))+"\t"+str(numpy.mean(recs))+"\t"+
                         str(simuls['stats'].simulator.get_oov()))
        FileWriter.write_file_append(filename, lines)
        pass


    @staticmethod
    def count_length_stats(numbers, filename):
        lines = ["name\tmean\tmedian\tstd_dev\tmin\tmax"]
        for name, nums in numbers.iteritems():
            lines.append(name+"\t"+str(numpy.mean(nums))+"\t"+str(numpy.median(nums))+"\t"+str(numpy.std(nums))+"\t" \
                         +str(numpy.min(nums))+"\t"+str(numpy.max(nums)))
        lines.append('')
        for name, nums in numbers.iteritems():
            lines.append(name+"\t"+str(nums))

        FileWriter.write_file(filename, lines)

    @staticmethod
    def print_lengths_tofile(numbers, total, filename):
        lines = ["total: "+ str(total)]
        for name, nums in numbers.iteritems():
            lines.append(str(name)+"\t"+str(nums))
        lines.append('')

        FileWriter.write_file(filename, lines)

    @staticmethod
    def count_freq_stats_nonzero(num_con_info, num_apology, filename):
        lines = ["name\t1\t2\t1and2\ttotal"]
        for name, nums_a in num_con_info.iteritems():
            nums_b = num_apology[name]

            a_b = [1 for a, b in zip(nums_a, nums_b) if a > 0 and b > 0]
            a = [1 for a in nums_a if a > 0]
            b = [1 for b in nums_b if b > 0]

            lines.append(name+"\t"+str(sum(a))+"\t"+str(sum(b))+"\t"+str(sum(a_b))+"\t" \
                         +str(len(nums_a)))
        lines.append('')

        for name, nums_a in num_con_info.iteritems():
            nums_b = num_apology[name]

            lines.append(name+' sum 1: '+str(numpy.sum(nums_a)))
            lines.append(name+' sum 2: '+str(numpy.sum(nums_b)))

        FileWriter.write_file(filename, lines)

    @staticmethod
    def count_length_stats_nonzero(numbers, filename):
        lines = ["name\tmean\tmedian\tstd_dev\tmin\tmax\t% of nonzero"]
        for name, nums_zeros in numbers.iteritems():
            nums = [a for a in nums_zeros if a > 0.0]
            lines.append(name+"\t"+str(numpy.mean(nums))+"\t"+str(numpy.median(nums))+"\t"+str(numpy.std(nums))+"\t" \
                         +str(numpy.min(nums))+"\t"+str(numpy.max(nums))+"\t"+str((0.0+len(nums))/len(nums_zeros)) )
        lines.append('')
        for name, nums in numbers.iteritems():
            lines.append(name+"\t"+str(nums))

        FileWriter.write_file(filename, lines)

    @staticmethod
    def plot_histograms(numbers, filename, title="", b=15):
        for name, nums in numbers.iteritems():
            plt.hist(nums, bins=b, histtype='step', normed=True, label=name)

        #plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_histograms_pointy(numbers, filename, title="", b=15):
        for name, nums in numbers.iteritems():
            y, bin = numpy.histogram(nums, bins=b, normed=True)
            bincenters = bin[:-1] + (bin[1] - bin[0])/2
            plt.plot(bincenters, y, "-", label=name)

        #plt.title(title)
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
        #plt.title(title)
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
        #plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    @staticmethod
    def plot_mean_lengths_stats_nonzero(numbers, filename, title=""):
        nums = defaultdict(str)
        for name, n in numbers.iteritems():
            nums[name] = [a for a in n if a > 0.0]
        Draw_plots.plot_mean_lengths_stats(nums, filename, title)

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

        plt.bar(ind, means, width, align='center', color='b', yerr=std_dev)
        # add some text for labels, title and axes ticks
        plt.ylabel('Means')
        #plt.title(title)
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
    parser.add_argument('-s', "--sim", type=str, required=True, nargs=2, action='append',
                        help='tuple of simulator config and a file with list of its dialogues')
    parser.add_argument('-p', "--precrec", default=False, action='store_true',
                        help='count only precision and recall')
    parser.add_argument('-t', "--test", type=str, required=True,
                        help='list of files with test dialogues')
    args = parser.parse_args()

    sim_lists = args.sim
    test_list = args.test
    if not os.path.isfile(test_list):
        print("ERROR: File list '"+test_list+"'could not be found.", file=sys.stderr)
        parser.print_help()
        exit(1)

    cfg = Config.load_configs(args.configs)

    simulators = defaultdict(str)
    for config, flist in sim_lists:
        if not os.path.isfile(flist) or not os.path.isfile(config):
            print("WARNING: File list '"+flist+"' or '"+config+"' could not be found.", file=sys.stderr)
            parser.print_help()
            exit(1)
        else:
            args.configs.append(config)
            cfg_s = Config.load_configs(args.configs)
            simulator = simulator_factory_load(cfg_s)
            simulators[cfg_s['UserSimulation']['short_name']] = [simulator, flist]

    # #########################################################################
    # cfg['Logging']['system_logger'].info("Dialogue evaluation\n" + "=" * 120)
    # cfg['Logging']['system_logger'].session_start("localhost")
    # cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))
    #
    # cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    # cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    # cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    # cfg['Logging']['session_logger'].input_source("dialogue acts")

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d--%H:%M:%S')

    dirname = "eval/"+st+"eval"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    evaluator = Eval(cfg, dirname, simulators)
    if args.precrec:
        evaluator.run_prec_rec(test_list)
    else:
        evaluator.run(test_list)