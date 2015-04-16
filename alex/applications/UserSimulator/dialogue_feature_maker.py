#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
import time
from collections import defaultdict
import autopath

import argparse
import pprint

from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActConfusionNetwork
from alex.utils.config import Config
from sklearn.feature_extraction import DictVectorizer
import codecs
from StateTracking import Tracker
from Simulators import MLsimulator

class Featurize:

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        #simulator = MLsimulator.MLsimulator(self.cfg)
        #simulator.train_simulator(self.cfg['UserSimulation']['files']['source'], False)
        simulator = MLsimulator.MLsimulator.load(self.cfg)
        simulator.make_stats_all("2015-04-15")
        #simulator.save()



class build_state_stats:

    def __init__(self):
        self.states = defaultdict(int)

    def add_acts(self, dialogue):
        for i,ut in enumerate(dialogue):
            if i%2 == 1:
                for dai in ut.dais:
                    self.states[unicode(dai)] += 1

    def print_sorted(self):
        for n, count in sorted(self.states.iteritems(), key=lambda (k,v): v,reverse=True):
            print(n, count)


#########################################################################
#########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        The program reads the default config in the resources directory ('../resources/default.cfg') config
        in the current directory.

        In addition, it reads all config file passed as an argument of a '-c'.
        The additional config files overwrites any default or previous values.

      """)

    parser.add_argument('-c', "--configs", nargs='+',
                        help='additional configuration files')
    args = parser.parse_args()

    cfg = Config.load_configs(args.configs)
    #
    # #########################################################################
    # #########################################################################
    # cfg['Logging']['system_logger'].info("State tracker\n" + "=" * 120)
    #
    cfg['Logging']['system_logger'].session_start("localhost")
    cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))

    # cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    # cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    # cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    # cfg['Logging']['session_logger'].input_source("dialogue acts")

    tracker = Featurize(cfg)
    tracker.run()