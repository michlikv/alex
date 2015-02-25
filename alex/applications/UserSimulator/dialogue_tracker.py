#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from Readers.FileReader import FileReader
from Readers.Preprocessing import Preprocessing
import time

import autopath

import argparse
import pprint

from alex.components.slu.da import DialogueAct, DialogueActNBList, DialogueActConfusionNetwork
from alex.components.dm.common import dm_factory, get_dm_type
from alex.utils.config import Config

from Simulators import constantSimulator, simpleNgramSimulator, NgramSimulatorFiltered
from Generators.randomGenerator import RandomGenerator
from StateTracking import Tracker


class Tracking_iface:

    def __init__(self, cfg):
        self.cfg = cfg


    def output_da(self, str, da):
        """Prints the system dialogue act to the output."""
        print str, unicode(da)
        self.cfg['Logging']['system_logger'].info(str + unicode(da))

    def output_nblist(self, str, nblist):
        """Prints the DA n-best list to the output."""
        self.cfg['Logging']['system_logger'].info(str + unicode(nblist.get_best_da()))
        print str, unicode(nblist.get_best_da())

    def run(self):
        """Tracks through passed dialogues."""
        #try:
        filename_filelist = 'data-lists/03-slu-500.txt'
        list_of_files = FileReader.read_file(filename_filelist)

        p = pprint.PrettyPrinter(indent=4)

        for file in list_of_files:
            print "processing file", file
            self.cfg['Logging']['system_logger'].info("processing file" + file)

            dialogue = FileReader.read_file(file)
            if dialogue:
                dialogue = Preprocessing.prepare_conversations(dialogue,
                                                               Preprocessing.create_act_from_stack_use_last,
                                                               Preprocessing.create_act_from_stack_use_last)
                Preprocessing.add_end_string(dialogue)
                dialogue = ['silence()'] + dialogue
                p.pprint(dialogue)

                self.cfg['Logging']['system_logger'].info(dialogue)

                self.tracker = Tracker.Tracker(self.cfg)

                while len(dialogue) > 1:
                    #user_da = DialogueActNBList().add(1.0, DialogueAct(dialogue[0]))
                    user_da = DialogueAct(dialogue[0])
                    self.output_da("User DA:", user_da)
                    self.output_da("System DA:", DialogueAct(dialogue[1]))
                    time.sleep(0.2)

                    self.tracker.update_state(user_da,
                                              DialogueAct(dialogue[1]))
                    self.tracker.log_state()

                    dialogue = dialogue[2:]
                    # nb = raw_input('enter to continue')

                if len(dialogue) > 0:
                    print dialogue
                    #except:
                    #   self.cfg['Logging']['system_logger'].info('Error: '+file)
        #except:
        #    self.cfg['Logging']['system_logger'].exception('Uncaught exception in Generation process.')
        #    raise

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

    #########################################################################
    #########################################################################
    cfg['Logging']['system_logger'].info("State tracker\n" + "=" * 120)

    cfg['Logging']['system_logger'].session_start("localhost")
    cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))

    cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    cfg['Logging']['session_logger'].input_source("dialogue acts")

    tracker = Tracking_iface(cfg)
    tracker.run()
