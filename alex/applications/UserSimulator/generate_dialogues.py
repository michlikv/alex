#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

import autopath

import argparse

from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.components.slu.exceptions import DialogueActException, DialogueActItemException
from alex.components.dm.common import dm_factory, get_dm_type
from alex.utils.config import Config

from Simulators import constantSimulator, simpleNgramSimulator, NgramSimulatorFiltered
from Generators.randomGenerator import RandomGenerator



class Generator:
    """
      Generator of dialogues generates specified amount of dialogues between selected Dialogue manager
      and User Simulator.

      It communicates in dialogue acts and produces text logs of the dialogues.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        dm_type = get_dm_type(cfg)
        self.dm = dm_factory(dm_type, cfg)

        self.simulator = None
        #TODO config user simulators somehow (?factory?)
        #self.bigram_filtered_init(cfg)\
        self.bigram_init(cfg)
        RandomGenerator()

    def constant_init(self):
        self.simulator = constantSimulator.ConstantSimulator()

    def bigram_init(self, cfg):
        self.simulator = simpleNgramSimulator.SimpleNgramSimulator(cfg)
        self.simulator.train_simulator('list-files-300.txt')

    def bigram_filtered_init(self, cfg):
        self.simulator = NgramSimulatorFiltered.NgramSimulatorFilterSlots(cfg)
        self.simulator.train_simulator('list-files-300.txt')

    def output_da(self, da):
        """Prints the system dialogue act to the output."""
        print "System DA:", unicode(da)
        print

    def output_nblist(self, nblist):
        """Prints the DA n-best list to the output."""
        print "User DA:", unicode(nblist.get_best_da())
        print

    def run(self):
        """Controls the dialogue manager and user simulator."""
        try:
            self.dm.new_dialogue()
            user_nblist = DialogueActNBList().add(1.0, DialogueAct())

            while unicode(user_nblist.get_best_da()).find('hangup()') == -1:
            #    self.cfg['Logging']['session_logger'].turn("system")

#               generate DM dialogue act
            #    self.dm.log_state()
                system_da = self.dm.da_out()
                self.output_da(system_da)

#               generate User dialogue act
            #    self.cfg['Logging']['session_logger'].turn("user")
                #TODO log state
                user_nblist = self.simulator.generate_response(system_da)
                self.output_nblist(user_nblist)
                #TODO log nb list?

#               pass it to the dialogue manager
                self.dm.da_in(user_nblist)
        except:
            self.cfg['Logging']['system_logger'].exception('Uncaught exception in Generation process.')
            raise

#########################################################################
#########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        Generator connects Dialogue Manager with User Simulator to generate dialogues.
        The output is logs from the simulated dialogues.

        The program reads the default config in the resources directory ('../resources/default.cfg') config
        in the current directory.

        In addition, it reads all config file passed as an argument of a '-c'.
        The additional config files overwrites any default or previous values.

      """)

    parser.add_argument('-c', "--configs", nargs='+',
                        help='additional configuration files')
    parser.add_argument('-n', "--num", type=int,
                        help='number of generated dialogues')
    args = parser.parse_args()

    cfg = Config.load_configs(args.configs)

    #########################################################################
    #########################################################################
    cfg['Logging']['system_logger'].info("User Simulator\n" + "=" * 120)

    # cfg['Logging']['system_logger'].session_start("localhost")
    # cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))
    #
    # cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    # cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    # cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    # cfg['Logging']['session_logger'].input_source("dialogue acts")

    generator = Generator(cfg)
    num_iter = args.num

    #todo for nejaky nastaveny pocet rozhovoru - zatim z comandliny
    for i in range(0,num_iter):
        generator.run()

    print "PYTHON Y U NOT END!"