#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

import autopath
import time
import datetime
import argparse
import codecs
import os.path

from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.components.dm.common import dm_factory, get_dm_type
from alex.utils.config import Config

from Simulators import constantSimulator, simpleNgramSimulator, NgramSimulatorFiltered, MLsimulator
from Generators.randomGenerator import RandomGenerator


class Generator:
    """
      Generator of dialogues generates specified amount of dialogues between selected Dialogue manager
      and User Simulator.

      It communicates in dialogue acts and produces text logs of the dialogues.
    """

    def write_file(self, filename, lines):
        """
        Writes list of lines to utf-8 encoded file.
        :param filename: name of a file
        :param lines: lines to write
        """
        f = codecs.open(filename, "w", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()



    def __init__(self, cfg):
        self.cfg = cfg
        self.simulator = None
        self.ml_init(cfg)

        dm_type = get_dm_type(cfg)
        self.dm = dm_factory(dm_type, cfg)
        #TODO config user simulators from config :-O

        #self.bigram_init(cfg)
        RandomGenerator()

    def constant_init(self):
        self.simulator = constantSimulator.ConstantSimulator()

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

    def output_da(self, da):
        """Prints the system dialogue act to the output."""
        cfg['Logging']['system_logger'].info("System DA:"+unicode(da))

    def output_nblist(self, nblist):
        """Prints the DA n-best list to the output."""
        cfg['Logging']['system_logger'].info("User DA:"+unicode(nblist.get_best_da()))

    def run(self, with_state=False):
        """Controls the dialogue manager and user simulator."""
        try:
            dialogue = []

            self.simulator.new_dialogue()
            self.dm.new_dialogue()
            user_nblist = DialogueActNBList().add(1.0, DialogueAct())

            while unicode(user_nblist.get_best_da()).find('hangup()') == -1:
            #    self.cfg['Logging']['session_logger'].turn("system")
#               generate DM dialogue act
                self.dm.log_state()
                system_da = self.dm.da_out()
                self.output_da(system_da)
                dialogue.append("system: "+unicode(system_da))

#               generate User dialogue act
            #    self.cfg['Logging']['session_logger'].turn("user")
                user_nblist = self.simulator.generate_response(system_da)
                self.output_nblist(user_nblist)
                dialogue.append("user: "+unicode(user_nblist.get_best_da()))

                if with_state:
                    dialogue.append("\n"+self.simulator.get_state().unicode_state()+"\n")

#               pass it to the dialogue manager
                self.dm.da_in(user_nblist)
            return dialogue
        except:
            self.cfg['Logging']['system_logger'].exception('Uncaught exception in Generation process.')
            self.cfg['Logging']['system_logger'].exception(dialogue)
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

    cfg['Logging']['system_logger'].session_start("localhost")
    cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))

    cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    cfg['Logging']['session_logger'].input_source("dialogue acts")

    generator = Generator(cfg)
    #num_iter = args.num
    #todo pocet rozhovoru - z comandliny
    num_iter = 100

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    dirname = "simulated/"+st+"sim"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    i = 1
    for i in range(0, num_iter):
        d = generator.run()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        generator.write_file(dirname+"/"+st+"-simulated-"+str(i), d)
        i += 1
    print "."