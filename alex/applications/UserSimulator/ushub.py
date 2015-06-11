#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

import autopath
import time
import datetime
import argparse
import os.path

from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.components.dm.common import dm_factory, get_dm_type
from alex.utils.config import Config

from Simulators.factory import simulator_factory_load
from Generators.randomGenerator import RandomGenerator
from Readers.FileWriter import FileWriter

class Generator:
    """
      Generator of dialogues generates specified amount of dialogues between selected Dialogue manager
      and User Simulator.

      It communicates in dialogue acts and produces text logs of the dialogues.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.simulator = simulator_factory_load(cfg)

        dm_type = get_dm_type(cfg)
        self.dm = dm_factory(dm_type, cfg)
        self._do_error_model = 'ErrorModel' in self.cfg['UserSimulation']
        RandomGenerator()

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
#               generate DM dialogue act

                #self.dm.log_state()
                system_da = self.dm.da_out()
                self.output_da(system_da)
                dialogue.append("system: "+unicode(system_da))

                if 'none' in unicode(system_da):
                    pass

#               generate User dialogue act
                user_nblist = self.simulator.generate_response(system_da)
                self.output_nblist(user_nblist)

                if self._do_error_model:
                    user_nblist_clean = self.simulator.get_luda_nblist()
                    cfg['Logging']['system_logger'].info("Real Intended DA")
                    self.output_nblist(user_nblist_clean)
                else:
                    user_nblist_clean = user_nblist

#               pass it to the dialogue manager
                self.dm.da_in(user_nblist)

                dialogue.append("user: "+unicode(user_nblist_clean.get_best_da()))

                if with_state:
                    dialogue.append("\n"+self.simulator.get_state().unicode_state()+"\n")

            return dialogue
        except:
            self.cfg['Logging']['system_logger'].exception(dialogue)
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

    cfg['Logging']['system_logger'].session_start("localhost")
    cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))

    cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    cfg['Logging']['session_logger'].input_source("dialogue acts")

    generator = Generator(cfg)
    num_iter = args.num

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    dirname = "simulated/"+st+"-sim-"+cfg['UserSimulation']['type']
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    i = 1
    errors = 0
    while i <= num_iter:
        try:
            d = generator.run()
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            FileWriter.write_file(dirname+"/"+st+"-simulated-"+str(i), d)
            i += 1
        except:
            cfg['Logging']['system_logger'].exception('Exception in Generation process!')
            errors += 1
        print i
        print "Errors:", errors
    print "Errors:", errors
    print "."