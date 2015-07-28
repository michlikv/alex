#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

import autopath
import argparse
import ntpath

from alex.components.slu.da import DialogueAct, DialogueActNBList
from alex.utils.config import Config

from alex.components.slu.common import slu_factory
from Readers.FileReader import FileReader
from Readers.FileWriter import FileWriter
from Readers.Preprocessing import Preprocessing


class Transform:
    def __init__(self, cfg):
        self.cfg = cfg

    def transform(self, files_list, result_path):
        files_from = FileReader.read_file(files_list)

        for path in files_from:
            try:
                file_name = ntpath.basename(path)
                dialogue = FileReader.read_file(path)
                dialogue = Preprocessing.prepare_conversations(dialogue,
                                                               Preprocessing.create_act_from_stack_use_last,
                                                               Preprocessing.create_act_from_stack_use_last)

                # transform odd user positions with slu
                dialogue = [unicode(Preprocessing.convert_string_to_dialogue_acts(a, self.slu).get_best_da_hyp().da)
                            if i % 2 == 1 else a
                            for i, a in enumerate(dialogue)]

                # add user and system annotation
                dialogue = ['system: ' + a if i % 2 == 0 else 'user: ' + a for i, a in enumerate(dialogue)]

                # ADD SYSTEM: and USER:
                FileWriter.write_file(result_path + file_name, dialogue)
                cfg['Logging']['system_logger'].info("Transformed " + path)
            except:
                self.cfg['Logging']['system_logger'].info('Error: '+ path)

# ########################################################################
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
    parser.add_argument('-f', "--from_list", type=unicode,
                        help='list of files to be transformed')
    parser.add_argument('-t', "--to", type=unicode,
                        help='path to the folder to save files')
    args = parser.parse_args()

    cfg = Config.load_configs(args.configs)

    cfg['Logging']['system_logger'].info("Transform data\n" + "=" * 120)

    # cfg['Logging']['system_logger'].session_start("localhost")
    # cfg['Logging']['system_logger'].session_system_log('config = ' + unicode(cfg))
    #
    # cfg['Logging']['session_logger'].session_start(cfg['Logging']['system_logger'].get_session_dir_name())
    # cfg['Logging']['session_logger'].config('config = ' + unicode(cfg))
    # cfg['Logging']['session_logger'].header(cfg['Logging']["system_name"], cfg['Logging']["version"])
    # cfg['Logging']['session_logger'].input_source("dialogue acts")

    transformer = Transform(cfg)
    file_from = args.from_list
    path_to = args.to

    transformer.transform(file_from, path_to)