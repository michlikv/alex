#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import autopath
import argparse

from alex.utils.config import Config
from Simulators.factory import simulator_factory_train

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
    cfg['Logging']['system_logger'].info("Training simulator\n" + "=" * 120)

    simulator = simulator_factory_train(cfg)
    #    simulator.make_stats_all("2015-04-15")
    simulator.save(cfg)
