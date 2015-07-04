#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_sim_type(cfg):
    return cfg['UserSimulation']['type']

def simulator_factory_load(cfg):
    """
    Initialize Simulator from config file, only load models
    """
    sim = None
    sim_type = get_sim_type(cfg)

    try:
        sim = sim_type.load(cfg)
    except Exception, e:
        print e
   #     raise Exception('Unsupported simulator: %s' % sim_type)

    return sim

def simulator_factory_train(cfg):
    """
    Initialize Simulator from config file and train it using files and parameters specified in config
    """
    sim = None
    sim_type = get_sim_type(cfg)

    try:
        sim = sim_type(cfg)
        sim.train_simulator(cfg)
    except Exception, e:
        print e

    return sim


