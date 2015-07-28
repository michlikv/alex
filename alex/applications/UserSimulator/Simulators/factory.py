#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_sim_type(cfg):
    """Return type of Simulator

       :param cfg: configuration
    """
    return cfg['UserSimulation']['type']

def simulator_factory_load(cfg):
    """Load Simulator from files in config file

       :param cfg: configuration
    """
    sim = None
    sim_type = get_sim_type(cfg)

    try:
        sim = sim_type.load(cfg)
    except Exception, e:
        raise Exception('Error loading simulator %s: %s' % sim_type, e.message)

    return sim

def simulator_factory_train(cfg):
    """Initialize Simulator from config file and train it using files and parameters specified in config

       :param cfg: configuration
    """
    sim = None
    sim_type = get_sim_type(cfg)

    try:
        sim = sim_type(cfg)
        sim.train_simulator(cfg)
    except Exception, e:
        raise Exception('Error training simulator %s: %s' % sim_type, e.message)

    return sim


