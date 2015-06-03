#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
import constantSimulator, simpleNgramSimulator, NgramSimulatorFiltered, MLsimulator, UnigramSimulator, RandomSimulator

def get_sim_type(cfg):
    return cfg['UserSimulation']['type']

def simulator_factory_load(cfg):
    """
    Initialize Simulator from config file, only load models
    """
    sim = None
    sim_type = get_sim_type(cfg)

    # do not forget to maintain all supported dialogue managers
    if sim_type == 'MLSimulator':
        sim = MLsimulator.MLsimulator.load(cfg)
    elif sim_type == 'NgramSimulator':
        sim = simpleNgramSimulator.SimpleNgramSimulator.load(cfg)
    elif sim_type == 'NgramSimulatorFiltered':
        sim = NgramSimulatorFiltered.NgramSimulatorFilterSlots.load(cfg)
    elif sim_type == 'UnigramSimulator':
        sim = UnigramSimulator.UnigramSimulator.load(cfg)
    elif sim_type == 'RandomSimulator':
        sim = RandomSimulator.RandomSimulator.load(cfg)
    elif sim_type == 'ConstantSimulator':
        sim = constantSimulator.ConstantSimulator()
    else:
        raise Exception('Unsupported simulator: %s' % sim_type)
    return sim

def simulator_factory_train(cfg):
    """
    Initialize Simulator from config file and train it using files and parameters specified in config
    """
    sim = None
    sim_type = get_sim_type(cfg)

    # do not forget to maintain all supported dialogue managers
    if sim_type == 'MLSimulator':
        sim = MLsimulator.MLsimulator(cfg)
        sim.train_simulator(cfg, True)
    elif sim_type == 'NgramSimulator':
        sim = simpleNgramSimulator.SimpleNgramSimulator(cfg)
        sim.train_simulator(cfg)
    elif sim_type == 'NgramSimulatorFiltered':
        sim = NgramSimulatorFiltered.NgramSimulatorFilterSlots(cfg)
        sim.train_simulator(cfg)
    elif sim_type == 'UnigramSimulator':
        sim = UnigramSimulator.UnigramSimulator(cfg)
        sim.train_simulator(cfg)
    elif sim_type == 'RandomSimulator':
        sim = RandomSimulator.RandomSimulator(cfg)
        sim.train_simulator(cfg)
    elif sim_type == 'ConstantSimulator':
        sim = constantSimulator.ConstantSimulator()
    else:
        raise Exception('Unsupported simulator: %s' % sim_type)
    return sim


