#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_type(cfg):
    if 'UserSimulation' in cfg and 'dialogue_state' in cfg['UserSimulation']:
        return cfg['UserSimulation']['dialogue_state']['type']
    else:
        return None

def tracker_factory(cfg, ontology):
    """
    Initialize EM from config file, only load models
    """
    tr = None
    tr_type = get_type(cfg)

    try:
        tr = tr_type(cfg, ontology)
    except Exception, e:
        print e
   #     raise Exception('Unsupported simulator: %s' % sim_type)

    return tr


