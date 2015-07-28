#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_type(cfg):
    """Return type of tracker

       :param cfg: configuration
    """
    if 'UserSimulation' in cfg and 'dialogue_state' in cfg['UserSimulation']:
        return cfg['UserSimulation']['dialogue_state']['type']
    else:
        return None

def tracker_factory(cfg, ontology):
    """Create state tracker

       :param cfg: configuration
       :param ontology: ontology
    """
    tr = None
    tr_type = get_type(cfg)

    try:
        tr = tr_type(cfg, ontology)
    except Exception, e:
        raise Exception('Error loading tracker %s: %s' % tr_type, e.message)

    return tr


