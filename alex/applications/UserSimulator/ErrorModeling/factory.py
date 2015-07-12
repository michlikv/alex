#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_em_type(cfg):
    if 'ErrorModel' in cfg:
        return cfg['ErrorModel']['type']
    else:
        return None

def error_model_factory_load(cfg):
    """
    Initialize EM from config file, only load models
    """
    em = None
    em_type = get_em_type(cfg)

    try:
        em = em_type.load(cfg)
    except Exception, e:
        print e
   #     raise Exception('Unsupported simulator: %s' % sim_type)

    return em

def eror_model_factory_train(cfg):
    """
    Initialize Simulator from config file and train it using files and parameters specified in config
    """
    em = None
    em_type = get_em_type(cfg)

    try:
        em = em_type(cfg)
        em.train(cfg)
    except Exception, e:
        print e

    return em


