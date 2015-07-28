#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals

def get_em_type(cfg):
    """Return type of EM

       :param cfg: configuration
    """
    if 'ErrorModel' in cfg:
        return cfg['ErrorModel']['type']
    else:
        return None

def error_model_factory_load(cfg):
    """Load EM from files in config file

       :param cfg: configuration
    """
    em = None
    em_type = get_em_type(cfg)

    try:
        em = em_type.load(cfg)
    except Exception, e:
        raise Exception('Error loading simulator %s: %s' % em_type, e)

    return em

def eror_model_factory_train(cfg):
    """Initialize EM from config file and train it using files and parameters specified in config

       :param cfg: configuration
    """
    em = None
    em_type = get_em_type(cfg)

    try:
        em = em_type(cfg)
        em.train(cfg)
    except Exception, e:
        raise Exception('Error training simulator %s: %s' % em_type, e)

    return em


