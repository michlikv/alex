#!/usr/bin/env python
# -*- coding: utf-8 -*-


class TemplateClassifier(object):
    """
      This parser is based on matching examples of utterances with known
      semantics against input utterance.  The semantics of the example
      utterance which is closest to the input utterance is provided as a output
      semantics.

      "Hi"                                         => hello()
      "I can you give me a phone number"           => request(phone)
      "I would like to have a phone number please" => request(phone)

      The first match is reported as the resulting dialogue act.

    """
    def __init__(self, config):
        reg_exps = self.readRules(
            config['SLU']['TemplateParser']['TemplatesFile'])

    def readRules(self, file_name):
        pass

    def parse(self, asr_hyp):
        pass
