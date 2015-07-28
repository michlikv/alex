
from __future__ import unicode_literals

from alex.components.dm import Ontology
from alex.components.slu.da import DialogueActConfusionNetwork
from factory import tracker_factory

class Tracker:
    """ interface to the dialogue state
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ontology = Ontology(cfg['UserSimulation']['ontology'])
        self.use_log = 'log' in cfg['UserSimulation'] and cfg['UserSimulation']['log']
        self.dialogue_state = tracker_factory(cfg, self.ontology)

    def update_state(self, user_da, system_da):
        """ Update dialogue state using pair of user and system action

        :param user_da: user action
        :param system_da: system action
        """
        cn = DialogueActConfusionNetwork().make_from_da(user_da)
        self.dialogue_state.update(cn, system_da)

    def new_dialogue(self):
        """ Start new dialogue """
        self.dialogue_state = tracker_factory(self.cfg, self.ontology)

    def log_state(self):
        """ Log the state """
        if self.use_log:
            self.dialogue_state.log_state()

    def unicode_state(self):
        """ Return unicode of the state content """
        return unicode(self.dialogue_state)

    def get_featurized_hash(self):
        """ Return a hash structure with state features """
        return self.dialogue_state.get_featurized_hash()

    def get_value_said_system(self, slot):
        """ Get slot value that was said by the system

        :param slot: slot name
        :return: slot value or None
        """
        return self.dialogue_state.get_value_said_system(slot)

    def is_connection_correct(self):
        """ Check if connection information met the requirements.

        :return: boolean
        """
        return self.dialogue_state.is_connection_correct()

    def get_value_said_user(self, slot):
        """ Get slot value that was said by the user

        :param slot: slot name
        :return: slot value or None
        """
        return  self.dialogue_state.get_value_said_user(slot)

    def get_compatible_values(self, dai, da):
        """ Return compatible value to put into a DAI

        :param dai: DAI to fill in
        :param da: the DA of response that is being built
        :return:
        """
        return self.dialogue_state.get_compatible_values(dai, da)

