
from __future__ import unicode_literals

from alex.components.dm import Ontology
from alex.components.slu.da import DialogueActConfusionNetwork
from TrackerDDDState import DDDSTracker
from factory import tracker_factory

class Tracker:

    def __init__(self, cfg):
        # self.dialogue_state_class = cfg['UserSimulation']['dialogue_state']['type']
        self.cfg = cfg
        self.ontology = Ontology(cfg['UserSimulation']['ontology'])
        self.use_log = 'log' in cfg['UserSimulation'] and cfg['UserSimulation']['log']
        # self.dialogue_state = self.dialogue_state_class(cfg, self.ontology)
        self.dialogue_state = tracker_factory(cfg, self.ontology)
        #self.dialogue_state = DDDSTracker(cfg, self.ontology)

    def update_state(self, user_da, system_da):
        cn = DialogueActConfusionNetwork().make_from_da(user_da)
        self.dialogue_state.update(cn, system_da)

    def new_dialogue(self):
        self.dialogue_state = DDDSTracker(self.cfg, self.ontology)

    def log_state(self):
        if self.use_log:
        #print unicode(self.dialogue_state)
            self.dialogue_state.log_state()
        # print unicode(self.dialogue_state)

    def unicode_state(self):
        #print unicode(self.dialogue_state)
        return unicode(self.dialogue_state)

    def get_featurized_hash(self):
        return self.dialogue_state.get_featurized_hash()

    def get_value_said_system(self, slot):
        return  self.dialogue_state.get_value_said_system(slot)

    def get_value_said_user(self, slot):
        return  self.dialogue_state.get_value_said_user(slot)

    def get_compatible_values(self, dai, da):
        return self.dialogue_state.get_compatible_values(dai, da)
    #
    # def update_state(self, user_da, system_da):
    #     cn = DialogueActConfusionNetwork().make_from_da(system_da)
    #     self.dialogue_state.update(cn, user_da.get_best_da())
    #
    # def log_state(self):
    #     #print unicode(self.dialogue_state)
    #     self.dialogue_state.log_state()



