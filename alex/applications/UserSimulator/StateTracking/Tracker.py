
from __future__ import unicode_literals

from alex.components.dm import Ontology
from alex.components.slu.da import DialogueActConfusionNetwork
from TrackerDDDState import DDDSTracker

class Tracker:

    def __init__(self, cfg):
        # self.dialogue_state_class = cfg['UserSimulation']['dialogue_state']['type']
        self.ontology = Ontology(cfg['UserSimulation']['ontology'])
        # self.dialogue_state = self.dialogue_state_class(cfg, self.ontology)
        self.dialogue_state = DDDSTracker(cfg, self.ontology)

    def update_state(self, user_da, system_da):
        cn = DialogueActConfusionNetwork().make_from_da(user_da)
        self.dialogue_state.update(cn, system_da)

    def log_state(self):
        #print unicode(self.dialogue_state)
        self.dialogue_state.log_state()
        # print unicode(self.dialogue_state)

    #
    # def update_state(self, user_da, system_da):
    #     cn = DialogueActConfusionNetwork().make_from_da(system_da)
    #     self.dialogue_state.update(cn, user_da.get_best_da())
    #
    # def log_state(self):
    #     #print unicode(self.dialogue_state)
    #     self.dialogue_state.log_state()



