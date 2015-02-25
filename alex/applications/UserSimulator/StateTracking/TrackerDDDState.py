#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from collections import defaultdict
from copy import deepcopy

from alex.components.dm.base import DiscreteValue, DialogueState
from alex.components.dm.dddstate import D3DiscreteValue
from alex.components.dm.exceptions import DeterministicDiscriminativeDialogueStateException
from alex.components.slu.da import DialogueAct, DialogueActItem, DialogueActNBList, DialogueActConfusionNetwork


class DDDSTracker(DialogueState):
    """
    State tracker fo user simulator.
    """

    def __init__(self, cfg, ontology):
        super(DDDSTracker, self).__init__(cfg, ontology)

        self.user_slots = defaultdict(D3DiscreteValue)
        # structures for remembering user dialogue acts
        # urh_ prefix
        self.user_request_history_slots = defaultdict(D3DiscreteValue)
        # uch_ prefix
        self.user_confirm_history_slots = defaultdict(D3DiscreteValue)
        # ush_ prefix
        self.user_select_history_slots = defaultdict(D3DiscreteValue)


        self.system_slots = defaultdict(D3DiscreteValue)
        # structures for remembering system dialogue acts
        # srh_ prefix
        self.system_request_history_slots = defaultdict(D3DiscreteValue)
        # sch_ prefix
        self.system_confirm_history_slots = defaultdict(D3DiscreteValue)
        # ssh_ prefix
        self.system_select_history_slots = defaultdict(D3DiscreteValue)
        # what system informed about
        # sih_ prefix
        self.system_informed_slots = defaultdict(D3DiscreteValue)

        self.last_system_da = DialogueAct("silence()")
        # last system/user dialogue act item type
        self.lsdait = D3DiscreteValue()
        self.ludait = D3DiscreteValue()

        self.all_lists = [self.user_slots,
                          self.user_request_history_slots,
                          self.user_confirm_history_slots,
                          self.user_select_history_slots,
                          self.system_slots,
                          self.system_request_history_slots,
                          self.system_confirm_history_slots,
                          self.system_select_history_slots,
                          self.system_informed_slots]

        self.turns = []
        self.turn_number = 0

        #todo settingy z kategorie USimulate
        self.debug = cfg['DM']['basic']['debug']
        self.type = cfg['DM']['DeterministicDiscriminativeDialogueState']['type']
        self.session_logger = cfg['Logging']['session_logger']
        self.system_logger = cfg['Logging']['system_logger']

    def __unicode__(self):
        """Get the content of the dialogue state in a human readable form."""
        s = ["D3State - Dialogue state content:", "",
             "{slot:20} = {value}".format(slot="ludait", value=unicode(self.ludait)),
             "{slot:20} = {value}".format(slot="lsdait", value=unicode(self.lsdait)), "USER SLOTS:"]

        #todo "and not sl.startswith('lta_')"

        #printing slot values
        str_pom = DDDSTracker.slots_dict_to_string(self.user_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.user_confirm_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.user_request_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.user_select_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        #printing slot values
        s.append("SYSTEM SLOTS:")
        str_pom = DDDSTracker.slots_dict_to_string(self.system_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.system_request_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.system_select_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.system_confirm_history_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        str_pom = DDDSTracker.slots_dict_to_string(self.system_informed_slots)
        if len(str_pom) > 0:
            s.extend(str_pom)

        s.append("")

        return '\n'.join(s)

    @staticmethod
    def slots_dict_to_string(slots):
        """
        :param slots:
        :return:string
        """
        s = []
        for name in slots:
            if isinstance(slots[name], D3DiscreteValue):
                s.append("{slot:20} = {value}".format(slot=name, value=unicode(slots[name])))
        return s

    def __getitem__(self, key):
        return self.user_slots[key]

    def __delitem__(self, key):
        del self.user_slots[key]

    def __setitem__(self, key, value):
        self.user_slots[key] = value

    def __contains__(self, key):
        return key in self.user_slots

    def __iter__(self):
        return iter(self.user_slots)

    def log_state(self):
        """Log the state using the the session logger."""
        state = [unicode(self)]
        self.session_logger.dialogue_state("system", [state, ])

    def restart(self):
        """Reinitialise the dialogue state so that the dialogue manager
        can start from scratch.

        Nevertheless, remember the turn history.
        """
        self.user_slots = defaultdict(D3DiscreteValue)
        self.user_request_history_slots.clear()
        self.user_confirm_history_slots.clear()
        self.user_select_history_slots.clear()
        self.system_slots.clear()
        self.system_request_history_slots.clear()
        self.system_confirm_history_slots.clear()
        self.system_select_history_slots.clear()
        self.system_informed_slots.clear()


    def update(self, user_da, system_da):
        """Dialogue act update.

        User_da is supposed to be a dialogue act confusion network
        System_da should be a dialogue act (?)

        :param user_da: Dialogue act to process.
        :type user_da: :class:`~alex.components.slu.da.DialogueAct`,
            :class:`~alex.components.slu.da.DialogueActNBList` or
            :class:`~alex.components.slu.da.DialogueActConfusionNetwork`
        :param system_da: Last system dialogue act.

        """
        if not isinstance(user_da, DialogueActConfusionNetwork):
            raise DeterministicDiscriminativeDialogueStateException("Unsupported input for the dialogue manager. "
                                                                    "User dialogue act is of incorrect type in "
                                                                    "DDDStateTracker.")

        if self.debug:
            self.system_logger.debug('D3State Dialogue Act in:\n%s\n%s' % (unicode(user_da), unicode(system_da)))

        if self.last_system_da != "silence()":
            user_da = self.context_resolution(user_da, self.last_system_da)

        if self.debug:
            self.system_logger.debug('Context Resolution - Dialogue Act:\n%s\n%s' % (user_da, system_da))

        if system_da == "silence()":
            # use the last non-silence dialogue act
            # if the system said nothing the last time, lets assume that the
            # user acts in the context of the previous dialogue act
            system_da = self.last_system_da
        else:
            # save the last non-silence dialogue act
            self.last_system_da = system_da

        # user_da = self.last_talked_about(user_da, system_da)
        #
        # if self.debug:
        #     self.system_logger.debug('Last Talked About Inference - Dialogue Act: \n%s' % user_da)

        # perform the state update
        self.state_update(user_da, system_da)
        self.turn_number += 1

        # store the result
        self.turns.append([deepcopy(user_da), deepcopy(system_da), deepcopy(self.all_lists)])

        # print the dialogue state if requested
        if self.debug:
            self.system_logger.debug(unicode(self))

    def context_resolution(self, user_da, system_da):
        """Resolves and converts meaning of some user dialogue acts
        given the context."""
        old_user_da = deepcopy(user_da)
        new_user_da = DialogueActConfusionNetwork()

        if isinstance(system_da, DialogueAct):
            for system_dai in system_da:
                for prob, user_dai in user_da:
                    new_user_dai = None

                    if system_dai.dat == "confirm" and user_dai.dat == "affirm":
                        new_user_dai = DialogueActItem("inform", system_dai.name, system_dai.value)

                    elif system_dai.dat == "confirm" and user_dai.dat == "negate":
                        new_user_dai = DialogueActItem("deny", system_dai.name, system_dai.value)

                    elif system_dai.dat == "request" and user_dai.dat == "inform" and \
                                    user_dai.name in self.ontology['context_resolution'] and \
                                    system_dai.name in self.ontology['context_resolution'][user_dai.name] and \
                                    user_dai.value == "dontcare":
                        new_user_dai = DialogueActItem("inform", system_dai.name, system_dai.value)

                    elif system_dai.dat == "request" and user_dai.dat == "inform" and \
                                    user_dai.name in self.ontology['context_resolution'] and \
                                    system_dai.name in self.ontology['context_resolution'][user_dai.name] and \
                                    self.ontology.slot_has_value(system_dai.name, user_dai.value):
                        new_user_dai = DialogueActItem("inform", system_dai.name, user_dai.value)

                    elif system_dai.dat == "request" and system_dai.name != "" and \
                                    user_dai.dat == "affirm" and self.ontology.slot_is_binary(system_dai.name):
                        new_user_dai = DialogueActItem("inform", system_dai.name, "true")

                    elif system_dai.dat == "request" and system_dai.name != "" and \
                                    user_dai.dat == "negate" and self.ontology.slot_is_binary(system_dai.name):
                        new_user_dai = DialogueActItem("inform", system_dai.name, "false")

                    if new_user_dai:
                        new_user_da.add(prob, new_user_dai)

        old_user_da.extend(new_user_da)

        return old_user_da

    # def last_talked_about(self, user_da, system_da):
    #     """This adds dialogue act items to support inference of the last slots the user talked about."""
    #     old_user_da = deepcopy(user_da)
    #     new_user_da = DialogueActConfusionNetwork()
    #
    #     for prob, user_dai in user_da:
    #         new_user_dais = []
    #         lta_tsvs = self.ontology.last_talked_about(user_dai.dat, user_dai.name, user_dai.value)
    #
    #         for name, value in lta_tsvs:
    #             new_user_dais.append(DialogueActItem("inform", name, value))
    #
    #         if new_user_dais:
    #             for nudai in new_user_dais:
    #                 new_user_da.add(prob, nudai)
    #
    #     old_user_da.extend(new_user_da)
    #
    #     return old_user_da

    def state_update(self, user_da, system_da):
        """Records the information provided by the system and by the user."""

        # since there is a state update, the silence_time from the last from the user voice activity is 0.0
        # unless this update fired just to inform about the silence time. This case is taken care of later.
        # - this slot is not probabilistic
        self.user_slots['silence_time'] = 0.0

        # process the user dialogue act
        # processing the low probability DAIs first, emphasize the dialogue acts with high probability
        for prob, dai in sorted(user_da.items()):
            #print "#0 ", self.type
            #print "#1 SType:", prob, dai
            ##print "#51", self.slots

            if self.type == "MDP":
                if prob >= 0.5:
                    weight = 0.0
                else:
                    continue
            else:
                weight = 1.0 - prob

            if dai.dat == "inform":
                if dai.name:
                    self.user_slots[dai.name].scale(weight)
                    self.user_slots[dai.name].add(dai.value, prob)
                    if "srh_" + dai.name in self.system_request_history_slots:
                        self.system_request_history_slots["srh_" + dai.name].scale(weight)
                        self.system_request_history_slots["srh_" + dai.name].add("user-informed", prob)
                    if "sch_" + dai.name in self.system_confirm_history_slots:
                         self.system_confirm_history_slots["sch_" + dai.name].scale(weight)
                         self.system_confirm_history_slots["sch_" + dai.name].add("user-informed", prob)

            elif dai.dat == "deny":
                # handle true and false values because we know their opposite values
                if dai.value == "true" and self.ontology.slot_is_binary(dai.name):
                    self.user_slots[dai.name].scale(weight)
                    self.user_slots[dai.name].add('false', prob)
                elif dai.value == "false" and self.ontology.slot_is_binary(dai.name):
                    self.user_slots[dai.name].scale(weight)
                    self.user_slots[dai.name].add('true', prob)
                else:
                    self.user_slots[dai.name].distribute(dai.value, prob)
            elif dai.dat == "request":
                self.user_request_history_slots["urh_" + dai.name].scale(weight)
                self.user_request_history_slots["urh_" + dai.name].add("user-requested", prob)
            elif dai.dat == "confirm":
                self.user_confirm_history_slots["uch_" + dai.name].scale(weight)
                self.user_confirm_history_slots["uch_" + dai.name].add(dai.value, prob)
            elif dai.dat == "select":
                self.user_select_history_slots["ush_" + dai.name].scale(weight)
                self.user_select_history_slots["ush_" + dai.name].add(dai.value, prob)
            elif dai.dat in set(["ack", "apology", "bye", "hangup", "hello", "help", "null", "other",
                             "repeat", "reqalts", "reqmore", "restart", "thankyou"]):
                self.ludait.scale(weight)
                self.ludait.add(dai.dat, prob)
            elif dai.dat == "silence":
                self.ludait.scale(weight)
                self.ludait.add(dai.dat, prob)
                if dai.name == "time":
                    self.user_slots['silence_time'] = float(dai.value)

        weight = 0.0;
        #system dialogue act in this case is a reaction to a previous user act!
        if isinstance(system_da, DialogueAct):
            # self.system_request_history_slots.clear()
            #todo affirm a deny a negate...???

            for dai in system_da:

                if dai.name and dai.value:
                    self.system_slots[dai.name].scale(weight)
                    self.system_slots[dai.name].add(dai.value, prob)

                if dai.dat == "inform":
                    # set that the system already informed about the slot
                    self.user_request_history_slots["urh_" + dai.name].set({"system-informed": 1.0, })
                    self.user_confirm_history_slots["uch_" + dai.name].set({"system-informed": 1.0, })
                    self.user_select_history_slots["ush_" + dai.name].set({"system-informed": 1.0, })
                    if dai.value:
                        self.system_informed_slots["sih_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "iconfirm" or dai.dat == "confirm":
                    # set that the system already informed about the slot
                    self.user_request_history_slots["urh_" + dai.name].set({"system-informed": 1.0, })
                    self.user_confirm_history_slots["uch_" + dai.name].set({"system-informed": 1.0, })
                    self.user_select_history_slots["ush_" + dai.name].set({"system-informed": 1.0, })
                    if dai.value:
                        self.system_confirm_history_slots["sch_" + dai.name].scale(weight)
                        self.system_confirm_history_slots["sch_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "select":
                    self.system_select_history_slots["ssh_" + dai.name].scale(weight)
                    self.system_select_history_slots["ssh_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "request":
                    self.system_request_history_slots["srh_" + dai.name].scale(weight)
                    self.system_request_history_slots["srh_" + dai.name].add('system-requested', 1.0)

                elif dai.dat in set(["silence", "apology", "bye", "hangup", "hello", "help", "null", "other",
                             "irepeat", "notunderstood", "reqmore", "restart", "help", ]):
                    self.lsdait.set({dai.dat: prob, })

    def get_slots_being_requested(self, req_prob=0.8):
        """Return all slots which are currently being requested by the user along with the correct value."""
        requested_slots = {}

        for slot in self.user_request_history_slots:
            if isinstance(self.user_request_history_slots[slot], D3DiscreteValue) and slot.startswith("urh_"):

                if self.user_slots[slot]["user-requested"] > req_prob:
                    if slot[3:] in self.user_slots:
                        requested_slots[slot[3:]] = self.user_slots[slot[3:]]
                    else:
                        requested_slots[slot[3:]] = "none"

        return requested_slots

    def get_slots_being_confirmed(self, conf_prob=0.8):
        """Return all slots which are currently being confirmed by the user along with the value being confirmed."""
        confirmed_slots = {}

        for slot in self.user_confirm_history_slots:
            if isinstance(self.user_slots[slot], D3DiscreteValue) and slot.startswith("uch_"):
                prob, value = self.user_slots[slot].mph()
                if value not in ['none', 'system-informed', None] and prob > conf_prob:
                    confirmed_slots[slot[3:]] = self.user_slots[slot[3:]]

        return confirmed_slots

    def get_slots_being_noninformed(self, noninf_prob=0.8):
        """Return all slots provided by the user and the system has not informed about them yet along with
        the value of the slot.

        This will not detect a change in a goal. For example::

            U: I want a Chinese restaurant.
            S: Ok, you want a Chinese restaurant. What price range you have in mind?
            U: Well, I would rather want an Italian Restaurant.
            S: Ok, no problem. You want an Italian restaurant. What price range you have in mind?

        Because the system informed about the food type and stored "system-informed", then
        we will not notice that we confirmed a different food type.
        """
        noninformed_slots = {}

        for slot in self.user_slots:
            if not isinstance(self.user_slots[slot], D3DiscreteValue):
                continue

            # test whether the slot is not currently requested
            if "rh_" + slot not in self.user_request_history_slots or self.user_request_history_slots["rh_" + slot]["none"] > 0.999:
                prob, value = self.user_slots[slot].mph()
                # test that the non informed value is an interesting value
                if value not in ['none', None] and prob > noninf_prob:
                    noninformed_slots[slot] = self.user_slots[slot]

        return noninformed_slots

    def get_accepted_slots(self, acc_prob):
        """Returns all slots which have a probability of a non "none" value larger then some threshold.
        """
        accepted_slots = {}

        for slot in self.user_slots:
            if not isinstance(self.user_slots[slot], D3DiscreteValue):
                continue

            prob, value = self.user_slots[slot].mph()
            if value not in ['none', 'system-informed', None] and prob >= acc_prob:
                accepted_slots[slot] = self.user_slots[slot]

        return accepted_slots

    def get_slots_tobe_confirmed(self, min_prob, max_prob):
        """Returns all slots which have a probability of a non "none" value larger then some threshold and still not so
        large to be considered as accepted.
        """
        tobe_confirmed_slots = {}

        for slot in self.user_slots:
            if not isinstance(self.user_slots[slot], D3DiscreteValue):
                continue

            prob, value = self.user_slots[slot].mph()
            if value not in ['none', 'system-informed', None] and min_prob <= prob and prob < max_prob:
                tobe_confirmed_slots[slot] = self.user_slots[slot]

        return tobe_confirmed_slots

    def get_slots_tobe_selected(self, sel_prob):
        """Returns all slots which have a probability of the two most probable non "none" value larger then some threshold.
        """
        tobe_selected_slots = {}

        for slot in self.user_slots:
            if not isinstance(self.user_slots[slot], D3DiscreteValue):
                continue

            (prob1, value1), (prob2, value2) = self.user_slots[slot].tmphs()

            if value1 not in ['none', 'system-informed', None] and prob1 > sel_prob and \
                value2 not in ['none', 'system-informed', None] and prob2 > sel_prob:
                tobe_selected_slots[slot] = self.user_slots[slot]

        return tobe_selected_slots

    def get_changed_slots(self, cha_prob):
        """Returns all slots that have changed from the previous turn. Because the change is determined by change in
        probability for a particular value, there may be very small changes. Therefore, this will only report changes
        for values with a probability larger than the given threshold.

        :param cha_prob: minimum current probability of the most probable hypothesis to be reported
        :rtype: dict
        """
        changed_slots = {}

        # compare the accepted slots from the previous and the current turn
        if len(self.turns) >= 2:
            cur_slots = self.turns[-1][2][0]
            prev_slots = self.turns[-2][2][0]

            for slot in cur_slots:
                if not isinstance(cur_slots[slot], D3DiscreteValue):
                    continue

                cur_prob, cur_value = cur_slots[slot].mph()
                prev_prob, prev_value = prev_slots[slot].mph()

                if cur_value not in ['none', 'system-informed', None] and cur_prob > cha_prob and \
                    prev_value not in ['system-informed', None] and \
                    cur_value != prev_value:
                    #prev_prob > cha_prob and \ # only the current value must be accepted
                    changed_slots[slot] = cur_slots[slot]

            return changed_slots
        elif len(self.turns) == 1:
            # after the first turn all accepted slots are effectively changed
            return self.get_accepted_slots(cha_prob)
        else:
            return {}

    def state_changed(self, cha_prob):
        """Returns a boolean indicating whether the dialogue state changed significantly
        since the last turn. True is returned if at least one slot has at least one value
        whose probability has changed at least by the given threshold since last time.

        :param cha_prob: minimum probability change to be reported
        :rtype: Boolean
        """
        if len(self.turns) >= 2:
            cur_all_slots = self.turns[-1][2]
            prev_all_slots = self.turns[-2][2]

            for cur_slots,prev_slots in zip(cur_all_slots, prev_all_slots):
                for slot in cur_slots:
                    if not isinstance(cur_slots[slot], D3DiscreteValue):
                        continue

                    for value, cur_prob in cur_slots[slot].items():
                        if value in ['none', 'system-informed', None]:
                            continue
                        prev_prob = prev_slots[slot].get(value, 0.0)
                        if abs(cur_prob - prev_prob) > cha_prob:
                            return True
        elif len(self.turns) == 1:
            slots = self.turns[-1][2]
            for slot in slots:
                if not isinstance(slots[slot], D3DiscreteValue):
                    continue
                prob, value = slots[slot].mph()
                if value in ['none', 'system-informed', None]:
                    continue
                if prob > cha_prob:
                    return True
            pass
        return False
