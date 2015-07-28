#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from collections import defaultdict
from copy import deepcopy
from Readers.Preprocessing import Preprocessing

from alex.components.dm.base import DiscreteValue, DialogueState
from alex.components.dm.dddstate import D3DiscreteValue
from alex.components.dm.exceptions import DeterministicDiscriminativeDialogueStateException
from alex.components.slu.da import DialogueAct, DialogueActItem, DialogueActNBList, DialogueActConfusionNetwork


class DDDSTracker(DialogueState):
    """
    State tracker for user simulator.
    It is atapted from DeterministicDiscriminativeDialogueState
    """

    def __init__(self, cfg, ontology):
        super(DDDSTracker, self).__init__(cfg, ontology)

        self.user_slots = defaultdict(D3DiscreteValue)
        # structures for remembering user dialogue acts
        # urh_ prefix
        self.user_request_history_slots = defaultdict(D3DiscreteValue)
        # uch_ prefix
        self.user_confirm_history_slots = defaultdict(D3DiscreteValue)

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

        self.connection_info_said = False
        self.connection_info_matches = False
        self.last_connection_alternative = 1
        self.connection_error = False

        # last system/user dialogue act item type
        self.lsdait = D3DiscreteValue()
        self.ludait = D3DiscreteValue()

        self.all_lists = [self.user_slots,
                          self.user_request_history_slots,
                          self.user_confirm_history_slots,
                          #self.user_select_history_slots,
                          self.system_slots,
                          self.system_request_history_slots,
                          self.system_confirm_history_slots,
                          self.system_select_history_slots,
                          self.system_informed_slots]

        self.turns = []
        self.turn_number = 0

        if 'debug' in cfg['UserSimulation']:
            self.debug = cfg['UserSimulation']['debug']
        else:
            self.debug = False

        self.type = cfg['DM']['DeterministicDiscriminativeDialogueState']['type']
        self.session_logger = cfg['Logging']['session_logger']
        self.system_logger = cfg['Logging']['system_logger']

    def __unicode__(self):
        """Get the content of the dialogue state in a human readable form."""
        s = ["D3State - Dialogue state content:", "",
             "{slot:20} = {value}".format(slot="ludait", value=unicode(self.ludait)),
             "{slot:20} = {value}".format(slot="lsdait", value=unicode(self.lsdait)),
             "{slot:20} = {value}".format(slot="con_info", value=unicode(self.connection_info_said)),
             "{slot:20} = {value}".format(slot="con_error", value=unicode(self.connection_error)),
             "{slot:20} = {value}".format(slot="con_match", value=unicode(self.connection_info_matches)),
             "{slot:20} = {value}".format(slot="last_alt", value=unicode(self.last_connection_alternative)),
             "USER SLOTS:"]

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
        """ build a string from hash

        :param slots: hash
        :return: str
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
        can start from scratch. Remember the turn history.
        """
        self.user_slots = defaultdict(D3DiscreteValue)
        self.user_request_history_slots.clear()
        self.user_confirm_history_slots.clear()
        self.system_slots.clear()
        self.system_request_history_slots.clear()
        self.system_confirm_history_slots.clear()
        self.system_select_history_slots.clear()
        self.system_informed_slots.clear()
        self.last_system_da = DialogueAct("silence()")
        self.connection_info_said = False
        self.connection_error = False
        self.connection_info_matches = False
        self.last_connection_alternative = 1

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

        this_system_da = deepcopy(system_da)

        for dai in system_da:
            if dai.dat == "apology":
                self.connection_error = True

        for dai in system_da:
            if dai.dat == "restart":
                self.restart()

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

        # track information about connection info
        shortened = Preprocessing.shorten_connection_info(this_system_da)
        if unicode(shortened) != unicode(this_system_da):
            self.connection_info_said = True
            self.connection_error = False
            self.connection_info_matches = self.is_connection_info_consistent(this_system_da)


        self.turn_number += 1

        # store the result
        self.turns.append([deepcopy(user_da), deepcopy(system_da), deepcopy(self.all_lists)])

        # print the dialogue state if requested
        if self.debug:
            self.system_logger.debug(unicode(self))

    def is_connection_info_consistent(self, da):
        has_er = False

        # #check alternatives
        # if 'alternative' in self.user_slots:
        #     said_alternative = 1
        #     for dai in da:
        #         # if da is inform(alternative="some positive number") and user specified alternative
        #         if dai.dat == 'inform' and dai.name and dai.name == 'alternative' and dai.value and dai.value.isdigit():
        #             said_alternative = int(dai.value)
        #
        #     wanted_num = 0
        #     user_val = self.user_slots['alternative'].mph()[1]
        #     if user_val.isdigit():
        #         wanted_num = int(user_val)
        #     elif user_val == 'next':
        #         wanted_num = self.last_connection_alternative +1
        #     elif user_val == 'prev' and self.last_connection_alternative >1:
        #         wanted_num = self.last_connection_alternative -1
        #
        #     if user_val == 'last' and said_alternative == 1:
        #         has_er = True # It is not known how many alternatives were found
        #     #if values mismatch
        #     if wanted_num != 0 and wanted_num != said_alternative:
        #         has_er = True
        #
        #     self.last_connection_alternative = said_alternative

        # #check vehicle
        # if 'vehicle' in self.user_slots:
        #     wanted_value = self.user_slots['vehicle'].mph()[1]
        #     contains_value = False
        #     for dai in da:
        #         if dai.dat == 'inform' and dai.name and dai.name == 'vehicle' and dai.value and  wanted_value.lower() in dai.value.lower():
        #             contains_value = True
        #             break
        #     # if there is not a correct vehicle even once, the connection is incorrect
        #     if not contains_value:
        #         has_er = True

        #check from and to stop
        said_from_stop = None
        walkto_from = False
        said_to_stop = None
        walkto_to = False

        for dai in da:
            if dai.value and dai.name and dai.name == 'exit_at':
               said_to_stop = dai.value
            elif dai.value and dai.name and dai.name == 'walk_to' and said_from_stop is None:
                walkto_from = True
                said_from_stop = dai.value
            elif dai.value and dai.name and dai.name == 'enter_at' and said_from_stop is None:
                said_from_stop = dai.value
            elif dai.value and dai.name and dai.name == 'walk_to' and dai.value == 'FINAL_DEST':
                walkto_to = True

        from_stop = self.user_slots['from_stop'].mph()[1] if 'from_stop' in self.user_slots else None
        from_city = self.user_slots['from_city'].mph()[1] if 'from_city' in self.user_slots else None
        to_stop = self.user_slots['to_stop'].mph()[1] if 'to_stop' in self.user_slots else None
        to_city = self.user_slots['to_city'].mph()[1] if 'to_city' in self.user_slots else None

        #if from_stop is None and from_city is None:
        #    has_er = True
        #if to_stop is None and to_city is None:
        #    has_er = True

        if not self._is_compatible_stops(said_from_stop, walkto_from, from_stop, from_city):
            has_er = True
        if not self._is_compatible_stops(said_to_stop, walkto_to, to_stop, to_city):
            has_er = True

        return not has_er

    def _is_compatible_stops(self, said_stop, walkto, stop, city):
        if not walkto:
            if said_stop and stop and (stop.lower() not in said_stop.lower() and said_stop.lower() not in stop.lower()):
                return False
        else:
            #we should walk to the destination
            if said_stop and stop and not self._is_stops_in_same_city(said_stop, stop):
                return False
        if said_stop and city and not self._is_stop_in_city(said_stop, city):
            return False
        return True

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

        old_user_da.merge(new_user_da, combine='max')

        return old_user_da

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
            # elif dai.dat == "select":
            #     self.user_select_history_slots["ush_" + dai.name].scale(weight)
            #     self.user_select_history_slots["ush_" + dai.name].add(dai.value, prob)
            elif dai.dat in set(["ack", "apology", "bye", "hangup", "hello", "help", "null", "other",
                             "repeat", "reqalts", "reqmore", "restart", "thankyou"]):
                self.ludait.scale(weight)
                self.ludait.add(dai.dat, prob)
            elif dai.dat == "silence":
                self.ludait.scale(weight)
                self.ludait.add(dai.dat, prob)
                if dai.name == "time":
                    self.user_slots['silence_time'] = float(dai.value)

        weight = 0.0
        #system dialogue act in this case is a reaction to a previous user act!
        if isinstance(system_da, DialogueAct):
            for dai in system_da:

                if dai.name and dai.value and dai.dat != 'help':
                    self.system_slots[dai.name].scale(weight)
                    self.system_slots[dai.name].add(dai.value, prob)

                if dai.dat == "inform":
                    # set that the system already informed about the slot
                    self.user_request_history_slots["urh_" + dai.name].set({"system-informed": 1.0, })
                    #self.user_confirm_history_slots["uch_" + dai.name].set({"system-informed": 1.0, })
                    #self.user_select_history_slots["ush_" + dai.name].set({"system-informed": 1.0, })
                    if dai.value:
                        self.system_informed_slots["sih_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "iconfirm" or dai.dat == "confirm":
                    # set that the system already informed about the slot
                    self.user_request_history_slots["urh_" + dai.name].set({"system-informed": 1.0, })
                    #self.user_confirm_history_slots["uch_" + dai.name].set({"system-informed": 1.0, })
                    #self.user_select_history_slots["ush_" + dai.name].set({"system-informed": 1.0, })
                    if dai.value:
                        self.system_confirm_history_slots["sch_" + dai.name].scale(weight)
                        self.system_confirm_history_slots["sch_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "select":
                    self.system_select_history_slots["ssh_" + dai.name].scale(weight)
                    self.system_select_history_slots["ssh_" + dai.name].add(dai.value, 1.0)

                if dai.dat == "request":
                    self.system_request_history_slots["srh_" + dai.name].scale(weight)
                    self.system_request_history_slots["srh_" + dai.name].add('system-requested', 1.0)

                if dai.dat == "help":
                    if dai.name:
                        self.system_slots["help_"+dai.name].scale(weight)
                        if dai.value:
                            self.system_slots["help_"+dai.name].add(dai.value, prob)
                        else:
                            self.system_slots["help_"+dai.name].add("noval", prob)
                    else:
                        self.lsdait.set({dai.dat: prob, })

                elif dai.dat in set(["silence", "apology", "bye", "hangup", "hello", "null", "other",
                             "irepeat", "notunderstood", "reqmore", "restart" ]):
                    self.lsdait.set({dai.dat: prob, })

    def _hash_values(self, hash):
        result = defaultdict(str)
        # slots with prefix "uch_"
        # has either slot value or "system-informed".
        for slot, value in hash.iteritems():
            if value.mph()[1] == 'none':
                next
            result[slot+"_in"] = "used"
            prob, val = hash[slot].mph()
            if val in ["system-informed", "user-informed"]:
                result[slot] = val
            elif slot[4:] in self.user_slots and val == self.user_slots[slot[4:]].mph()[1]:
                result[slot] = "user-value"
            elif slot[4:] in self.system_slots and val == self.system_slots[slot[4:]].mph()[1]:
                result[slot] = "system-value"
            else:
                result[slot] = "other"
        return result

    def get_value_said_user(self, slot):
        if slot in self.user_slots:
            val =  self.user_slots[slot].mph()[1]
            if val != "none":
                return val
            else: return None
        else:
            return None

    def get_value_said_system(self, slot):
        if slot in self.system_slots:
            val = self.system_slots[slot].mph()[1]
            if val != "none":
                return val
            else: return None
        else:
            return None

    def _is_stops_in_same_city(self, stopA, stopB):
        stopA_city = self.ontology.get_compatible_vals("stop_city", stopA)
        stopB_city = self.ontology.get_compatible_vals("stop_city", stopB)
        for city in stopA_city:
            if city in stopB_city:
                return True
        if stopA_city and stopB_city:
            return False
        else:
            return True

    def _is_stop_in_city(self, stop, city):
        return stop.lower().startswith(city.lower()) or stop in self.ontology.get_compatible_vals("city_stop", city)

    def is_connection_correct(self):
        return self.connection_info_matches


    def get_compatible_values(self, dai, da):
        slot_name = dai.name

        # prepare known values
        set_values = defaultdict(str)
        if "from_city" in self.user_slots and self.user_slots["from_city"].mph()[1] != "none":
            set_values["from_city"] = self.user_slots["from_city"].mph()[1]
        if "to_city" in self.user_slots and self.user_slots["to_city"].mph()[1] != "none":
            set_values["to_city"] = self.user_slots["to_city"].mph()[1]
        if "from_stop" in self.user_slots and self.user_slots["from_stop"].mph()[1] != "none":
            set_values["from_stop"] = self.user_slots["from_stop"].mph()[1]
        if "to_stop" in self.user_slots and self.user_slots["to_stop"].mph()[1] != "none":
            set_values["to_stop"] = self.user_slots["to_stop"].mph()[1]
        for dai_set in da:
            if (dai_set is not None and dai_set.name is not None and
                ("city" in dai_set.name or "stop" in dai_set.name) and
                (dai_set.value and dai_set.value != "&")):
                set_values[dai_set.name] = dai_set.value

        # get compatible values
        if "city" in slot_name or "stop" in slot_name:

            #resolve "from" or "to" from context
            if slot_name == "city" or slot_name == "stop":
                dai_system = None

                # find first request with correct substring
                for dai in self.last_system_da:
                    if dai.name is not None and slot_name in dai.name and dai.dat == "request":
                        dai_system = dai
                        break

                if dai_system is not None and dai_system.name in self.user_slots:
                    slot_name = dai_system.name

                # if dai_system is None:
                #     for dai in self.last_system_da:
                #         if dai.name is not None and slot_name in dai.name:
                #             dai_system = dai
                #             break

            if "via" in slot_name:
                if ( "from_city" in set_values and "to_city" in set_values and
                   set_values["from_city"] == set_values["to_city"]):
                    return [set_values["from_city"]]
                else:
                    return list(self.ontology['slots'][slot_name])

            elif "in" in slot_name:
                if "to_city" in set_values: # use city you travel to
                    return [set_values["to_city"]]
                else:
                    return list(self.ontology['slots'][slot_name])

            elif "stop" in slot_name:
                #avoid same destination and origin
                exclude = None
                if slot_name == "from_stop" and "to_stop" in set_values:
                    exclude = set_values["to_stop"]
                elif slot_name == "to_stop" and "from_stop" in set_values:
                    exclude = set_values["from_stop"]

                # if we know from which city to go, find only compatible stops:
                if slot_name[:-4] + "city" in set_values:
                    result = list(self.ontology.get_compatible_vals("city_stop", set_values[slot_name[:-4] + "city"]))
                    if exclude and exclude in result:
                        result.remove(exclude)
                    return result
                # if no contraint on city, it can be anything
                else:
                    result = list(self.ontology['slots'][slot_name])
                    if exclude and exclude in result:
                        result.remove(exclude)
                    return result

            elif "city" in slot_name:
                if slot_name[:-4] + "stop" in set_values:
                    return list(self.ontology.get_compatible_vals("stop_city", set_values[slot_name[:-4] + "stop"]))
                else:
                    return list(self.ontology['slots'][slot_name])
            else:
                return None
        else:
            return None

    def get_featurized_hash(self):
        result = defaultdict(str)

       # result["num_turns"] = self.turn_number
        # turn number intervals [0-4, 5-7, 8-12, 13-19, >19]
        if self.turn_number <= 4:
            result["num_turns_0-4"] = "true"
        elif self.turn_number <= 7:
            result["num_turns_5-7"] = "true"
        elif self.turn_number <= 12:
            result["num_turns_8-12"] = "true"
        elif self.turn_number <= 19:
            result["num_turns_13-19"] = "true"
        else:
            result["num_turns_19-"] = "true"

        if self.connection_info_said:
            result["connection_info_said"] = "true"
        if self.connection_error:
            result["connection_info_error"] = "true"
        if self.connection_info_matches:
            result["connection_info_maches"] = "true"

        # add most recent requests
        for dai in self.last_system_da:
            if dai is not None and dai.dat == "request" and dai.name:
                result["lr_"+dai.name] = "recently-requested"

        #add ludait, lsdait
        prob, val = self.ludait.mph()
        result["ludait"] = val
        prob, val = self.lsdait.mph()
        result["lsdait"] = val

        #add slots used by user with its value
        for slot, value in self.user_slots.iteritems():
            if isinstance(value, D3DiscreteValue) and value.mph()[1] == 'none':
                next
            result["u"+slot+"_in"] = "used"
            if slot == "task":
                result["u"+slot] = value.mph()[1]
            elif slot in self.system_slots:
                if self.system_slots[slot].mph()[1] == value.mph()[1]:
                    result[slot] = "same-value"
                else:
                    result[slot] = "diff-value"
            else:
                result[slot] = "user-only"

            if slot in self.ontology['fixed_values']:
                result["u"+slot+"_val"] = value.mph()[1]

        #for slot, value in self.user_slots.iteritems():
        #    result["u"+slot] = "user-value"

        #add slots used by system with respect to user values
        for slot, value in self.system_slots.iteritems():
            if value.mph()[1] == 'none':
                next
            result["s"+slot+"_in"] = "used"
            if slot.startswith("help"):
                if value.mph()[1] is not None:
                    result[slot] = value.mph()[1]
                else:
                    result[slot] = "no-val"
            elif slot not in result:
                result[slot] = "system-only"

            if slot in self.ontology['fixed_values']:
                result["s"+slot+"_val"] = value.mph()[1]
#            if slot in self.user_slots and self.user_slots[slot].mph()[1] == self.system_slots[slot].mph()[1]:
#                result["s"+slot] = "user-value"
#            else:
#                result["s"+slot] = "system-value"

        # slots with prefix "urh_"
        # has only "user-requested" and "system-informed" values.
        for slot, value in self.user_request_history_slots.iteritems():
            if value.mph()[1] == 'none':
                next
            result[slot+"_in"] = "used"
            prob, val = self.user_request_history_slots[slot].mph()
            result[slot] = val

        result.update(self._hash_values(self.user_confirm_history_slots))
        #result.update(self._hash_values(self.user_select_history_slots))

        # slots with prefix "srh_"
        # has only "system-requested" and "user-informed" values.
        for slot, value in self.system_request_history_slots.iteritems():
            if value.mph()[1] == 'none':
                next
            result[slot+"_in"] = "used"
            prob, val = self.system_request_history_slots[slot].mph()
            result[slot] = val

        result.update(self._hash_values(self.system_confirm_history_slots))
        result.update(self._hash_values(self.system_select_history_slots))
        result.update(self._hash_values(self.system_informed_slots))

        return result

