#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals
from alex.components.asr.utterance import Utterance
from alex.components.slu.da import DialogueAct
from compiler.ast import flatten
import re


class Preprocessing:

    user_del = "user:"
    system_del = "system:"
    end_of_dialogue = 'hangup()'

    @staticmethod
    def filter_acts_one_only(acts_list):
        """
        Create a conversation between system and user. System acts are at even positions,
        User acts at odd positions.
        Makes each dialogue a sequence of alternating user and system turns.
        Dialogue acts must be annotated with "user:" or "system:" turn indicator prefix.
        :param acts_list: list of actions
        :return: filtered list where user and system alternate
        """
        dialogue = []
        stack = []
        user_turn = False

        id_fist_system = next((i for x, i in zip(acts_list, range(0,len(acts_list)))
                                if (x.startswith(Preprocessing.system_del,0))),None)
        if id_fist_system == None:
            return []

        for line in acts_list[id_fist_system:]:
            user_match = line.startswith(Preprocessing.user_del,0)
            system_match = line.startswith(Preprocessing.system_del,0)

            #print "/", line, "/"
            if user_match:
                line = line[6:]
            elif system_match:
                line = line[8:]

            # if input is well formatted
            if user_match or system_match:
                if ((user_match and user_turn) or
                   (system_match and not user_turn)):
                    stack.append(line)
                elif ((system_match and user_turn) or
                    (user_match and not user_turn)):
                    dialogue.append(Preprocessing._create_act_from_stack(stack))
                    stack = [line]
                    user_turn = not user_turn
            else:
                raise Exception("Incorrect input format on line:|"+line+"|")

        if len(stack) >= 1 and user_turn:
            dialogue.append(Preprocessing._create_act_from_stack(stack)+"&"+Preprocessing.end_of_dialogue)
        elif len(stack) >= 1 and not user_turn:
            dialogue.append(Preprocessing._create_act_from_stack(stack))
            dialogue.append(Preprocessing.end_of_dialogue)
        else:
            #todo this should never happen
            dialogue.append(Preprocessing.end_of_dialogue)

        return dialogue

    @staticmethod
    def _create_act_from_stack(stack):
        """
        Implements method that constructs dialogue act from more consecutive user or system acts.
        :param stack: Consecutive Dialogue acts
        :return:
        """
        if len(stack) >= 1:
            return(stack[-1])
        else:
            raise Exception("Incorrect input format")

    @staticmethod
    def convert_string_to_dialogue_acts(self, text, slu):
        """
        Uses SLU module to get Dialogue act(s) from text utterance.

        :param text: Text to be transformed
        :param slu: SLU object
        :return: Dialogue act
        """
        utt = text.strip()
        try:
            if utt == "":
                utt = "_silence_"
            utt = Utterance(utt)
        except Exception:
            raise Exception("Invalid utterance: %s" % utt)
        das = self.slu.parse(utt)
        return das

    @staticmethod
    def remove_slot_values_from_dialogue(dialogue):
        return [Preprocessing.remove_slot_values(x) for x in dialogue]

    @staticmethod
    def remove_slot_values(da):
        return re.sub('"[^"]*"', '""', da)

    @staticmethod
    def get_slot_names_plus_values_from_dialogue(dialogue):
        slot_pairs = [Preprocessing.get_slot_names_plus_values(x) for x in dialogue]
        #todo see what it does, then flatten the list
        # TODO what IF there IS NO VALUE FOR THE SLOT
        return flatten(slot_pairs)

    @staticmethod
    def get_slot_names_plus_values(da):
        da_obj = DialogueAct(da)
        slot_pairs = da_obj.get_slots_and_values()
        return slot_pairs


    @staticmethod
    def shorten_connection_info(da):
        #todo not implemented yet :-O
        pass
