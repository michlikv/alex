#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals
import autopath
from alex.components.asr.utterance import Utterance
from alex.components.slu.da import DialogueAct, DialogueActItem
from compiler.ast import flatten


class Preprocessing:
    user_del = "user:"
    system_del = "system:"
    end_of_dialogue = 'hangup'
    connection_info_da = DialogueActItem('connectionInfo')

    @staticmethod
    def prepare_conversations(acts_list, user_method, system_method):
        """
        Create a conversation between system and user. System acts are at even positions,
        User acts are at odd positions.
        Makes each dialogue a sequence of alternating user and system turns.
        Dialogue acts must be annotated with "user:" or "system:" turn indicator prefix.
        This method does not add "end of the dialogue" to the conversation.
        :param acts_list: list of actions
        :return: filtered list where user and system alternate
        """
        dialogue = []
        stack = []
        user_turn = False

        id_fist_system = next((i for x, i in zip(acts_list, range(0, len(acts_list)))
                               if (x.startswith(Preprocessing.system_del, 0))), None)
        if id_fist_system == None:
            return []

        for line in acts_list[id_fist_system:]:
            user_match = line.startswith(Preprocessing.user_del, 0)
            system_match = line.startswith(Preprocessing.system_del, 0)

            # print "/", line, "/"
            if user_match:
                line = line[6:]
            elif system_match:
                line = line[8:]

            # if input is well formatted
            if user_match or system_match:
                # build stack
                if ((user_match and user_turn) or
                        (system_match and not user_turn)):
                    stack.append(line)
                # process stack and start building a new one
                elif ((system_match and user_turn) or
                          (user_match and not user_turn)):
                    if (user_turn):
                        dialogue.append(user_method(stack))
                    else:
                        dialogue.append(system_method(stack))
                    stack = [line]
                    user_turn = not user_turn
            else:
                raise Exception("Incorrect input format on line:|" + line + "|")

        if len(stack) >= 1 and user_turn:
            dialogue.append(user_method(stack))
        elif len(stack) >= 1 and not user_turn:
            dialogue.append(system_method(stack))

        return dialogue

    @staticmethod
    def create_act_from_stack_use_last(stack):
        """
        Constructs dialogue act from more consecutive user or system actions.
        It takes only the last act.
        :param stack: Consecutive Dialogue acts
        :return:
        """
        if len(stack) >= 1:
            return (stack[-1])
        else:
            raise Exception("Incorrect input format")

    # @staticmethod
    # def create_act_from_stack_concat_da(stack):
    # """
    #     Constructs dialogue act from more consecutive user or system acts.
    #     Concatenates unique not empty acts from the stack of dialogue acts.
    #
    #     :param stack: Consecutive Dialogue acts
    #     :return:
    #     """
    #
    #     # NOT IMPLEMENTED

    @staticmethod
    def create_act_from_stack_concat_text(stack):
        """
        Constructs dialogue act from more consecutive user or system acts.
        Concatenates not empty text from the stack.

        :param stack: Consecutive transcribed text
        :return:
        """

        if len(stack) >= 1:
            # take non empty strings
            stack = [a for a in stack if a]
            if len(stack) >= 1:
                return ' '.join(stack)
            else:
                return ""
        else:
            raise Exception("Incorrect input format")

    @staticmethod
    def convert_string_to_dialogue_acts(text, slu):
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
        das = slu.parse(utt)
        return das

    @staticmethod
    def remove_slot_values_from_dialogue(dialogue):
        return [Preprocessing.remove_slot_values(x) for x in dialogue]

    @staticmethod
    def remove_slot_values(da, exclude=None):
        for dai in da.dais:
            if dai.value:
                if exclude is None or dai.name not in exclude:
                    dai.value = '&'

    @staticmethod
    def get_slot_names_plus_values_from_dialogue(dialogue, ignore_slots=[], ignore_values=[]):
        dialogue
        slot_pairs = [Preprocessing.get_slot_names_plus_values(x, ignore_slots, ignore_values) for x in dialogue]
        return flatten(slot_pairs)

    @staticmethod
    def get_slot_names_plus_values(da, ignore_slots=[], ignore_values=[]):
        slot_pairs = da.get_slots_and_values()
        slot_pairs = [[s, v] for s, v in slot_pairs if s not in ignore_slots and v not in ignore_values]
        return slot_pairs


    @staticmethod
    def shorten_connection_info(da):
        new_da = DialogueAct()

        for dai in da.dais:
            if dai.dat == 'apology':
                return da
            elif (dai.dat == 'inform' and
                          dai.name == 'vehicle'):
                new_da.append(Preprocessing.connection_info_da)
                return new_da
            else:
                new_da.append(dai)
        return new_da

    @staticmethod
    def clear_numerics(dialogue):
        for i, a in enumerate(dialogue):
            if a.startswith("1.000 "):
                dialogue[i] = a[6:]


    @staticmethod
    def add_end_da(dialogue):
        if len(dialogue) % 2 == 0:
            dialogue[-1].append(DialogueActItem(Preprocessing.end_of_dialogue))
            return dialogue
        else:
            return dialogue.append(DialogueAct(Preprocessing.end_of_dialogue + '()'))

    @staticmethod
    def add_end_string(dialogue):
        if len(dialogue) % 2 == 0:
            dialogue[-1] = dialogue[-1] + '&' + Preprocessing.end_of_dialogue + '()'
            return dialogue
        else:
            return dialogue.append(Preprocessing.end_of_dialogue + '()')