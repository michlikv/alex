#!/usr/bin/env python
# encoding: utf8

from __future__ import unicode_literals


class Preprocessing:

    user_del = "user:"
    system_del = "system:"
    end_of_dialogue = 'hangup()'

    @staticmethod
    def filter_acts_one_only(acts_list):
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

        if len(stack) >= 1:
            dialogue.append(Preprocessing._create_act_from_stack(stack))

        dialogue.append(Preprocessing.end_of_dialogue)
        return dialogue

    @staticmethod
    def _create_act_from_stack(stack):
        if len(stack) >= 1:
            return(stack[-1])
        else:
            raise Exception("Incorrect input format")

