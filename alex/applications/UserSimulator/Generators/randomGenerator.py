from __future__ import unicode_literals
import random

class RandomGenerator:

    def __init__(self):
        random.seed(19910604)
        pass

    @staticmethod
    def generate_random_response(responses, counts, total):
        """
        Choose one of the responses according to given probability distribution.
        Generation is the most efficient if responses are in descending order sorted
        by its counts (probability).
        :param responses: list of generated objects
        :param counts: list of counts that represent response probabilities
        :param total: sum of counts
        :return: random response from list of responses
        """
        # random number [0,total)
        r = random.randrange(0, total)

        # list of running sums
        run_sums = list(RandomGenerator._running_sum(counts))
        # choose response at the position where subtract is first positive number
        chosen = next((a for a, s in zip(responses, run_sums) if (s-r) > 0), None)

        return chosen

    @staticmethod
    def _running_sum(a):
        """
        Count running sum of a list
        :param a: list of numeric type
        :return:
        """
        tot = 0
        for item in a:
            tot += item
            yield tot
