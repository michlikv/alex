from __future__ import unicode_literals
import random

class RandomGenerator:

    def __init__(self):
        random.seed(19910604)

    # choose one of the responses by given probability distribution
    # responses are assumed to be sorted in #todo sestupne
    @staticmethod
    def generate_random_response(responses, counts, total):
        # random number [0,total]
        r = random.randrange(0, total)

        # list of running sums)
        run_sums = list(RandomGenerator._running_sum(counts))
        # choose response at the position where subtract is first positive number
        chosen = next((a for a, s in zip(responses, run_sums) if (s-r) > 0), None)

        return chosen

    @staticmethod
    def _running_sum(a):
        tot = 0
        for item in a:
            tot += item
            yield tot
