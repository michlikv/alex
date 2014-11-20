#!/usr/bin/env python

from randomGenerator import RandomGenerator

gen = RandomGenerator()
probs = [80, 10, 5, 2, 1]
tags = ['a', 'b', 'c', 'd', 'e']

rand = [gen.generate_random_response(tags, probs, 100) for t in range(0, 1000)]
res = {t:(rand.count(t)/1000.0) for t in tags}
print res
