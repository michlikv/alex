#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals

import codecs

class FileReader:

    @staticmethod
    def read_file(filename):
        f = codecs.open(filename, "r", "utf-8")
        #f = open(filename, "r")
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
        f.close()
        return lines




