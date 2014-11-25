#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals

import codecs


class FileReader:

    @staticmethod
    def read_file(filename):
        """
        Reads utf-8 encoded file to list of its lines.
        :param filename: name of a file
        :return: list of String lines
        """
        f = codecs.open(filename, "r", "utf-8")
        #f = open(filename, "r")
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
        f.close()
        return lines




