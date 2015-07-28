#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals
import codecs


class FileReader:

    @staticmethod
    def read_file(filename):
        """Read utf-8 encoded file and make list of its lines.

           :param filename: name of a file
           :type filename: str
           :return: list of lines
           :rtype: list(str)
        """
        f = codecs.open(filename, "r", "utf-8")
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
        f.close()
        return lines




