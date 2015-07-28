#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals
import codecs


class FileWriter:

    @staticmethod
    def write_file(filename, lines):
        """Write utf-8 encoded file from list of its lines.

           :param filename: name of a file
           :type filename: str
           :param lines: list of lines
           :type lines: list(str)
        """
        f = codecs.open(filename, "w", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()

    @staticmethod
    def write_file_append(filename, lines):
        """Append to utf-8 encoded file from list of its lines.

           :param filename: name of a file
           :type filename: str
           :param lines: list of lines
           :type lines: list(str)
        """
        f = codecs.open(filename, "a", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()