#!/usr/bin/env python
# encoding: utf8
from __future__ import unicode_literals

import codecs


class FileWriter:

    @staticmethod
    def write_file(filename, lines):
        """
        Writes list of lines to utf-8 encoded file.
        :param filename: name of a file
        :param lines: lines to write
        """
        f = codecs.open(filename, "w", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()

    @staticmethod
    def write_file_append(filename, lines):
        """
        Writes list of lines to utf-8 encoded file.
        :param filename: name of a file
        :param lines: lines to write
        """
        f = codecs.open(filename, "a", "utf-8")
        for l in lines:
            f.write(l)
            f.write('\n')
        f.close()