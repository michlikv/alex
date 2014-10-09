#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import autopath

import alex.utils.audio as audio
import alex.utils.audio_play as audio_play

from alex.components.tts.google import GoogleTTS
from alex.utils.config import Config

if __name__ == '__main__':
    print "Testing Google TTS service"
    print "=" * 120
    print

    text = 'Hello. Thank you for calling.'
    language = 'en'

    print "Synthesize text:", text
    print "Language:       ", language
    print

    c = {
        'TTS': {
            'Google': {
                'debug': False,
                'language': language
            }
        }
    }
    cfg = Config.load_configs(log=False)
    cfg.update(c)

    tts = GoogleTTS(cfg)

    print 'calling TTS'
    wav = tts.synthesize(text)

    print 'saving the TTS audio in ./tmp/google_tts.wav'
    audio.save_wav(cfg, './tmp/google_tts.wav', wav)

    print 'playing audio'
    audio_play.play(cfg, wav)
