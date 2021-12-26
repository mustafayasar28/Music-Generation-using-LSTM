# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 21:52:24 2021

@author: muhittin can
"""

import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
import pickle
def convert_music(in_music, step_dur=0.25, in_format="mid", file_name="output_music.midi"):
    """
    :param in_music: Music string we have created by generate_music function
    :param step_dur: Step duration of the music
    :param in_format: File format of the output music file
    :param file_name: Name of the output music file
    """

    m21_stream = m21.stream.Stream()

    """
    We have the notes in numbered formats we have been using such as 74 _ _ _ . We need to convert the numbers
    to the notes, and the underscores to the duration of the notes.
    """
    step_ctr = 1
    start_sym = None

    for i, sym in enumerate(in_music):
        if sym != "_" or i + 1 == len(in_music):
            if start_sym is not None:
                quarter_dur = step_dur * step_ctr

                if start_sym == "r":
                    e = m21.note.Rest(quarterLength=quarter_dur)
                else:
                    e = m21.note.Note(int(start_sym), quarterLength=quarter_dur)

                m21_stream.append(e)
                step_ctr = 1
            start_sym = sym
        else:
            step_ctr += 1

    m21_stream.write(in_format, file_name)

    







with open('final.txt') as f:
    lines = f.readline()

music_list = lines.split( 64*" /"+" ")
music_list_for_midi = music_list.copy()
for i in range(len(music_list)):
    music_list[i] = music_list[i].split(" ")[:-1]
    music_list_for_midi[i] = music_list_for_midi[i][:-1]
music_list.pop(-1)#last element is empty



            
          
            
Real_Musics = music_list[1000:1050]

for i in range(len(Real_Musics)):
    Real_Musics[i] = Real_Musics[i] + ["Real"]


open_file = open("1000 Fake Musics", "rb")
Fake_Musics = pickle.load(open_file)
open_file.close()


Fake_Musics = Fake_Musics[450:500]
for i in range(len(Fake_Musics)):
    Fake_Musics[i] = Fake_Musics[i] + ["Fake"]



Test = Real_Musics + Fake_Musics[0:50]
Test = np.array(Test)
np.random.shuffle(Test)




for i in range(Test.shape[0]):
     convert_music(Test[i][:-1],file_name="Test Musics LSTM\Test "+str(i+1)+".midi")


open_file = open("Test_Data_LSTM", "wb")

pickle.dump(Test, open_file)

open_file.close()