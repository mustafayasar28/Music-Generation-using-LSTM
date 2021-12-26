# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:06:43 2021

@author: muhittin can
"""

import pickle
import numpy as np
open_file = open("Test_Data_LSTM", "rb")
Tests = pickle.load(open_file)
open_file.close()


print("In our test, you will try to identify if a song is real or computer generated. Listen to each song provided and answer:")
print("1 if you think the song is real")
print("0 if you think the song is computer generated")

Real_accuracy = 0
Fake_accuracy = 0

for i in range(0,100):
    
     answer = input("Answer for song number "+str(i+1)+" : ")
    
     if answer == "1" and Tests[i][-1] == "Real":
        Real_accuracy = Real_accuracy + 1
    
     if answer == "0" and Tests[i][-1] == "Fake":
            Fake_accuracy = Fake_accuracy + 1   
            

print("True answers for Real:",Real_accuracy)
print("True answers for Computer_generated:",Fake_accuracy)

