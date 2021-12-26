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

    









def convert(Music_List):
    notes_and_durations = []
    for music in Music_List:
        notes_and_durations.append([])
        for i in range(len(music)):
            if music[i] != ("_"):
                note_count = 1
                for j in range(20):
                    if i+j+1<len(music):
                        if music[i+j+1] == "_":
                            note_count = note_count + 1
                        else:
                            break
                
                
                if music[i] == 'r':
                    notes_and_durations[-1].append([0,note_count])
                else:
                    notes_and_durations[-1].append([float(music[i]),note_count])
        
        
    return notes_and_durations


def Fill_the_End(Real,Fake):
    max_length = 0
    for e in Real:
        if len(e)>max_length:
            max_length = len(e)
    
    for e in Fake:
        if len(e)>max_length:
            max_length = len(e)
    
    for i in range(len(Real)):
        
        fillcount = max_length - len(Real[i])
        Real[i] = Real[i]  + [[0.0,2.0]]*fillcount
    
    for i in range(len(Fake)):
        fillcount = max_length - len(Fake[i])
        Fake[i] = Fake[i]  + [[0.0,2.0]]*fillcount

        
        
    return Real,Fake,max_length




open_file = open("1000 Fake Musics LSTM", "rb")
Fake_Musics = pickle.load(open_file)
open_file.close()

with open('final.txt') as f:
    lines = f.readline()

Real_Musics = lines.split( 64*" /"+" ")
for i in range(len(Real_Musics)):
    Real_Musics[i] = Real_Musics[i].split(" ")[:-1]
Real_Musics.pop(-1)#last element is empty

# Convert the data from String format to nx2 matrix where 1st column is note, 2nd is duration
#of the note
Real_Musics = convert(Real_Musics)
Fake_Musics = convert(Fake_Musics)
print(Real_Musics[1])

#Fill the end with ['r','2']
Real_Musics,Fake_Musics, Max_Length = Fill_the_End(Real_Musics,Fake_Musics)



# Converting data into 1 row [notes,durations]

for i in range(len(Real_Musics)):
    a = []
    b = []
    for j in range(221):
        a = a + [Real_Musics[i][j][0]]
        b = b + [Real_Musics[i][j][1]]

    Real_Musics[i] = a + b


for i in range(len(Fake_Musics)):
    a = []
    b = []
    for j in range(221):
        a = a + [Fake_Musics[i][j][0]]
        b = b + [Fake_Musics[i][j][1]]

    Fake_Musics[i] = a + b
    


    
#1000 fake musics, 1683 real musics

Train_Label_Set = [[1.0, 0.0]]*(1683-200) + [[0.0, 1.0]]*(1000-200)
Train_Set = Real_Musics[0:1683-200] + Fake_Musics[0:1000-200]
Test_Label_Set = [[1.0,0.0]]*(200) + [[0.0,1.0]]*(200)
Test_Set = Real_Musics[1683-200:] + Fake_Musics[1000-200:]

open_file = open("Train_Label_Set", "wb")
pickle.dump(Train_Label_Set, open_file)
open_file.close()
    
open_file = open("Train_Set", "wb")
pickle.dump(Train_Set, open_file)
open_file.close()

open_file = open("Test_Label_Set", "wb")
pickle.dump(Test_Label_Set, open_file)
open_file.close()

open_file = open("Test_Set", "wb")
pickle.dump(Test_Set, open_file)
open_file.close()



# open_file = open("Test", "wb")

# pickle.dump(Test, open_file)

# open_file.close()








