# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:32:39 2021

@author: Kirito
"""


import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pickle

def acc(Y,Y_pred):

    TP = np.sum((Y_pred + Y == 2)*1)
    TN = np.sum((Y_pred + Y == 0)*1)
    FP = np.sum((Y_pred - Y == 1)*1)
    FN = np.sum((Y_pred - Y == -1)*1)
    
    acc = (TP+TN)/(TP+TN+FP+FN)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    
    print("tp " + str(TP))
    print("tn " + str(TN))
    print("fp " + str(FP))
    print("fn " + str(FN))
    
    print("acc " + str(acc))
    print("P " + str(P))
    print("R " + str(R))
    
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
                notes_and_durations[-1].append([music[i],note_count])
        
    return notes_and_durations
def clear_silence(data):
    for music in data:
        for notes in music:
            if(notes[0] == 'r'):
                music.remove(notes)
    return data

#pitch related
def PV(data):
    
    pv = []
    
    for music in data:
        m = []
        for notes in music:
            m.append(notes[0])
        pv.append(len(set(m)))
    return pv

def PH(data):
    
    ph = []
    
    for music in data:
        notes = 12*[0]
        for note in music:
                notes[int(note[0])%12] += 1
        ph.append(notes)
        
    return ph


# def PCTM():
    
def PR(data):
    
    pr = []
    for music in data:
        temp_min = 100
        temp_max = 0
        for note in music:
            curr_note = int(note[0])

            if(curr_note > temp_max):
                temp_max = curr_note
                
            if(curr_note < temp_min):
                temp_min = curr_note
                
        pr.append(temp_max-temp_min)
    return pr
    

def PI(data):
    pi = []
    
    for music in data:
            pi_s = 0
            num = 0
            for i in range(len(music) - 1):
                interval = int(music[i][0]) - int(music[i+1][0])
                pi_s = pi_s + abs(interval)
                if(interval != 0):
                    num = num + 1
            pi.append(pi_s/num)
    return pi

#note related
def NC(data):
    
    nc = []
    
    for music in data:
        nc.append(len(music))
        
    return nc

def IOI(data):
    
    ioi = []
    
    for music in data:
        total_dur = 0
        music_l = len(music)
        for notes in music:
            total_dur = total_dur + int(notes[1])
        dur = total_dur/music_l
        ioi.append(dur)
        
    return ioi

def NLH(data):
    
    nh = []
    
    for music in data:
        notes = 4*[0]
        for note in music:
            notes[int(note[0])%4] += 1
        nh.append(notes)
        
    return nh
    
# def NLTM():

with open('final.txt') as f:
    lines = f.readline()

music_list = lines.split( 64*" /"+" ")

for i in range(len(music_list)):
    music_list[i] = music_list[i].split(" ")[:-1]
    
music_list.pop(-1)#last element is empty


data = []
for music in music_list:
    data.append([])
    for i in range(len(music)):
        if music[i] != ("_"):
            note_count = 1
            for j in range(10):
                if i+j+1<len(music):
                    if music[i+j+1] == "_":
                        note_count = note_count + 1
                    else:
                        break
            data[-1].append([music[i],note_count])
            


open_file = open("generated", "rb")
loaded_list = pickle.load(open_file)
open_file.close()

generated = []
for music in loaded_list:
    generated.append([])
    for i in range(len(music)):
        if music[i] != ("_"):
            note_count = 1
            for j in range(10):
                if i+j+1<len(music):
                    if music[i+j+1] == "_":
                        note_count = note_count + 1
                    else:
                        break
            generated[-1].append([music[i],note_count])
            
generated = clear_silence(generated)
generated = clear_silence(generated)

data = clear_silence(data)
data = clear_silence(data)

pv = np.array(PV(generated))
ph = np.array(PH(generated))
pi = np.array(PI(generated))
pr = np.array(PR(generated))
nc = np.array(NC(generated))
ioi = np.array(IOI(generated))
nlh = np.array(NLH(generated))

pv = pv.reshape(1000,1)
pr = pr.reshape(1000,1)
pi = pi.reshape(1000,1)
nc = nc.reshape(1000,1)
ioi = ioi.reshape(1000,1)
labels = np.zeros((1000,1))

features = (ioi,pv,pr,labels)
X_gen = np.concatenate(features, axis = 1)

gen_tr = X_gen[:900,:]
gen_t = X_gen[900:,:]

pv = np.array(PV(data))
ph = np.array(PH(data))
pi = np.array(PI(data))
pr = np.array(PR(data))
nc = np.array(NC(data))
ioi = np.array(IOI(data))
nlh = np.array(NLH(data))

pv = pv.reshape(1682,1)
pr = pr.reshape(1682,1)
pi = pi.reshape(1682,1)
nc = nc.reshape(1682,1)
ioi = ioi.reshape(1682,1)
labels = np.ones((1682,1))

features = (ioi,pv,pr,labels)
X_data = np.concatenate(features, axis = 1)

real_tr = X_data[:900,:]
real_t = X_data[900:1000,:]

Train_set = np.concatenate((real_tr,gen_tr), axis = 0)
Test_set = np.concatenate((real_t,gen_t), axis = 0)

np.random.shuffle(Train_set)
np.random.shuffle(Test_set)

y_train = Train_set[:,-1]
X_train = Train_set[:,:-1]

y_test = Test_set[:,-1]
X_test = Test_set[:,:-1]


#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

acc(y_test,y_pred)


tr = Train_set[:,0:2]

for i in range(tr.shape[0]):
    if(y_train[i] == 1):
        plt.scatter(tr[i,0],tr[i,1], c = 'r')
    else:
        plt.scatter(tr[i,0],tr[i,1], c = 'b')

plt.show()

pca = PCA(n_components=2)
normalized = preprocessing.normalize(X_train)
normalized = X_train - np.mean(X_train)
pca.fit(normalized)
X = pca.transform(X_train)

for i in range(X.shape[0]):
    if(y_train[i] == 1):
        plt.scatter(X[i,0],X[i,1], c = 'r')
    else:
        plt.scatter(X[i,0],X[i,1], c = 'b')

plt.show()


kmeans = KMeans(n_clusters=2).fit(X)
acc(y_train,kmeans.labels_)

for i in range(X.shape[0]):
    if(kmeans.labels_[i] == 1):
        plt.scatter(X[i,0],X[i,1], c = 'b')
    else:
        plt.scatter(X[i,0],X[i,1], c = 'r')

plt.show()

#pairwise crossvalidation



mean1 = [0.5733642,  1.14317128, 3.91470311]
distance = 0
var = 0
X_data = X_data[:,:-1]

for i in range(X_data.shape[0]):
    arr = X_data[i]
    for j in range(i,X_data.shape[0]):
        distance = distance + (arr-X_data[j])**2
        var = var + ((arr-X_data[j])**2 - mean1)**2
dim1 = X_data.shape[0]

print(distance/dim1/(dim1-1)/2)
var1 = (var**0.5/dim1/(dim1-1)/2)
std1 = var1**0.5

distance = 0
var = 0
X_gen = X_gen[:,:-1]
mean2 = [ 0.85561054,  2.16161712, 17.7199995 ]

for i in range(X_gen.shape[0]):
    arr = X_gen[i]
    for j in range(i,X_gen.shape[0]):
        distance = distance + (arr-X_gen[j])**2
        var = var + ((arr-X_gen[j])**2 - mean2)**2
        
dim2 = X_gen.shape[0]

print(distance/dim2/(dim2-1)/2)
var2 = var**0.5/dim2/(dim2-1)/2
std2 = var2**0.5

distance = 0
var = 0
mean3 = [ 1.29138885,  2.26189358, 14.48795006]

for i in range(X_data.shape[0]):
    arr = X_data[i]
    for j in range(i,X_gen.shape[0]):
        distance = distance + (arr-X_gen[j])**2
        var = var + ((arr-X_gen[j])**2 - mean3)**2

print(distance/dim1/dim2)
var3 = var/dim1/dim2
std3 = var3**0.5

std1 = np.array(std1)
std2 = np.array(std2)
std3 = np.array(std3)

mean1 = np.array(mean1)
mean2 = np.array(mean2)
mean3 = np.array(mean3)

KLD1 = np.log(std3/std1) + (std1**2+(mean1 - mean3)**2)/(2*std3**2) -0.5
KLD2 = np.log(std3/std2) + (std2**2+(mean2 - mean3)**2)/(2*std3**2) -0.5
KLD3 = np.log(std2/std1) + (std1**2+(mean1 - mean2)**2)/(2*std2**2) -0.5

print(KLD1)
print(KLD2)
print(KLD3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    