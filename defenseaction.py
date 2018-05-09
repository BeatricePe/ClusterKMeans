# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:50:28 2018

@author: Bea
"""

import pandas as pd
import numpy as np; np.random.seed(0)
import seaborn.apionly as sns
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


def ReadingAndColumnSelector (DataIn, sepFile):
    Data = pd.read_csv (DataIn, sep =sepFile)
    return Data
''' 
Elimination of first section of beh
'''
def DFcorrection (Data, columnLabel, beh):
    lista = []
    for i in range (len( Data)):
        if Data[columnLabel][0] == beh: 
            if Data [columnLabel][0] == Data [columnLabel][i]:
                lista.append (i)
            elif Data [columnLabel][0] != Data [columnLabel][i]:
                break
    Data2 = Data.drop(Data.index[0:len(lista)])
    Data2.index = range(len(Data2))
    return (Data2)


'''
funzione che stabilisce il numero di frames da prendere: il numero di frames da prendere deve essere tale che sia a dx che a 
sx non abbia transizioni di comportamento. Dato un comportamento per ogni blocco in cui questo comportamento si manifesta
dovrò rispettare la condizione definita sopra e quindi tra tutti andrò a prendere il minimo
'''
def fun2(serieBeh, action):
    
    minimum = len (serieBeh)
    indexList = []
   
    for i in range (len (serieBeh)):
        if serieBeh[i] == action and serieBeh[i-1] != action:
            for u in range (-12,0):
                serieBeh[i+u] = action
            g = i-12
            indexList.append(g)
                
            for j in range (0,minimum):
                if serieBeh[g+ j] != action:
                    minimum = j
                    break

            for k in range (0,minimum):
                if serieBeh [g-1-k] != serieBeh[g-1]:
                    minimum = k+1    
                    break 
            break 
               
                
           
                
    return (int (minimum), indexList)


'''
funzione che data la lista degli indici trovata con fun2 e il numero di comportamenti da prendere trovato con fun2, della matrice iniziale prende solo gli 
gli indici compresi  -n:i:n
'''
def fun1 (Data, BehIndexes, n):
    DataOut = pd.DataFrame ()
    for index in BehIndexes:
        DataOut = DataOut.append(Data.ix [index-n:index+n])
    return DataOut  

    return DataOut
'''
funzione che crea una matrice di zeri che deve essere composta dal numero dei neuroni, e che deve essere lunga
 quanto il numero di comportamenti da prendere*2 +1, e che sostituisce all'intera riga di zeri (cioè per ogni neurone),
la somma delle attività relative a quell'indice (alla linea zero composta da zeri va a sostituire la somma delle attività ottenute 
per tutte le volte che incontra il comportamento di interesse a -n, alla linea uno la stessa cosa, ma somma le attività 
realtive a n-1 ecc)
'''
def fun3 (Means, firstBehIndexes, n):
    feature_list = []
    for column in Means:
        Means[column] = Means[column].astype('float')
        Means [column] = Means[column]*100
        feature_list.append (column)
    DataEmpty = pd.DataFrame(0, index=np.arange(n*2+1), columns=feature_list)
    meanIndex = -n 
    for i,row in DataEmpty.iterrows():    
        for index in firstBehIndexes:
            DataEmpty.loc[i] = DataEmpty.loc[i] + Means.loc[index + meanIndex]
        meanIndex+=1
    DataEmpty.loc[i] = DataEmpty.loc[i]/len(firstBehIndexes)
    return DataEmpty

File = ReadingAndColumnSelector (r'C:\Users\Bea\Desktop\lab\Programmi\Vmh4SF21beh (7).csv', ';')
File2 = DFcorrection (File, 'Beh', 'defense action')
Beh = File2.loc [:, 'Beh']
tupla = fun2(Beh,'defense action')
Data = fun1 (File2, tupla[1], tupla[0])
Mean = Data.loc [:, 'Mean(1)':'Mean(39)']
t = fun3 (Mean, tupla[1], tupla[0])
a = tupla[0]
t.index = range(-a,a+1)
d = t.transpose()


'''
Kmeans and HM
'''
km = KMeans(n_clusters=6, init='k-means++', n_init=100)
km.fit(d)
x = km.fit_predict(d)
d['Cluster'] = x
d = d.sort_values(by=['Cluster'])



t =d.drop ('Cluster',1)






fig, ax = plt.subplots(figsize=(15,10)) 


ax.vlines([a+0.5],0,1, transform=ax.get_xaxis_transform(), colors='k')   
ax.hlines([9],1,0, transform=ax.get_yaxis_transform(), colors='k')   
ax.hlines([23],1,0, transform=ax.get_yaxis_transform(), colors='k')   
ax.hlines([31],1,0, transform=ax.get_yaxis_transform(), colors='k')   
ax.hlines([32],1,0, transform=ax.get_yaxis_transform(), colors='k')   
ax.hlines([33],1,0, transform=ax.get_yaxis_transform(), colors='k')   
#ax.hlines([39],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([15],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([20],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([21],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([24],1,0, transform=ax.get_yaxis_transform(), colors='k')   

sns.heatmap(t,annot = False, xticklabels=1,yticklabels=1, cmap="YlGnBu")
               
