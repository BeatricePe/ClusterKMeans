# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:48:20 2018

@author: Bea
"""

import pandas as pd
import numpy as np; np.random.seed(0)
import seaborn.apionly as sns
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans


def ReadingAndColumnSelector (DataIn, sepFile):
    Data = pd.read_csv (DataIn, sep =sepFile)
    return Data
''' 
Elimination of first section of beh
'''
def DFcorrection (Data, columnLabel, beh):
    lista = []
    i = 0
    while True:
        if Data[columnLabel][len(Data)-i-1] == beh:
            i = i +1
            lista.append(i)
        else:
            break
    Data2 = Data.drop(Data.index[len(Data)-len(lista):len(Data)])
#        Data2.index = range(len(Data2))
    return (Data2)


'''
funzione che stabilisce il numero di frames da prendere: il numero di frames da prendere deve essere tale che sia a dx che a 
sx non abbia transizioni di comportamento. Dato un comportamento per ogni blocco in cui questo comportamento si manifesta
dovrò rispettare la condizione definita sopra e quindi tra tutti andrò a prendere il minimo
'''

def fun2(serieBeh, action):
                
  
    minimum = len (serieBeh)
    indexList = []
    threshold = 20
    
    for i in range (len (serieBeh)):    
        if serieBeh[i] == action and serieBeh[i+1] != action:
            after = 1
            action2 = serieBeh[i+1] 
            while True:
                if i+after == len (serieBeh):
                    break
                elif serieBeh[i + after] != action2:
                    break
                after +=1
            if after < threshold:
                continue
            before = 0   
            while True:
                if i-1-before == 0:
                    break
                elif serieBeh[i -1 - before] != serieBeh[i]:
                    break
                before +=1
            if before < threshold:
                continue
            indexList.append(i)
            if after < before:
                before = after
            if before < minimum :
                minimum = before
    return (minimum, indexList)
                
            
            
            
    


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

File = ReadingAndColumnSelector (r"C:\Users\Bea\Desktop\lab\Programmi\Vmh4M20dist.csv", ';')
File2 = DFcorrection (File, 'Beh', 'Home')
Beh = File2.loc [:, 'Beh']
tupla = fun2(Beh,'Home')
Data = fun1 (File2, tupla[1], tupla[0])
Mean = Data.loc [:, 'Mean(1)':'Mean(39)']
t = fun3 (Mean, tupla[1], tupla[0])
a = tupla[0]
t.index = range(-a,a+1)
d = t.transpose()
d = d.drop ('Mean(5)',0)
d = d.drop ('Mean(6)',0)
d = d.drop ('Mean(7)',0)
d = d.drop ('Mean(9)',0)
d = d.drop ('Mean(17)',0)
d = d.drop ('Mean(18)',0)
d = d.drop ('Mean(25)',0)
d = d.drop ('Mean(26)',0)
d = d.drop ('Mean(30)',0)
d = d.drop ('Mean(34)',0)
d = d.drop ('Mean(35)',0)
d = d.drop ('Mean(23)',0)
d = d.drop ('Mean(1)',0)
'''
Kmeans and HM
'''
km = KMeans(n_clusters=6, init='k-means++', n_init=100)
km.fit(d)
x = km.fit_predict(d)
d['Cluster'] = x
d = d.sort_values(by=['Cluster'])



t =d.drop ('Cluster',1)






fig, ax = plt.subplots(figsize=(25,10)) 


ax.vlines([a+0.5],0,1, transform=ax.get_xaxis_transform(), colors='k')   
#ax.hlines([3],1,0, transform=ax.get_yaxis_transform(), colors='k')   
#ax.hlines([10],1,0, transform=ax.get_yaxis_transform(), colors='k')   
#ax.hlines([12],1,0, transform=ax.get_yaxis_transform(), colors='k')   
#ax.hlines([21],1,0, transform=ax.get_yaxis_transform(), colors='k')   
#ax.hlines([25],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([22],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([15],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([20],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([21],1,0, transform=ax.get_yaxis_transform(), colors='k')   
##ax.hlines([24],1,0, transform=ax.get_yaxis_transform(), colors='k')   

sns.heatmap(t,annot = False, xticklabels=1,yticklabels=1, cmap="YlGnBu")
               