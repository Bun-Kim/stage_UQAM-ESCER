#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:03:28 2022

@author: san
"""
import xarray as xr
import numpy as np

def data_saison(dataset,champs,saison,dates,lon,lat):
    mois=[]
    for i in range(1,13):
            mois.append(np.any([dates.month==i],axis=0))
   
    data_mensuel=[dataset[champs].values[mois[i]] for i in range(12)]
    if saison == 'DJF':
        
        xarrayseason=xr.DataArray(data_mensuel[11].mean(axis=0) + data_mensuel[0].mean(axis=0) + 
                              data_mensuel[0].mean(axis=0))
        season_dataset=xr.DataArray.to_dataset(xarrayseason,name='DJF_' + champs)
        season_dataset=season_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    if saison == 'MAM':
        xarrayseason=xr.DataArray(data_mensuel[11].mean(axis=0) + data_mensuel[0].mean(axis=0) + 
                              data_mensuel[0].mean(axis=0))
        season_dataset=xr.DataArray.to_dataset(xarrayseason,name='MAM_' + champs)
        season_dataset=season_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    if saison == 'JJA':
        xarrayseason=xr.DataArray(data_mensuel[11].mean(axis=0) + data_mensuel[0].mean(axis=0) + 
                              data_mensuel[0].mean(axis=0))
        season_dataset=xr.DataArray.to_dataset(xarrayseason,name='JJA_' + champs)
        season_dataset=season_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    if saison == 'SON':
        xarrayseason=xr.DataArray(data_mensuel[11].mean(axis=0) + data_mensuel[0].mean(axis=0) + 
                              data_mensuel[0].mean(axis=0))
        season_dataset=xr.DataArray.to_dataset(xarrayseason,name='SON_' + champs)
        season_dataset=season_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    return season_dataset

dico_mois = {'Janvier':0, 'Fevrier':1, 'Mars':2, 'Avril':3, 'Mai' :4, 'Juin':5,
             'Juillet':6, 'Aout':7, 'Septembre' :8, 'Octobre':9,'Novembre':10,
             'Decembre':11}
liste_mois =  ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai' , 'Juin',
             'Juillet', 'Aout', 'Septembre' , 'Octobre','Novembre',
             'Decembre']

def data_mensuelle(dataset,champs,mois,dates,lon,lat):
    liste_mois=[]
    

    for i in range(1,13):
            liste_mois.append(np.any([dates.month==i],axis=0))
    if np.shape(dataset[champs].values) != (730, 30, 100) :
        temp = np.reshape(dataset[champs].values,(730, 30, 100))
        data_mensuel=[temp[liste_mois[i]] for i in range(12)]
   
        xarraymensuel=xr.DataArray(data_mensuel[dico_mois[mois]].mean(axis=0))
        mensuel_dataset=xr.DataArray.to_dataset(xarraymensuel, name=mois +'_' + champs)
        mensuel_dataset=mensuel_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    else :
        data_mensuel=[dataset[champs].values[liste_mois[i]] for i in range(12)]
   
        xarraymensuel=xr.DataArray(data_mensuel[dico_mois[mois]].mean(axis=0))
        mensuel_dataset=xr.DataArray.to_dataset(xarraymensuel, name=mois +'_' + champs)
        mensuel_dataset=mensuel_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    
    return mensuel_dataset

def data_journaliere(dataset,champs,mois,dates,lon,lat):
    liste_jours=[]
    

    for i in range(1,366):
            liste_mois.append(np.any([dates.day==i],axis=0))
    if np.shape(dataset[champs].values) != (730, 30, 100) :
        temp = np.reshape(dataset[champs].values,(730, 30, 100))
        data_mensuel=[temp[liste_jours[i]] for i in range(365)]
   
        xarraymensuel=xr.DataArray(data_mensuel[dico_mois[mois]].mean(axis=0))
        mensuel_dataset=xr.DataArray.to_dataset(xarraymensuel, name=mois +'_' + champs)
        mensuel_dataset=mensuel_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    else :
        data_mensuel=[dataset[champs].values[liste_mois[i]] for i in range(12)]
   
        xarraymensuel=xr.DataArray(data_mensuel[dico_mois[mois]].mean(axis=0))
        mensuel_dataset=xr.DataArray.to_dataset(xarraymensuel, name=mois +'_' + champs)
        mensuel_dataset=mensuel_dataset.assign_coords({'lat' : lat, 'lon' : lon})
    
    return mensuel_dataset


def moyenne_occurence_mensuelle(dataset,champs,dates,lon,lat):
    liste_dataset_mensuel = []
    nom_variables = []

    for mois in liste_mois : 
        liste_dataset_mensuel.append( data_mensuelle(dataset,champs,mois,dates,lon,lat) )
        nom_variables.append(mois+'_'+champs)
    occurences = []
    
    for k in range(len(liste_dataset_mensuel)) :
        occurences.append(np.nanmean(liste_dataset_mensuel[k][nom_variables[k]].values))
    return occurences

def nombre_occurence_mensuelle(dataset,champs,dates,lon,lat):
    liste_dataset_mensuel = []
    nom_variables = []

    for mois in liste_mois : 
        liste_dataset_mensuel.append( data_mensuelle(dataset,champs,mois,dates,lon,lat) )
        nom_variables.append(mois+'_'+champs)
    occurences = []
    
    for k in range(len(liste_dataset_mensuel)) :
        occurences.append(np.nansum(liste_dataset_mensuel[k][nom_variables[k]].values))
    return occurences    
        

    
    