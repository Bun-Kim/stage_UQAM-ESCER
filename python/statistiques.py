#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:03:28 2022

@author: san
"""
import xarray as xr
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

ecozones=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
          'Southern_Arctic','Boreal_Plains','Montane_Cordillera','Prairies','Pacific_Maritime','Arctic_Cordillera',
          'Taiga_Shield_E','Atlantic_Maritime','Boreal_Shield_E',
          'Mixedwood_Plains','Hudson_Plains','Boreal_Shield_S']

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

def correlation_glissante_point(dataset_obs,champs_obs,dataset_model,champs_model,dates,longitude,latitude,window):
    rolling_obs = dataset_obs.sel(lat=latitude).sel(lon=longitude).rolling(time=window,center=True)
   
    with_dim_obs = rolling_obs.construct('window_dim')      
    
    rolling_model = dataset_model.sel(lat=latitude).sel(lon=longitude).rolling(time=window,center=True)
    with_dim_model = rolling_model.construct('window_dim') 
    
    delta = datetime.timedelta(days = window)
    startday2 = dates[0] + delta
    endday2 = dates[-1]-delta
    dates3 = pd.date_range(startday2, endday2, freq='D')
    jours=np.any([dates3.day!=29,dates3.month!=2],axis=0)
    dates3=dates3[jours]
    
    with_dim_obs=  with_dim_obs.sel(time=slice(startday2,endday2))
    with_dim_model=  with_dim_model.sel(time=slice(startday2,endday2))
    
    correlations=[]
    compteur = 0
    for k in range(len(correlations)):
        if np.isnan(correlations[k])== True:
            compteur = compteur +1
            
    
    
    for k in range(len(with_dim_obs[champs_obs].values)):
        correlations.append(np.corrcoef(with_dim_obs[champs_obs].values[k],with_dim_model[champs_model].values[k])[0,1])
    #plt.plot(correlations)
    return correlations
   
    
    plt.title('correlation sur une fenetre glissante de ' +str(window) + ' jours')


def correlation_glissante_membres(dataset_obs,champs_obs,dataset_model,champs_model,dates,lon,lat,window):
    membres=[]
    for latitude in lat:
        for longitude in lon:
            membres.append(correlation_glissante_point(dataset_obs,champs_obs,dataset_model,champs_model,dates,longitude,latitude,window))
    
    A=np.isnan(np.nanmean(membres,axis=1))==False
    points = np.where(A)[0]
    membres_non_nuls=[]
    
    for point in points :
        print(point)
        membres_non_nuls.append(membres[point])
    return membres_non_nuls

def correlation_glissante_all(dataset_obs,champs_obs,dataset_model,champs_model,dates,lon,lat,window):
    membres=[]
    for latitude in lat:
        for longitude in lon:
            membres.append(correlation_glissante_point(dataset_obs,champs_obs,dataset_model,champs_model,dates,longitude,latitude,window))
   
    return membres

def trace_membre(membres,ecozone_name):
    plt.plot(np.nanmean(membres,axis=0),'r',label='moyenne',alpha=0.6)
    plt.plot(np.nanpercentile(membres,25,axis=0),'k--',linewidth=0.5,label='1er quartile')
    plt.plot(np.nanpercentile(membres,75,axis=0),'k--',linewidth=0.5,label='3e quartile')

    plt.xlabel('jours')
    plt.ylabel('correlation de Pearson')
    plt.fill_between(np.linspace(0,len(membres[0])-1,len(membres[0])),np.nanpercentile(membres,25,axis=0), np.nanpercentile(membres,75,axis=0), color='#539ecd',alpha=0.7)
    plt.legend()
    plt.title('correlations de Pearson proxy/F sur les points de grille de l ecozone ' + ecozone_name)
    plt.grid()
    plt.show()
    
def correlations_glissante_membres_ecozones(dataset_obs_ecozones,champs_obs,dataset_model_ecozones,champs_model,dates,lon,lat,window):
    n = len(dataset_model_ecozones)
    for k in range(n):
        membres_ecozone_k = correlation_glissante_membres(dataset_obs_ecozones[k], champs_obs, dataset_model_ecozones[k]
                                                                       , champs_model, dates, lon, lat, window)
        trace_membre(membres_ecozone_k, ecozones[k] )

        