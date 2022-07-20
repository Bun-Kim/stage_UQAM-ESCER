#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:10:01 2022

@author: san
"""
import xesmf as xe
import xarray as xr
import copy

method_list = [
    "bilinear",
    "nearest_s2d",
    "nearest_d2s",
    "patch",
]


def resolution_temporelle_obs(dico,data):
    #transforme les data obs en des data de pas de temps dico['frequence']
  if dico['method_obs'] == 'max':
      return data.resample(time=dico['frequence']).max()

def resolution_temporelle_model(dico2,data):
 #transforme les data model en des data de pas de temps dico['frequence']
 dico = copy.deepcopy(dico2)
 if dico['methode_temporelle_proxy'][-1] == 'max':
     dataset= data[dico['proxy']].resample(time=dico['frequence']).max()
 if dico['methode_temporelle_proxy'][-1] == 'mean':
     dataset= data[dico['proxy']].resample(time=dico['frequence']).mean()
 if dico['methode_temporelle_proxy'][-1] == 'sum':
     dataset= data[dico['proxy']].resample(time=dico['frequence']).sum()
     
 dico['proxy'].pop()
 dico['methode_temporelle_proxy'].pop()


 for k in range (len(dico['methode_temporelle_proxy'])):
   
     if dico['methode_temporelle_proxy'][k] == 'max':
         
         dataset[ dico ['proxy' ][k] ] = data [dico ['proxy' ][k]].resample(time=dico['frequence']).max()
     if dico['methode_temporelle_proxy'][k] == 'mean':
         dataset[ dico ['proxy' ][k] ] = data [dico ['proxy' ][k]].resample(time=dico['frequence']).mean()
     if dico['methode_temporelle_proxy'][k] == 'mean':
         dataset[ dico ['proxy' ][k] ] = data [dico ['proxy' ][k]].resample(time=dico['frequence']).sum()  
      
 return dataset

def test_data_a_reshape(data_obs,data_model):
    'retourne le tableau de donnees a ajuster, celui avec la plus petite resolution'
    resolution_obs= abs(data_obs.lat.values[0]-data_obs.lat.values[1])
    resolution_model= abs(data_model.lat.values[0]-data_model.lat.values[1])
    
    if resolution_obs < resolution_model :
        return ('obs')
    if resolution_obs > resolution_model :
        return ('model')
    else : 

        return ('meme resolution')    
'''
def resolution_spatiale_model(dico,data_obs,data_model):
   
    if test_data_a_reshape(data_obs,data_model) == 'model':
       
        data_regridded=[]
        for k in range (len(dico['proxy'])):
                        print(k)
                        champsk_to_regrid= data_model[dico['proxy'][k]]
                       
                        data_regridded.append(xe.Regridder(champsk_to_regrid,
                                              data_obs,
                                              dico['methode_spatiale'][k],
                                              periodic=True)(champsk_to_regrid)) 
                        
        return (xr.Dataset.merge(xr.DataArray.to_dataset(data_regridded[0]),
                                xr.DataArray.to_dataset(data_regridded[1])),data_regridded)
    else:
        return data_model
'''

import numpy as np

def resolution_spatiale_model(dico,data_obs,data_model):
    
    if test_data_a_reshape(data_obs,data_model) == 'model':
        resolution_obs= abs(data_obs.lat.values[0]-data_obs.lat.values[1])
        #resolution_obs=0.1
        data_regridded=[]
        
        ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(dico['latS'], dico['latN'], resolution_obs)),
        "lon": (["lon"], np.arange(dico['lonW'], dico['lonE'], resolution_obs)),})

        
        
        for k in range (len(dico['proxy'])):
                        print(k)
                        champsk_to_regrid= data_model[dico['proxy'][k]]
                       
                        data_regridded.append(xe.Regridder(champsk_to_regrid,
                                              ds_out,
                                              dico['methode_spatiale'][k],
                                              periodic=True,reuse_weights=True)(champsk_to_regrid)) 
                        
        return (xr.Dataset.merge(xr.DataArray.to_dataset(data_regridded[0]),
                                xr.DataArray.to_dataset(data_regridded[1])))
    else:
        return data_model
def resolution_spatiale_obs(dico,data,champs,resolution):
    

        data_regridded=[]
        
        ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(dico['latS'], dico['latN'], resolution)),
        "lon": (["lon"], np.arange(dico['lonW'], dico['lonE'], resolution)),})

        
        
        
        champs_to_regrid= data[champs]
                       
        data_regridded.append(xe.Regridder(champs_to_regrid,
                                              ds_out,
                                              'bilinear',
                                              periodic=True)(champs_to_regrid)) 
                        
        return xr.DataArray.to_dataset(data_regridded[0])
   
'''
def resolution_spatiale_obs(dico,data_obs,data_model):
    
    if test_data_a_reshape(data_obs,data_model) == 'obs':
        return xe.Regridder(data_obs,data_model[dico['proxy'][0]],'conservative',periodic =True)
    else:
        return data_obs
'''

def controle_resolution_spatiale_model_cape(dico,data_obs,data_model):
     Dataset=[]
     if test_data_a_reshape(data_obs,data_model) == 'model':
         resolution_obs= abs(data_obs.lat.values[0]-data_obs.lat.values[1])
         
         for method in method_list:
             print(method)
             data_regridded=[]
             
             ds_out = xr.Dataset({
             "lat": (["lat"], np.arange(dico['latS'], dico['latN'], resolution_obs)),
             "lon": (["lon"], np.arange(dico['lonW'], dico['lonE'], resolution_obs)),})
    
             
             
            
             champsk_to_regrid= data_model['cape']
                             
             data_regridded.append(xe.Regridder(champsk_to_regrid,
                                                   ds_out,
                                                   method,
                                                   periodic=True)(champsk_to_regrid)) 
            
             
             Dataset.append(xr.DataArray.to_dataset(data_regridded[0]))
         return Dataset
   
    


