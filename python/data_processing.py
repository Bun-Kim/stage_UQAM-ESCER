#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:56:14 2022

@author: san
"""


import os
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import subprocess

dir_obs='../data/observation/'
dir_model='../data/model/'
dir_res='./result/'
dir_figs='./figs/'
dir_anim='./anim/'

if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)
if not os.path.exists(dir_anim):
    os.makedirs(dir_anim)
if not os.path.exists(dir_res):
    os.makedirs(dir_res)
    
#etape 1 : initialisation


def mask_canada(dataset):
    
    mask = xr.open_mfdataset('ERA5_mask_Canadian_timezone_ESRI_v4.nc')
    return dataset.where(mask.region>0)
def initialise_variables(dico):
   
    if dico['model_obs'] == 'WWLLN':
        infile_obs = dir_obs+'WWLLN_2010-2019.nc'
        
    if dico['model_obs'] == 'Blitz':
         infile_obs = dir_obs+'Blitz_2015-2021.nc' 

    if dico['model_proxy'] == 'Era5':
        infile_model = dir_model
        for k in range(len(dico['proxy'])):
            infile_model = infile_model + dico['proxy'][k] + '_' 
        infile_model = infile_model + 'era5_' + str(dico['anneeDebut_model']) + '-' + str(dico['anneeFin_model']) + '.nc'
    varname_obs = dico['model_obs'] + str(dico['resolution']) + ' deg'
    varname_model = dico['model_proxy'] + str(dico['resolution']) + ' deg'
    
    return [infile_obs,infile_model,varname_obs,varname_model]


'''
def fusion_xarray(dico):
    all_data=[]
    infile_model = dir_model
    for k in range(len(dico['proxy'])):
        infile_model = infile_model + dico['proxy'][k] + '_' 
    infile_model = infile_model + 'era5_'
    for i in range(dico['anneeDebut_model'],dico['anneeFin_model']+1):
        all_data.append(xr.open_dataset(infile_model+ str(i) + '.nc'))
        
    return(xr.merge(all_data))
'''    
def ouvre_fichier(infile):
    return xr.open_dataset(infile)
   
def changement_nom_coordonnees_obs(dico,data):
    if dico['model_obs'] == 'WWLLN':
        return data.rename({'Time': 'time'})
  
def changement_nom_coordonnees_model(dico,data):
    if dico['model_proxy'] == 'Era5':
        
        return data.rename({'longitude': 'lon','latitude': 'lat'})
       

def selectionDonnees(dico,data):
    debut = str(dico['anneeDebut'])+'-01-01'
    fin = str(dico['anneeFin'])+'-12-31'
    dates=pd.date_range(debut,fin,freq=dico['frequence'])
    
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon').sortby('lat')
   
    data_season = data.sel(time=dates)
    data_season = data.sel(lat=slice(dico['latS'],dico['latN'])).sel(lon=slice(dico['lonW'],dico['lonE'])).sel(time=dates)
    lat  = data_season.lat.values
    lon  = data_season.lon.values
    time = data_season.time.values
    return [time,lat,lon,data_season,dates]


def domaine_canada(data):
    mask = xr.open_mfdataset('ERA5_mask_Canadian_timezone_ESRI_v4.nc')
    return data.where(mask.region > 0 )

def sauvegarde_donnees(data,lat,lon,t,champs):
    if champs == 'proxy':
        infile = '../var/model/proxy.nc'
        data.to_netcdf(infile)
        filename = infile
    if champs == 'cape':
        infile = '../var/model/cape.nc'
        data.to_netcdf(infile)
        filename = infile
    if champs == 'cp':
        infile = '../var/model/cp.nc'
        data.to_netcdf(infile)
        filename = infile
        
        
    if champs == 'F':
        infile = '../var/observation/F.nc'
    
    data.to_netcdf(infile)
    filename = infile
    ncin = Dataset(filename, 'r')
    lon = ncin.variables['lon'][:]
    lat = ncin.variables['lat'][:]
    t = ncin.variables['time'][:]
    data[champs].values = ncin.variables[champs][:]
    ncin.close()
        
      
