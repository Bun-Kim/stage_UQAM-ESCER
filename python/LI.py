#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:24:24 2022

@author: san
"""


import xarray as xr
import metpy
import data_processing
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np

base = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2019,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','mean'],'methode_proxy':'mul','methode_spatiale':['bilinear','bilinear']}



t = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/t/t_era5_2018-2019_1.nc').sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))

t =t.sortby('lat')
t =t.sortby('level',ascending = False)
#t = t.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))

data_model = xr.open_dataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/t2m_d2m_era5_2018-2019.nc').sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))


Canada = []
mask =  data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/mask/ecozone_mask_1.nc').sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
for k in range(30):
    print(k)
    for i in range(100):
        if np.isnan(mask.ecozones.values[k,i]) ==False:
            Canada.append([k,i])



       

import time
from numba import jit

p = t.level.values * units.hPa

 #slice les data en lon/lat avant si ce n'est pas encore fait
data_t2m=data_model.t2m.values
data_d2m=data_model.d2m.values
all_t_levels = t.t.values



def calcul_LI():
    all_LI = np.empty(np.shape(data_model.d2m.values))
    all_LI.fill(np.nan)
    
    
    for k in range(17520):
        print(k)
        start = time.time()
        for lat,lon in Canada:
                       t2m = data_t2m[k,lat,lon]
                  
                       d2m = data_d2m[k,lat,lon]
                     
                       t_levels = all_t_levels[k,:,lat,lon]
                       
                       parcel_prof = mpcalc.parcel_profile(p, t2m * units.K, d2m* units.K)    
                       
                       LI= mpcalc.lifted_index(p, t_levels * units.K, parcel_prof)
                       p
                       all_LI[k,lat,lon]=np.array(LI)[0]
                       print('durée3 =' + str(time.time() - start))
        
        return all_LI
    
    
calcul_LI()


parcel_prof = mpcalc.parcel_profile(p, data_model.t2m.values * units.K, data_model.d2m.values* units.K) 


