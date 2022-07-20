#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:18:34 2022

@author: san
"""
import xarray as xr

def proxy_calculus(dico,data_model):
    temp=1
    if dico['methode_proxy']== 'mul':
        for champs in dico['proxy']:
            temp = temp * data_model[champs]
            print(champs)
    return xr.DataArray.to_dataset(temp,name='proxy')
            
    