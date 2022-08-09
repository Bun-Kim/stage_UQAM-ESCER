#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:15:10 2022

@author: san
"""
import glob
import numpy as np
import xarray as xr

files='/bwk01_01/san/stage_UQAM-ESCER/data/model/*.nc'
data=xr.open_mfdataset(files)

data.to_netcdf('/bwk01_01/san/stage_UQAM-ESCER/data/model/test.nc', 'w')