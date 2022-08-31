#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:56:14 2022

@author: san
"""

import data_processing
import ajustement
import proxy_calculation as pc
import carte
import statistiques

base = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2019,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','mean'],'methode_proxy':'mul','methode_spatiale':['bilinear','bilinear']}



#etape 1 : initialisation
infile0_obs=[]
infile0_model=[]
varname_obs=[]
varname_model=[]
debut=[]
fin=[]
time=[]
lat=[]
lon=[]
data_season=[]
saison=[]
dates=[]
nom_fichier=[]


a,b,c,d= data_processing.initialise_variables(base)
#b='../data/model/cape_cp_era5_2018-2020_1.nc'
infile0_obs.append(a)
infile0_model.append(b)
varname_obs.append(c)
varname_model.append(d)

print('ouverture des fichiers')

temp = data_processing.ouvre_fichier(a)
temp1 = data_processing.ouvre_fichier(b)

#print('ajustement mask')
#temp1 = data_processing.mask_canada(temp1)

temp = data_processing.changement_nom_coordonnees_obs(base, temp)
temp1 = data_processing.changement_nom_coordonnees_model(base, temp1)

temp1 = temp1.assign_coords(lon=(((temp1.lon + 180) % 360) - 180)).sortby('lon').sortby('lat')
temp1= temp1.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))

print('ajustement des obs')
print('ajustement temporel')
temp=ajustement.resolution_temporelle_obs(base, temp)
print('ajustement spatial')
#temp3=ajustement.resolution_spatiale_obs(base,temp,0.1)


[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,temp  )



print('ajustement des donnees pour le proxy')
print('ajustement temporel')
temp1=ajustement.resolution_temporelle_model(base, temp1)
print('ajustement spatial')
temp2 = ajustement.resolution_spatiale_model(base,temp,temp1)

#Dataset_controle_methode=ajustement.controle_resolution_spatiale_model_cape(base,temp,temp1)

[t_model,la_model,lo_model,da_model,date2_model]=data_processing.selectionDonnees(base,temp2)

proxy= pc.proxy_calculus(base, da_model) *20

#attention si on ajuste sur les donnees modeles, faire une fonction pour trier les lat

###############
#carte.trace_controle_methode_regrid(base, Dataset_controle_methode, 'cape')

#proxy = proxy.where(proxy.proxy<10)

carte.tracer(base, proxy/10000, 'proxy')
carte.tracer(base, da_obs/(100*100),'F')


####################3

from netCDF4 import Dataset
infile1 = '../data/observation/F.nc'
da_obs.to_netcdf(infile1)

infile2='../data/model/proxy.nc'
proxy.to_netcdf(infile2)

filename = infile1
ncin = Dataset(filename, 'r')
lo_obs = ncin.variables['lon'][:]
la_obs = ncin.variables['lat'][:]
t_obs = ncin.variables['time'][:]
da_obs.F.values = ncin.variables['F'][:]
ncin.close()

filename = infile2
ncin = Dataset(filename, 'r')
lo_model = ncin.variables['lon'][:]
la_model = ncin.variables['lat'][:]
t_model = ncin.variables['time'][:]
proxy.proxy.values = ncin.variables['proxy'][:]
ncin.close()
##############

A= data_processing.ouvre_fichier(infile2
                                 )
















import numpy as np

F_values= da_obs.F.values/(100*100)
proxy_values = proxy.proxy.values/10000

F_mean_annuel= np.mean(da_obs.F.values,axis=0)/(100*100)
proxy_mean_annuel = np.mean(proxy.proxy.values,axis = 0)/10000


##############
#attention application du mask tres couteuse
#carte.tracer(base, data_processing.domaine_canada(proxy/10000), 'proxy')
#carte.tracer(base, data_processing.domaine_canada(da_obs/(100*100)),'F')

####
#saison = {'DJF','MAM','JJA','SON'}
DJF_dataset_obs = statistiques.data_saison(da_obs/(100*100), 'F', 'DJF', date2_obs, lo_obs, la_obs)
DJF_dataset_proxy = statistiques.data_saison(0.39*proxy/10000, 'proxy', 'DJF', date2_obs, lo_obs, la_obs)

MAM_dataset_obs = statistiques.data_saison(da_obs/(100*100), 'F', 'MAM', date2_obs, lo_obs, la_obs)
MAM_dataset_proxy = statistiques.data_saison(0.39*proxy/10000, 'proxy', 'MAM', date2_obs, lo_obs, la_obs)

JJA_dataset_obs = statistiques.data_saison(da_obs/(100*100), 'F', 'JJA', date2_obs, lo_obs, la_obs)
JJA_dataset_proxy = statistiques.data_saison(0.39*proxy/10000, 'proxy', 'JJA', date2_obs, lo_obs, la_obs)

SON_dataset_obs = statistiques.data_saison(da_obs/(100*100), 'F', 'SON', date2_obs, lo_obs, la_obs)
SON_dataset_proxy = statistiques.data_saison(0.39*proxy/10000, 'proxy', 'SON', date2_obs, lo_obs, la_obs)

#saison_dataset_obs names = {'DJF_F','MAM_F','JJA_F','SON_F'}
#saison_dataset_proxy names = {'DJF_proxy','MAM_proxy','JJA_proxy','SON_proxy'}
carte.tracer_saison(base, DJF_dataset_obs, "DJF_F")
carte.tracer_saison(base, DJF_dataset_proxy, "DJF_proxy")

carte.tracer_saison(base, MAM_dataset_obs, "MAM_F")
carte.tracer_saison(base, MAM_dataset_proxy, "MAM_proxy")

carte.tracer_saison(base, JJA_dataset_obs, "JJA_F")
carte.tracer_saison(base, JJA_dataset_proxy, "JJA_proxy")

carte.tracer_saison(base, SON_dataset_obs, "SON_F")
carte.tracer_saison(base, SON_dataset_proxy, "SON_proxy")
'''
import xarray as xr
import numpy as np
mask = xr.open_mfdataset('ERA5_mask_Canadian_timezone_ESRI_v4.nc')
#mask = mask.rename({'longitude': 'lon','latitude': 'lat'})


DJF_dataset_obs=DJF_dataset_obs.where(mask.region > 0 )
carte.tracer_saison(base, DJF_dataset_obs, "DJF_F")
'''    if np.isnan(mask.ecozones.values[k,i]) ==False:
            Canada.append([k,i])
data= data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/lfc_el/03-2018_1.nc')

#mask_array=np.asarray(mask.region.values)
#mask_dataaray=xr.DataArray(mask_array)
#mask_dataset=xr.DataArray.to_dataset(mask_dataaray,name='region')
#mask_dataset=mask_dataset.assign_coords({'lat' : mask.latitude.values, 'lon' : mask.longitude.values})
#mask_regridded = ajustement.resolution_spatiale_obs(base, mask_dataset,'region', 1)

infile_obs = '/home/bun-kim/Documents/stage_ESCER/var/observation/F.nc'
infile_model  ='/home/bun-kim/Documents/stage_ESCER/var/model/proxy.nc'
F =  data_processing.ouvre_fichier(infile_obs)
proxy = data_processing.ouvre_fichier(infile_model)

F = F.assign_coords(lon=(((F.lon + 180) % 360) - 180)).sortby('lon').sortby('lat')
proxy = proxy.assign_coords(lon=(((proxy.lon + 180) % 360) - 180)).sortby('lon').sortby('lat')


[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,F )
[t_model,la_model,lo_model,proxy,date2_model]=data_processing.selectionDonnees(base,proxy)

carte.tracer(base, proxy/10000, 'proxy')
carte.tracer(base, da_obs/(100*100),'F')



Canada = []
mask =  data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/mask/ecozone_mask_1.nc').sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
for k in range(30):
    print(k)
    for i in range(100):
        if np.isnan(mask.ecozones.values[k,i]) ==False:
            Canada.append([k,i])
data= data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/lfc_el/03-2018_1.nc')


r_xarray= data.r.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
r_xarray = r_xarray.sortby('level',ascending = False)
r_xarray = xr.DataArray.to_dataset(r_xarray, name='r')

t_xarray= data.t.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
t_xarray =t_xarray.sortby('level',ascending = False)
t_xarray = xr.DataArray.to_dataset(t_xarray, name='t')

#r_xarray = data_processing.mask_canada(base,r_xarray )
#t_xarray = data_processing.mask_canada(base,t_xarray )


import metpy
from metpy.units import units
import metpy.calc as mpcalc


pressure = r_xarray.level.values * units.hPa
t = t_xarray.t.values * units.K
r =  r_xarray.r.values/100 

'''
temp = []
for k in range(3000):
    
    temp.append(r_t.level.values)

temp = np.reshape(temp,(19,100,30))
temp= -np.sort(-temp,axis=0)

all_pressure = []
for k in range(1488):
    
    all_pressure.append(temp)
all_pressure = np.array(all_pressure) * units.hPa
'''
periode,levels,latitudes,longitudes = np.shape(t)



dp = mpcalc.dewpoint_from_relative_humidity(t, r)

import xarray as xr
xrdp = xr.DataArray(dp,name='dp')

xrdp = xrdp.rename({'dim_0':'time','dim_1':'level','dim_2':'lat','dim_3':'lon'})
xrdp=xrdp.assign_coords({'time' : r_xarray.time.values, 'level' : r_xarray.level.values,'lat' : r_xarray.lat.values,'lon' : r_xarray.lon.values})

xrdp = xr.DataArray.to_dataset(xrdp)
periode,levels,latitudes,longitudes = np.shape(r_xarray.r.values)

data_processing.sauvegarde_donnees(xrdp, xrdp.lat.values, xrdp.lon.values, xrdp.time.values, 'dp')

####

import time





start = time.time()

DATA = ["02-1_dp.nc","02-2_dp.nc","03-1_dp.nc","03-2_dp.nc","04-1_dp.nc","04-2_dp.nc"]

for fichier in DATA :
    xrdp = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/var/model/' + fichier)
    periode,levels,latitudes,longitudes = np.shape(xrdp.dp.values)
    
    liste_probleme =[]
    lfc=np.zeros((periode,latitudes,longitudes))
    lfc_T=np.zeros((periode,latitudes,longitudes))
    el=np.zeros((periode,latitudes,longitudes))
    el_T=np.zeros((periode,latitudes,longitudes))
    pressure = xrdp.level.values * units.hPa
    for pas in range(periode):
        print(pas)
        for lat,lon in Canada: 
            
    
                    print(pas,lat,lon)
                    try :
                        temp_lfc = list(mpcalc.lfc(pressure, t[pas,:,lat,lon], dp[pas,:,lat,lon], parcel_temperature_profile=None, dewpoint_start=None, which='bottom'))
                        temp_el = list(mpcalc.el(pressure, t[pas,:,lat,lon], dp[pas,:,lat,lon], parcel_temperature_profile=None,  which='top'))
                        lfc[pas,lat,lon]=   float(temp_lfc[0]/units.hPa)
                        lfc_T[pas,lat,lon]=  float(temp_lfc[1]/units.K)
                        el[pas,lat,lon]= float(temp_el[0]/units.hPa)
                        el_T[pas,lat,lon]= float(temp_el[1]/units.K)
                        
                    except ValueError:
                        print("probleme")
                        liste_probleme.append([pas,lat,lon])
                        lfc[pas,lat,lon] = np.nan
                        lfc_T[pas,lat,lon] =np.nan
                        el[pas,lat,lon] =np.nan
                        el_T[pas,lat,lon] =np.nan
    end = time.time()
    print('durée =' + str(end - start))            
    
    
    
    
    
    LFC = lfc 
    LFC_T = lfc_T 
    EL = el 
    EL_T = el_T 
    
    LFC = xr.DataArray.to_dataset(xr.DataArray(LFC), name='lfc')
    LFC_T = xr.DataArray.to_dataset(xr.DataArray(LFC_T),name='lfc_t')
    EL= xr.DataArray.to_dataset(xr.DataArray(EL),name='el')
    EL_T = xr.DataArray.to_dataset(xr.DataArray(EL_T), name='el_t')
    
    
    LFC = LFC.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
    LFC_T = LFC_T.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
    EL = EL.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
    EL_T = EL_T.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
    
    
    LFC = LFC.assign_coords({'time' : xrdp.time.values,'lat' : xrdp.lat.values,'lon' : xrdp.lon.values})
    LFC_T = LFC_T.assign_coords({'time' : xrdp.time.values,'lat' : xrdp.lat.values,'lon' : xrdp.lon.values})
    EL = EL.assign_coords({'time' : xrdp.time.values,'lat' : xrdp.lat.values,'lon' : xrdp.lon.values})
    EL_T = EL_T.assign_coords({'time' : xrdp.time.values,'lat' : xrdp.lat.values,'lon' : xrdp.lon.values})
    
    
    
    
    data_processing.sauvegarde_donnees(LFC, xrdp.lat.values, xrdp.lon.values, xrdp.time.values, 'lfc','lfc'+fichier[:-3])
    data_processing.sauvegarde_donnees(LFC_T, xrdp.lat.values, xrdp.lon.values, xrdp.time.values, 'lfc_t','lfc_t'+fichier[:-3])
    data_processing.sauvegarde_donnees(EL, xrdp.lat.values, xrdp.lon.values, xrdp.time.values, 'el','el'+ fichier[:-3] )
    data_processing.sauvegarde_donnees(EL_T, xrdp.lat.values, xrdp.lon.values, xrdp.time.values, 'el_t','el_t' + fichier[:-3])





LFC = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/lfc.nc').isel(time = slice(0,278))
LFC_T = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/lfc_t.nc').isel(time = slice(0,278))
EL = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/el.nc').isel(time = slice(0,278))
EL_T = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/el_t.nc').isel(time = slice(0,278))

capecp = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/cape_cp_era5_2018_1.nc')
capecp = capecp.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
capecp = capecp.sel(time=slice('2018-07-01T00:00:00.000000000','2018-07-12T13:00:00.000000000'))

d950 = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/d950_era5.nc')
d950 = d950.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
d950 = d950.sel(time=slice('2018-07-01T00:00:00.000000000','2018-07-12T13:00:00.000000000'))

K=(LFC.lfc-EL.el)
K = xr.DataArray.to_dataset(K, name='K')

proxy = capecp.cape * capecp.cp
proxy = xr.DataArray.to_dataset(proxy, name='proxy')

F =  data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/observation/WWLLN_2010-2019.nc')
F = F.rename({'Time':'time'})
F = F.assign_coords(lon=(((F.lon + 180) % 360) - 180)).sortby('lon').sortby('lat')
F = F.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
F = F.sel(time=slice('2018-07-01','2018-07-12'))

proxy_d950 = proxy.proxy * d950.d
proxy_d950 = xr.DataArray.to_dataset(proxy_d950, name='proxy_d950')

proxy_d950_K = proxy_d950.proxy_d950 * capecp.cape/ K.K
proxy_d950_K = xr.DataArray.to_dataset(proxy_d950_K, name='proxy_d950_K')


base_proxy = {'method_obs':'max','frequence':'1D'}
proxy_journalier = ajustement.resolution_temporelle_obs(base_proxy, proxy)

base_proxy_d950 = {'method_obs':'min','frequence':'1D'}
proxy_d950_journalier = ajustement.resolution_temporelle_obs(base_proxy_d950, proxy_d950.where(proxy_d950.proxy_d950<0))

base_proxy_d950_K = {'method_obs':'min','frequence':'1D'}
proxy_d950_K_journalier = ajustement.resolution_temporelle_obs(base_proxy_d950_K, proxy_d950_K)


base_K = {'method_obs':'max','frequence':'1D'}
K_journalier = ajustement.resolution_temporelle_obs(base_K, K)

cape_K= capecp.cape / K.K
cape_K = xr.DataArray.to_dataset(cape_K, name='cape_K')

base_cape_K = {'method_obs':'max','frequence':'1D'}
cape_K_journalier = ajustement.resolution_temporelle_obs(base_cape_K, cape_K)

proxy_K= proxy.proxy * capecp.cape / K.K
proxy_K = xr.DataArray.to_dataset(proxy_K, name='proxy_K')

base_proxy_K = {'method_obs':'max','frequence':'1D'}
proxy_K_journalier = ajustement.resolution_temporelle_obs(base_proxy_K, proxy_K)

cape_cp_K= capecp.cp * capecp.cape / K.K
cape_cp_K = xr.DataArray.to_dataset(cape_cp_K, name='cape_cp_K')

base_cape_cp_K = {'method_obs':'max','frequence':'1D'}
cape_cp_K_journalier = ajustement.resolution_temporelle_obs(base_cape_cp_K, cape_cp_K)

cape2_cp_K= capecp.cp * capecp.cape * capecp.cape/ K.K
cape2_cp_K = xr.DataArray.to_dataset(cape2_cp_K, name='cape2_cp_K')

base_cape2_cp_K = {'method_obs':'max','frequence':'1D'}
cape2_cp_K_journalier = ajustement.resolution_temporelle_obs(base_cape2_cp_K, cape2_cp_K)


cape_d950_K= capecp.cp * d950.d / K.K
cape_d950_K = xr.DataArray.to_dataset(cape_d950_K, name='cape_d950_K')

base_cape_d950_K = {'method_obs':'min','frequence':'1D'}
cape_d950_K_journalier = ajustement.resolution_temporelle_obs(base_cape_d950_K, cape_d950_K)

cape2_d950_K= capecp.cp * capecp.cp * d950.d / K.K
cape2_d950_K = xr.DataArray.to_dataset(cape2_d950_K, name='cape2_d950_K')

base_cape2_d950_K = {'method_obs':'min','frequence':'1D'}
cape2_d950_K_journalier = ajustement.resolution_temporelle_obs(base_cape2_d950_K, cape2_d950_K)


vimd = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/vmind_era5.nc')
vimd = vimd.sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))

cape_vimd = capecp.cp * vimd.vimd
cape_vimd = xr.DataArray.to_dataset(cape_vimd, name='cape_vimd')

base_cape_vimd = {'method_obs':'max','frequence':'1D'}
cape_vimd_journalier = ajustement.resolution_temporelle_obs(base_cape_vimd, cape_vimd)


#ondes de gravites
lgws_gwd = data_processing.ouvre_fichier('/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/lgws_gwd_era5.nc')
lgws_gwd  = lgws_gwd .sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE']))
lgws_gwd  = lgws_gwd .sel(time=slice('2018-07-01','2018-07-12'))

#proxy_lgws= proxy.proxy * lgws_gwd.lgws 
proxy_lgws= proxy.proxy * xr.ufuncs.exp( (1/2) * xr.ufuncs.log2(lgws_gwd.lgws))
proxy_lgws = xr.DataArray.to_dataset(xr.ufuncs.fabs(proxy_lgws), name='proxy_lgws')

base_proxy_lgws = {'method_obs':'max','frequence':'1D'}
proxy_lgws_journalier = ajustement.resolution_temporelle_obs(base_proxy_lgws, proxy_lgws)


proxy_gwd= proxy.proxy * lgws_gwd.gwd 
#proxy_gwd = xr.DataArray.to_dataset(xr.ufuncs.exp( (2) * xr.ufuncs.log2(proxy_gwd)), name='proxy_gwd')
proxy_gwd = xr.DataArray.to_dataset(proxy_gwd, name='proxy_gwd')
base_proxy_gwd = {'method_obs':'max','frequence':'1D'}
proxy_gwd_journalier = ajustement.resolution_temporelle_obs(base_proxy_gwd, proxy_gwd)


for k in range(10):

    carte.tracer(base, data_processing.mask_canada(base,F.isel(time=k)),'F',str(F.isel(time=k).time.values))
    
    #carte.tracer(base, data_processing.mask_canada(base,-cape_d950_K_journalier .isel(time=k)),'cape_d950_K',' cape_d950_K_'  + str(F.isel(time=k).time.values))
    #carte.tracer(base, data_processing.mask_canada(base,-cape2_d950_K_journalier .isel(time=k)),'cape2_d950_K',' cape2_d950_K ' + str(F.isel(time=k).time.values))
    
    carte.tracer(base, data_processing.mask_canada(base,proxy_journalier.isel(time=k)),'proxy','proxy cape*cp ' + str(F.isel(time=k).time.values))
    #carte.tracer(base, data_processing.mask_canada(base,proxy_K_journalier.isel(time=k)),'proxy_K','proxy_K' + str(F.isel(time=k).time.values))
    #carte.tracer(base, data_processing.mask_canada(base,-proxy_d950_journalier.isel(time=k)),'proxy_d950','proxy_d950 ' + str(F.isel(time=k).time.values))
    carte.tracer(base, data_processing.mask_canada(base,proxy_gwd_journalier.isel(time=k)),'proxy_gwd','proxy_gwd ' + str(F.isel(time=k).time.values))
    #carte.tracer(base, data_processing.mask_canada(base,-proxy_d950_K_journalier.isel(time=k)),'proxy_d950_K',' proxy cape*cp*d950*K ' + str(F.isel(time=k).time.values))
    plt.imshow(A)

cape = xr.DataArray.to_dataset(capecp.cape ,name='cape')
cp = xr.DataArray.to_dataset(capecp.cp ,name='cp')
for k in range(20,48):
      carte.tracer(base, data_processing.mask_canada(base,F.isel(time=1)),'F',str(F.isel(time=1).time.values))
      carte.tracer(base, data_processing.mask_canada(base,cape.isel(time=k)),'cape',str(cape.isel(time=k).time.values))
      carte.tracer(base, data_processing.mask_canada(base,cp.isel(time=k)),'cp',str(cape.isel(time=k).time.values))
      
for k in range(24,48):
      carte.tracer(base, data_processing.mask_canada(base,K.isel(time=k)),'K',str(K.isel(time=k).time.values))
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units        
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

lat_test = 15
lon_test = 15

mpcalc.lfc(pressure, t[0,:,lat_test,lon_test], dp[0,:,lat_test,lon_test ], parcel_temperature_profile=None, dewpoint_start=None, which='top')
mpcalc.el(pressure, t[0,:,lat_test,lon_test ], dp[0,:,lat_test,lon_test ], parcel_temperature_profile=None,  which='top')
np
# Create arrays of pressure, temperature, dewpoint, and wind components
p = pressure

# Calculate parcel profile
prof = mpcalc.parcel_profile(p, t[0,:,lat_test,lon_test ][0], dp[0,:,lat_test,lon_test ][0]).to('K')

skew.plot(p, t[0,:,lat_test,lon_test ], 'r')
skew.plot(p, dp[0,:,lat_test,lon_test ], 'g')
skew.plot(p, prof, 'k')  # Plot parcel profile

skew.ax.set_xlim(-50, 15)
skew.ax.set_ylim(1000, 100)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()


# Calculate LCL and LFC height and plot as black dot

el_pressure_top, el_temperature_top = mpcalc.el(p, t[0,:,lat_test,lon_test ], dp[0,:,lat_test,lon_test ],parcel_temperature_profile=None,  which='top')
el_pressure_bottom, el_temperature_bottom = mpcalc.el(p, t[0,:,lat_test,lon_test ], dp[0,:,lat_test,lon_test ],parcel_temperature_profile=None,  which='bottom')

skew.plot(el_pressure_top, el_temperature_top, 'ko', markerfacecolor='black')
#skew.plot(el_pressure_bottom, el_temperature_bottom, 'go', markerfacecolor='green')

plt.show()



data_processing.sauvegarde_donnees(xr.DataArray.to_dataset(r_t.t).sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE'])), xr.DataArray.to_dataset(r_t.t).lat.values, xr.DataArray.to_dataset(r_t.t).lon.values,
                                   xr.DataArray.to_dataset(r_t.t).sel(lat=slice(base['latS'],base['latN'])).sel(lon=slice(base['lonW'],base['lonE'])).time.values, 't')


