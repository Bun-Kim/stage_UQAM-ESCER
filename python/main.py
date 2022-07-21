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


base_Canada = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2020,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':89.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','mean'],'methode_proxy':'mul','methode_spatiale':['bilinear','bilinear']}


'''
base = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2020,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':45.5,
        'latN':49.5,'lonW':-109.5,'lonE':-85.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','mean'],'methode_proxy':'mul','methode_spatiale':['bilinear','bilinear']}
'''

base=base_Canada
liste_mois =  ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai' , 'Juin',
             'Juillet', 'Aout', 'Septembre' , 'Octobre','Novembre',
             'Decembre']


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
#temp1 = data_processing.changement_nom_coordonnees_model(base, temp1)




print('ajustement des obs')
print('ajustement temporel')
temp=ajustement.resolution_temporelle_obs(base, temp)
print('ajustement spatial')
#temp3=ajustement.resolution_spatiale_obs(base,temp,0.1)


[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,temp)



print('ajustement des donnees pour le proxy')
print('ajustement temporel')
temp1=ajustement.resolution_temporelle_model(base, temp1)
#print('ajustement spatial')
#temp1 = ajustement.resolution_spatiale_model(base,temp,temp1)

#Dataset_controle_methode=ajustement.controle_resolution_spatiale_model_cape(base,temp,temp1)

[t_model,la_model,lo_model,da_model,date2_model]=data_processing.selectionDonnees(base,temp1)

proxy= pc.proxy_calculus(base, da_model) *20

#attention si on ajuste sur les donnees modeles, faire une fonction pour trier les lat

###############
#carte.trace_controle_methode_regrid(base, Dataset_controle_methode, 'cape')

#proxy = proxy.where(proxy.proxy<10)


####################3

print('sauvegarde des donnees en fichier netcdf')
import xarray as xr

data_processing.sauvegarde_donnees(da_obs, la_obs, lo_obs, t_obs, "F")
data_processing.sauvegarde_donnees(proxy, la_model, lo_model, t_model, "proxy")
data_processing.sauvegarde_donnees(xr.DataArray.to_dataset(da_model['cp'],name='cp'),la_model, lo_model, t_model,'cp')
data_processing.sauvegarde_donnees(xr.DataArray.to_dataset(da_model['cape'],name='cape'),la_model, lo_model, t_model,'cape')
                                     
##############

print('mask.sh a aller executer ')

infile_var_cp = '../var/model/cp.nc'
infile_var_cape = '../var/model/cape.nc'
infile_var_proxy = '../var/model/proxy.nc'
infile_var_obs = '../var/observation/F.nc'
infile_var_obs_Canada = '../var/observation/F_Canada.nc'

cp=data_processing.ouvre_fichier(infile_var_cp)
cape=data_processing.ouvre_fichier(infile_var_cape)
proxy=data_processing.ouvre_fichier(infile_var_proxy)
da_obs=data_processing.ouvre_fichier(infile_var_obs)
da_obs_Canada=data_processing.ouvre_fichier(infile_var_obs_Canada)


[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,da_obs)
[t_model,la_model,lo_model,proxy,date2_model]=data_processing.selectionDonnees(base,proxy)
_,_,_,cp,_= data_processing.selectionDonnees(base,cp)
_,_,_,cape,_ = data_processing.selectionDonnees(base,cape)
_,_,_,da_obs_Canada,_ = data_processing.selectionDonnees(base_Canada,da_obs_Canada)

print('trace carte')
carte.tracer_moyenne(base, proxy/10000, 'proxy')
carte.tracer_moyenne(base, da_obs/(100*100),'F')
carte.tracer_moyenne(base, cp, 'cp')
carte.tracer_moyenne(base, cape,'cape')

#import numpy as np
#proxy=np.maximum(proxy,0)
#da_obs=np.maximum(da_obs,0)



##############
#attention application du mask tres couteuse
#carte.tracer(base, data_processing.domaine_canada(proxy/10000), 'proxy')
#carte.tracer(base, data_processing.domaine_canada(da_obs/(100*100)),'F')

####
'''
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
#data mensuelle et plot
#Janvier_dataset_F = statistiques.data_mensuelle(da_obs, 'F', 'Janvier', date2_obs, lo_obs, la_obs)
#carte.tracer(base, Janvier_dataset_F,'Janvier_F')
carte.trace_occurences_mensuelles(base_Canada,base,da_obs_Canada,da_obs, [cape,cp,proxy], ['F'], ['cape','cp','proxy'], date2_model, lo_model, la_model, 'sum')
carte.trace_occurences_mensuelles(base_Canada,base,da_obs_Canada,da_obs, [cape,cp,proxy], ['F'], ['cape','cp','proxy'], date2_model, lo_model, la_model, 'mean')

mask = xr.open_mfdataset('ERA5_mask_Canadian_timezone_ESRI_v4.nc')
temp=temp.where(mask.region > 0 )






