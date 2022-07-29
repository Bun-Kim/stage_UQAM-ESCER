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

import matplotlib.pyplot as plt

base_Canada = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2020,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','mean'],'methode_proxy':'mul','methode_spatiale':['bilinear','bilinear']}

ecozones=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
          'Southern_Arctic','Boreal_Plains','Montane_Cordillera','Prairies','Pacific_Maritime','Arctic_Cordillera',
          'Taiga_Shield_E','Atlantic_Maritime','Boreal_Shield_E',
          'Mixedwood_Plains','Hudson_Plains','Boreal_Shield_S']

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
'''
data_processing.sauvegarde_donnees(proxy, la_model, lo_model, t_model, "proxy")
data_processing.sauvegarde_donnees(xr.DataArray.to_dataset(da_model['cp'],name='cp'),la_model, lo_model, t_model,'cp')
data_processing.sauvegarde_donnees(xr.DataArray.to_dataset(da_model['cape'],name='cape'),la_model, lo_model, t_model,'cape')
'''                            
##############



'''
print("application des masks")
import os
path1='../var/model'
os.chdir(path1)
bashCommand = "bash mask.sh"
os.system(bashCommand) 

path2="../var/observation/"
os.chdir(path2)
bashCommand = "bash mask.sh"
os.system(bashCommand) 

'''

infile_mask = '../data/mask/ecozone_mask_1.nc'
ecozones=data_processing.ouvre_fichier(infile_mask)

liste_mask_ecozones = data_processing.liste_mask_ecozones(base_Canada,ecozones)

ecozones=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
          'Southern_Arctic','Boreal_Plains','Montane_Cordillera','Prairies','Pacific_Maritime','Arctic_Cordillera',
          'Taiga_Shield_E','Atlantic_Maritime','Boreal_Shield_E',
          'Mixedwood_Plains','Hudson_Plains','Boreal_Shield_S']

#############


print('mask.sh a aller executer ')


infile_var_cp = '../var/model/cp.nc'
infile_var_cape = '../var/model/cape.nc'
infile_var_proxy = '../var/model/proxy.nc'
infile_var_obs = '../var/observation/F.nc'
infile_var_obs_Canada = '../var/observation/F_Canada.nc'


#cp=data_processing.ouvre_fichier(infile_var_cp)
#cape=data_processing.ouvre_fichier(infile_var_cape)
#proxy=data_processing.ouvre_fichier(infile_var_proxy)

import xarray as xr
cp= xr.DataArray.to_dataset(da_model['cp'],name='cp')
cape= xr.DataArray.to_dataset(da_model['cape'],name='cape')
F=da_obs
da_obs_Canada=data_processing.ouvre_fichier(infile_var_obs_Canada)
#da_obs_Canada=data_processing.ouvre_fichier(infile_var_obs_Canada)


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
carte.tracer_moyenne(base, da_obs_Canada/(100*100),'F')


#############3
for i in range(0,18):
    #carte.tracer_moyenne(base_Canada,da_obs['F'] * liste_mask_ecozones[i] ,'ecozone')
    carte.tracer_moyenne_echelle(base_Canada,da_obs['F'] * liste_mask_ecozones[i], da_obs,'ecozone','F',)
   


cape_ecozones=data_processing.champs_ecozone(cape, liste_mask_ecozones, 'cape')
cp_ecozones=data_processing.champs_ecozone(cp, liste_mask_ecozones, 'cp')
proxy_ecozones=data_processing.champs_ecozone(proxy, liste_mask_ecozones, 'proxy')
F_ecozones=data_processing.champs_ecozone(F, liste_mask_ecozones, 'F')


for i in range(0,18):
    carte.trace_occurences_mensuelles(base_Canada,base,da_obs_Canada,F_ecozones[i],
                                              [cape_ecozones[i],cp_ecozones[i],proxy_ecozones[i]],
                                              ['F'], ['cape','cp','proxy'], date2_model, lo_model, 
                                              la_model, 'mean',ecozones[i])



#trace de la carte de correlation entre le proxy mensuel et F mensuel, method 
#permet d'enlever pour l'echelle une zone par exemple 6 souther arctic qui a une tres mauvaise correlation  
carte.trace_carte_correlation_mensuelle(base, liste_mask_ecozones, F_ecozones, 
                                        proxy_ecozones, 'F', 'proxy', date2_obs, lo_obs, la_obs,method=5)

proxy_journalier=proxy.proxy.values[:,10,65]
F_journalier=F.F.values[:,10,65]

plt.plot(F_journalier)


plt.plot(proxy_journalier)

import numpy as np
np.corrcoef(F_journalier,proxy_journalier)[0,1]
