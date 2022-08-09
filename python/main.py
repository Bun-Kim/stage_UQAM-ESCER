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
        'methode_spatiale_proxy':['max','sum'],'methode_proxy':'mul','methode_spatiale':['conservatif','conservatif']}

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

#datasets_anomalie=data_processing.anomalies(pc.proxy_calculus(base, temp1), 30)

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

liste_mask_ecozones = data_processing.fct_liste_mask_ecozones(base_Canada,ecozones)

ecozones_name=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
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

'''
cp=data_processing.ouvre_fichier(infile_var_cp)
cape=data_processing.ouvre_fichier(infile_var_cape)
proxy=data_processing.ouvre_fichier(infile_var_proxy)
F=data_processing.ouvre_fichier(infile_var_obs_Canada)
'''

import xarray as xr
cp= xr.DataArray.to_dataset(da_model['cp'],name='cp')
cape= xr.DataArray.to_dataset(da_model['cape'],name='cape')
F=da_obs
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
                                              la_model, 'mean',ecozones_name[i])


#trace de la carte de correlation entre le proxy mensuel et F mensuel, method 
#permet d'enlever pour l'echelle une zone par exemple 6 souther arctic qui a une tres mauvaise correlation  
carte.trace_carte_correlation_mensuelle(base, liste_mask_ecozones, F_ecozones, 
                                        proxy_ecozones, 'F', 'proxy', date2_obs, lo_obs, la_obs,method=5)



#plt.plot(datasets_anomalie[-1].proxy.values[:,10,65])

#np.corrcoef(F_journalier,datasets_anomalie[-1].proxy.values[:,10,65])[0,1]


'''
datasets_anomalie=data_processing.anomalies(proxy, 30)
datasets_anomalie_cape=data_processing.anomalies(cape, 30)


import datetime
import pandas as pd
delta = datetime.timedelta(days = 15)
startday2 = date2_model[0] + delta
endday2 = date2_model[-1]-delta

dates3 = pd.date_range(startday2, endday2, freq='D')
jours=np.any([dates3.day!=29,dates3.month!=2],axis=0)
dates3=dates3[jours]

F_journalier2=F.sel(time=slice(startday2,endday2))
ano_journalieres = datasets_anomalie[-3].sel(time=slice(startday2,endday2))

A = ano_journalieres.proxy.values[:,10,65]


plt.plot(F_journalier2.F.values[:,10,65])
plt.plot(A)


np.corrcoef(F_journalier2.F.values[:,10,65],A)[0,1]

for k in range(100):
    delta_k = datetime.timedelta(days = k+1)
    startday2_k = date2_model[0] + delta_k
    endday2_k = date2_model[-1]-delta_k
    dates3_k = pd.date_range(startday2_k, endday2_k, freq='D')
    jours=np.any([dates3_k.day!=29,dates3_k.month!=2],axis=0)
    dates3_k=dates3_k[jours]
   
    F_journalier2_k=F.sel(time=slice(startday2_k,endday2_k))   
    
    ano_journalieres_k_glissante=  datasets_anomalie[k].sel(time=slice(startday2_k,endday2_k)).proxy.values[:,10,65]
    ano_journalieres_k_glissante [ano_journalieres_k_glissante<0] = 0
    print(np.corrcoef(F_journalier2_k.F.values[:,10,65],ano_journalieres_k_glissante)[0,1])

for k in range(6):
    delta_k = datetime.timedelta(days = k+1)
    startday2_k = date2_model[0] + delta_k
    endday2_k = date2_model[-1]-delta_k
    dates3_k = pd.date_range(startday2_k, endday2_k, freq='D')
    jours=np.any([dates3_k.day!=29,dates3_k.month!=2],axis=0)
    dates3_k=dates3_k[jours]
   
    F_journalier2_k=F.sel(time=slice(startday2_k,endday2_k))   
    
    ano_journalieres_k_glissante=  datasets_anomalie_cape[k].sel(time=slice(startday2_k,endday2_k)).cape.values[:,10,65]
    print(k)
    print(np.corrcoef(F_journalier2_k.F.values[:,10,65],ano_journalieres_k_glissante)[0,1])

plt.plot(datasets_anomalie_cape[6].sel(time=slice(startday2_k,endday2_k)).cape.values[:,10,65]/3)
plt.plot(F_journalier2.F.values[:,10,65])

plt.plot(proxy_journalier)
plt.plot(F_journalier)
'''


proxy_journalier=proxy.proxy.values[:,10,65]
cape_journalier=cape.cape.values[:,10,65]
cp_journalier=cp.cp.values[:,10,65]
F_journalier=F.F.values[:,10,65]


plt.plot(cape_journalier/5,'k',label='cape',alpha=0.8,linewidth=0.8)
plt.plot(cp_journalier*20000,'g',label='cp',alpha=0.8,linewidth=0.8)
plt.plot(F_journalier,label='F',alpha=1,linewidth=0.8)
plt.plot(proxy_journalier,label='proxy',alpha=0.8)
plt.legend()
plt.xlabel('jours')
#plt.title('cp journalier')


import numpy as np

np.corrcoef(F_journalier,proxy_journalier)[0,1]

plt.plot(F_journalier)


### Pour voir les courbes
F_prairies = F_ecozones[9]
cape_prairies =cape_ecozones[9]
cp_prairies= cp_ecozones[9]
proxy_prairies = proxy_ecozones[9]

plt.plot(F_prairies.sum(dim='lat',skipna=True).sum(dim='lon',skipna=True).F.values[:365]/20000,'k-',alpha=0.3)


for longitude in proxy_prairies.lon.values:
    for latitude in proxy_prairies.lat.values:
        plt.plot(proxy_prairies.proxy.sel(lon=longitude).sel(lat=latitude).values[:365])
###    

### correlation
statistiques.correlation_glissante_point(F_prairies.isel(time=slice(0,365)),'F', proxy_prairies.isel(time=slice(0,365)), 'proxy', date2_model, -105.5, 50.5,30)

### correlations avec des membres pour la prairie
membres_prairies_proxy = statistiques.correlation_glissante_membres(F_prairies, 'F', proxy_prairies, 'proxy', date2_model, lo_model, la_model, 30)
statistiques.trace_membre(membres_prairies_proxy, 'prairies')

'''
membres_prairies_cape = statistiques.correlation_glissante_membres(F_prairies, 'F', cape_prairies, 'cape', date2_model, lo_model, la_model, 30)
statistiques.trace_membre(membres_prairies_cape, 'prairies')

membres_prairies_cp = statistiques.correlation_glissante_membres(F_prairies, 'F', cp_prairies, 'cp', date2_model, lo_model, la_model, 30)
statistiques.trace_membre(membres_prairies_cp, 'prairies')
'''

# =============================================================================
# correlations avec des membres pour chaque zone (tres long)
# =============================================================================
statistiques.correlations_glissante_membres_ecozones(F_ecozones, 'F', proxy_ecozones, 'proxy', date2_model, lo_model, la_model, 30)

# =============================================================================
# comparaison fenetre
# =============================================================================
comparaison = []
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', cp_prairies, 'cp', date2_model, lo_model, la_model, 10))
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', cp_prairies, 'cp', date2_model, lo_model, la_model, 15))
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', cp_prairies, 'cp', date2_model, lo_model, la_model, 30))

statistiques.trace_membre_comparaison(comparaison ,[10,15,30],'prairies')

# =============================================================================
# comparaison correlation/ occurrence eclairs
# =============================================================================
statistiques.trace_membre_avec_F(membres_prairies_proxy, F_prairies, 'prairies')

# =============================================================================
# correlation de Pearson sur 2 ans sur chaque ecozone
# =============================================================================
xr.DataArray.to_dataset(xr.corr(proxy.proxy, F.F, dim="time"),name='correlation')
correlations_ecozones = data_processing.champs_ecozone(xr.DataArray.to_dataset(xr.corr(proxy.proxy, F.F, dim="time"),name='correlation')
                                                       , liste_mask_ecozones,'correlation')

# =============================================================================
# carte de correlation mensuelles par point
# =============================================================================

for an in np.arange(2018,2020):
    for mois in liste_mois:
        corr = xr.DataArray.to_dataset(statistiques.correlation_mensuelle_par_point(da_obs, 'F', proxy, 'proxy', an, mois), name='correlation')
        carte.tracer(base_Canada, corr,'correlation',mois + ' ' + str(an) + ' ')
    
    
    
    
    
    