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
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
'''
base_Canada = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2020,
        'frequence':'1D', 'resolution':1, 
        'proxy':['cape','cp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max','sum'],
        'methode_spatiale_proxy':['mean','sum'],'methode_proxy':'mul','methode_spatiale':['conservatif','conservatif']}
'''
'''
base_Canada = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2019,
        'frequence':'1D', 'resolution':1, 
        'proxy':['t2m','d2m'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max'],'methode_proxy':'LI','methode_spatiale':['conservatif']}
'''

base_Canada = {'anneeDebut':2018, 'anneeFin':2019,'anneeDebut_model':2018,
        'anneeFin_model':2019,
        'frequence':'1D', 'resolution':1, 
        'proxy':['capecp'], 'model_proxy':'Era5', 'model_obs':'WWLLN',
        'latS':40.5,
        'latN':69.5,'lonW':-149.5,'lonE':-50.5,'method_obs':'max',
        'methode_temporelle_proxy':['max'],'methode_proxy':'capecp','methode_spatiale':['conservatif']}

ecozones=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
          'Southern_Arctic','Boreal_Plains','Montane_Cordillera','Prairies','Pacific_Maritime','Arctic_Cordillera',
          'Taiga_Shield_E','Atlantic_Maritime','Boreal_Shield_E',
          'Mixedwood_Plains','Hudson_Plains','Boreal_Shield_S']
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


temp = data_processing.changement_nom_coordonnees_obs(base, temp)

print('ajustement des obs')
print('ajustement temporel')
temp=ajustement.resolution_temporelle_obs(base, temp)
print('ajustement spatial')

[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,temp)

print('ajustement des donnees pour le proxy')
print('ajustement temporel')
temp1=ajustement.resolution_temporelle_model(base, temp1)

[t_model,la_model,lo_model,da_model,date2_model]=data_processing.selectionDonnees(base,temp1)

proxy= pc.proxy_calculus(base, da_model)



print('sauvegarde des donnees en fichier netcdf')
data_processing.sauvegarde_donnees(proxy, list(proxy.lat.values), list(proxy.lon.values), list(proxy.time.values),list(proxy.keys())[0])

data_processing.sauvegarde_donnees(da_obs, la_obs, lo_obs, t_obs, "F")
'''

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

mask_ecozones =ecozones.sel(lat=slice(40.5,69.5)).sel(lon=slice(-149.5,-50.5))       
mask_canada = xr.DataArray(mask_ecozones.ecozones.values * 0 + 1 )  


mask_canada=xr.DataArray.to_dataset(mask_canada, name='region')
mask_canada= mask_canada.rename({'dim_0': 'lat','dim_1': 'lon'})
mask_canada = mask_canada.assign_coords({'lat' : mask_ecozones.lat.values, 'lon' : mask_ecozones.lon.values})


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
if list(proxy.keys())[0] == 'proxy':
    proxy_vars = []
    for k in range(len(base_Canada['proxy'])):
        proxy_vars.append(xr.DataArray.to_dataset(da_model[base_Canada['proxy'][k]],name=base_Canada['proxy'][k]))

[t_obs,la_obs,lo_obs,da_obs,date2_obs]=data_processing.selectionDonnees(base,da_obs)
[t_model,la_model,lo_model,proxy,date2_model]=data_processing.selectionDonnees(base,proxy)

for var in proxy_vars :
    _,_,_,proxy_vars[k],_= data_processing.selectionDonnees(base,proxy_vars[k])

print('trace carte')

'''
# =============================================================================
# totalx >55 --> tempete
# =============================================================================
if list(proxy.keys())[0] == 'totalx':
    proxy = proxy.where(proxy.totalx>55)
'''
if list(proxy.keys())[0] == 'LI':
    proxy = proxy.where(proxy.LI<0)
# =============================================================================
# carte
# =============================================================================

carte.tracer_moyenne(base, data_processing.mask_canada(proxy,mask_canada), list(proxy.keys())[0])
carte.tracer_moyenne(base, data_processing.mask_canada(da_obs/(100*100),mask_canada),'F')




for k in range(len(base_Canada['proxy'])):
    carte.tracer_moyenne(base, proxy_vars[k],base_Canada['proxy'][k])

# =============================================================================
# exemple d'utilisation des ecozones
# =============================================================================
'''
for i in range(0,18):
    #carte.tracer_moyenne(base_Canada,da_obs['F'] * liste_mask_ecozones[i] ,'ecozone')
    carte.tracer_moyenne_echelle(base_Canada,da_obs['F'] * liste_mask_ecozones[i], da_obs,'ecozone','F',)
'''   
# =============================================================================
# calcul des variables pour chaque ecozone
# =============================================================================
proxy_vars_ecozones = []
for k in range(len(base_Canada['proxy'])):
    proxy_vars_ecozones.append(data_processing.champs_ecozone(proxy_vars[k],liste_mask_ecozones, base_Canada['proxy'][k]))
proxy_ecozones=data_processing.champs_ecozone(proxy, liste_mask_ecozones, list(proxy.keys())[0])
F_ecozones=data_processing.champs_ecozone(da_obs, liste_mask_ecozones, 'F')



    
if list(proxy.keys())[0] == 'cape':
    for i in range(0,18):
        carte.trace_occurences_mensuelles(base_Canada,base,da_obs,F_ecozones[i],
                                                      [proxy_vars_ecozones[0][i],proxy_vars_ecozones[1][i],proxy_ecozones[i]],
                                                      ['F'], [base_Canada['proxy'][0],base_Canada['proxy'][1],'proxy'], date2_model, lo_model, 
                                                      la_model, 'mean',ecozones_name[i])

for i in range(0,18):
    carte.trace_occurences_mensuelles(base_Canada,base,da_obs,F_ecozones[i],
                                                  [proxy_ecozones[i],proxy_ecozones[i],proxy_ecozones[i]],
                                                  ['F'], [list(proxy_ecozones[0].keys())[0],list(proxy_ecozones[0].keys())[0],'proxy'], date2_model, lo_model, 
                                                  la_model, 'mean',ecozones_name[i])


#trace de la carte de correlation entre le proxy mensuel et F mensuel, method 
#permet d'enlever pour l'echelle une zone par exemple 6 souther arctic qui a une tres mauvaise correlation  
carte.trace_carte_correlation_mensuelle(base, liste_mask_ecozones, F_ecozones, 
                                        proxy_ecozones, 'F', list(proxy.keys())[0], date2_obs, lo_obs, la_obs,method=5)



#plt.plot(datasets_anomalie[-1].proxy.values[:,10,65])

#np.corrcoef(F_journalier,datasets_anomalie[-1].proxy.values[:,10,65])[0,1]




# =============================================================================
# affichage des occurences de proxy et de foudre au point lat 50.5 lon -54.5 dans les prairies
# =============================================================================

proxy_journalier=proxy[list(proxy.keys())[0]].values[:,10,65]
F_journalier=da_obs.F.values[:,10,65]


plt.plot(F_journalier/np.nanmax(F_journalier),label='F',alpha=1,linewidth=0.8)
plt.plot(proxy_journalier/np.nanmax(proxy_journalier),label=list(proxy.keys())[0],alpha=0.8)
plt.legend()
plt.xlabel('jours')
plt.title('occurence F et proxy jounrnaliers normalises')
plt.grid()
plt.show()


# =============================================================================
# sur  les prairies
# =============================================================================
F_prairies = F_ecozones[9]
proxy_prairies = proxy_ecozones[9]


### correlations avec des membres pour la prairie
membres_prairies_proxy = statistiques.correlation_glissante_membres(F_prairies, 'F', proxy_prairies, list(proxy.keys())[0], date2_model, lo_model, la_model, 30)
statistiques.trace_membre_avec_F(membres_prairies_proxy, F_prairies, 'prairies')

# =============================================================================
# correlations avec des membres pour chaque zone (tres long)
# =============================================================================
statistiques.correlations_glissante_membres_ecozones(F_ecozones, 'F', proxy_ecozones, 'proxy', date2_model, lo_model, la_model, 30)

# =============================================================================
# comparaison fenetre
# =============================================================================
comparaison = []
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', proxy_prairies, 'proxy', date2_model, lo_model, la_model, 10))
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', proxy_prairies, 'proxy', date2_model, lo_model, la_model, 15))
comparaison.append(statistiques.correlation_glissante_membres(F_prairies, 'F', proxy_prairies, 'proxy', date2_model, lo_model, la_model, 30))

statistiques.trace_membre_comparaison(comparaison, [10,15,30],'prairies')


# =============================================================================
# correlation de Pearson sur 2 ans sur chaque ecozone
# =============================================================================

correlations_ecozones = data_processing.champs_ecozone(xr.DataArray.to_dataset(xr.corr(proxy.proxy, da_obs.F, dim="time"),
                                                                               name='correlation')
                                                       , liste_mask_ecozones,'correlation')

# =============================================================================
# carte de correlation mensuelles par point
# =============================================================================

for an in np.arange(2018,2020):
    for mois in liste_mois:
        corr = xr.DataArray.to_dataset(statistiques.correlation_mensuelle_par_point(da_obs, 'F', proxy, list(proxy.keys())[0] , an, mois), name='correlation')
        carte.tracer(base_Canada, corr,'correlation',mois + ' ' + str(an) + ' ')
    














