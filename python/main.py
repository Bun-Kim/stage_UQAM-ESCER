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
        'methode_spatiale_proxy':['mean','sum'],'methode_proxy':'mul','methode_spatiale':['conservatif','conservatif']}


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

proxy= pc.proxy_calculus(base, da_model)

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

mask_ecozones =ecozones.sel(lat=slice(40.5,69.5)).sel(lon=slice(-149.5,-50.5))       
mask_canada = xr.DataArray(mask_ecozones.ecozones.values * 0 + 1 )  


mask_canada=xr.DataArray.to_dataset(mask_canada, name='region')
mask_canada= mask_canada.rename({'dim_0': 'lat','dim_1': 'lon'})
mask_canada = mask_canada.assign_coords({'lat' : mask_ecozones.lat.values, 'lon' : mask_ecozones.lon.values})

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
carte.tracer_moyenne(base, data_processing.mask_canada(proxy,mask_canada), list(proxy.keys())[0])
carte.tracer_moyenne(base, data_processing.mask_canada(da_obs/(100*100),mask_canada),'F')

carte.tracer_moyenne(base, cp, 'cp')
carte.tracer_moyenne(base, cape,'cape')
carte.tracer_moyenne(base, da_obs_Canada/(100*100),'F')


#############3
for i in range(0,18):
    #carte.tracer_moyenne(base_Canada,da_obs['F'] * liste_mask_ecozones[i] ,'ecozone')
    carte.tracer_moyenne_echelle(base_Canada,da_obs['F'] * liste_mask_ecozones[i], da_obs,'ecozone','F',)
   


cape_ecozones=data_processing.champs_ecozone(cape, liste_mask_ecozones, 'cape')
cp_ecozones=data_processing.champs_ecozone(cp, liste_mask_ecozones, 'cp')
proxy_ecozones=data_processing.champs_ecozone(proxy, liste_mask_ecozones, list(proxy.keys())[0])
F_ecozones=data_processing.champs_ecozone(F, liste_mask_ecozones, 'F')


for i in range(0,18):
    carte.trace_occurences_mensuelles(base_Canada,base,da_obs,F_ecozones[i],
                                              [cape_ecozones[i],cp_ecozones[i],proxy_ecozones[i]],
                                              ['F'], ['cape','cp','proxy'], date2_model, lo_model, 
                                              la_model, 'mean',ecozones_name[i])


#trace de la carte de correlation entre le proxy mensuel et F mensuel, method 
#permet d'enlever pour l'echelle une zone par exemple 6 souther arctic qui a une tres mauvaise correlation  
carte.trace_carte_correlation_mensuelle(base, liste_mask_ecozones, F_ecozones, 
                                        proxy_ecozones, 'F', list(proxy.keys())[0], date2_obs, lo_obs, la_obs,method=5)



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

A=statistiques.data_chaque_mois(da_obs,'F', 'Juin', 2018, da_model, lo_model, la_model)
B = statistiques.data_chaque_mois(proxy,list(proxy.keys())[0], 'Juin', 2018, da_model, lo_model, la_model)
carte.tracer_moyenne(base_Canada, A, list(A.keys())[0])
carte.tracer_moyenne(base_Canada, B, list(B.keys())[0])


for an in np.arange(2018,2020):
    for mois in liste_mois:
        F_mensuel = statistiques.data_chaque_mois(da_obs,'F', mois, an, da_model, lo_model, la_model)
        proxy_mensuel = statistiques.data_chaque_mois(proxy,list(proxy.keys())[0], mois, an, da_model, lo_model, la_model)
        corr = xr.DataArray.to_dataset(statistiques.correlation_mensuelle_par_point(da_obs, 'F', proxy, list(proxy.keys())[0], an, mois), name='correlation')
        
        carte.tracer_moyenne(base_Canada, F_mensuel, list(F_mensuel.keys())[0])
        carte.tracer_moyenne(base_Canada, proxy_mensuel, list(proxy_mensuel.keys())[0])
        carte.tracer(base_Canada, corr,'correlation',mois + ' ' + str(an) + ' ')
        
        #carte.tracer_moyenne(base_Canada, data_processing.mask_canada(F_mensuel,mask_canada), list(F_mensuel.keys())[0])
        #carte.tracer_moyenne(base_Canada, data_processing.mask_canada(proxy_mensuel,mask_canada), list(proxy_mensuel.keys())[0])
        #carte.tracer(base_Canada, data_processing.mask_canada(corr,mask_canada),'correlation',mois + ' ' + str(an) + ' ')
        #carte.trace_comparaison_correlation_F(base_Canada, da_obs, 'F', corr,mois + ' ' + str(an) + ' ')
# =============================================================================
# LI
# =============================================================================
LI_vars = data_processing.ouvre_fichier(b) 
    
    
    
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
import metpy
from metpy.plots import SkewT
from metpy.units import units

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

# Create arrays of pressure, temperature, dewpoint, and wind components
p = [902, 501] * units.hPa
t = [-3, -21.9] * units.degC
td = [-22, -34] * units.degC

# Calculate parcel profile
prof = metpy.calc.parcel_profile(p, temperature, dewpoint)
all_prof = np.zeros(np.shape(LI_vars.d2m.values))

skew.plot(p, t, 'r')
skew.plot(p, td, 'g')
skew.plot(p, prof, 'k')  # Plot parcel profile


skew.ax.set_xlim(-50, 15)
skew.ax.set_ylim(1000, 100)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()



t2m = LI_vars.isel(time = 1000).isel(lat=133).isel(lon = 30).t2m.values
d2m = LI_vars.isel(time = 1000).isel(lat=133).isel(lon = 30).d2m.values

parcel_profile = metpy.calc.parcel_profile(p, t[0], td[0])
LI = metpy.calc.lifted_index(p,t, parcel_profile)


# =============================================================================
# cape et proxy
# =============================================================================
q_t = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/q_t_0708-2019.nc')
q_t = q_t.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))
q_t = q_t.reindex(level=list(reversed(q_t.level)))

t2m_d2m = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/t2m_d2m_era5_2018-2019.nc')
t2m_d2m = t2m_d2m.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

pressure = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/sp_era5_2018-2019.nc')
pressure = pressure.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

q = q_t['q'].values
t = q_t['t'].values  * units.K
sp = pressure['sp'].sel(time = slice('2019-07-01T00:00:00','2019-08-31T23:00:00')).sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE'])).values * units.Pa




periodes,niveaux,latitudes,longitudes = np.shape(t)

'''
for pas in range(periodes):
    print(pas)
    for level in range(niveaux): 
        for lat in range(latitudes):
            for lon in range(longitudes):
                q_k = q[pas,level,lat,lon]
                t_k = t[pas,level,lat,lon]
                sp_k = sp[pas,lat,lon]
                
                dp = metpy.calc.dewpoint_from_specific_humidity(sp_k, t_k, q_k)
                
             
                        
                all_dp[pas,level,lat,lon]=np.array(dp)
'''                
all_dp = []
for level in range(niveaux): 
    print('level' + str(level))
    
    dp = metpy.calc.dewpoint_from_specific_humidity(sp, t[:,0,:,:], 1000*q[:,0,:,:])
    all_dp.append(np.array(dp) + 273.15)
             
dp = xr.DataArray.to_dataset(xr.DataArray(all_dp),name='dp')
dp = dp.rename({'dim_0': 'level','dim_1': 'time','dim_2': 'lat','dim_3': 'lon'})
dp = dp.assign_coords({'lat' : q_t.lat.values,'lon' : q_t.lon.values,'time' : q_t.time.values,'level' : q_t.level.values})
dp = dp.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))
             
#calcul lfc

levels = q_t.level.values* units.Pa
dewpoint = dp.dp.values * units.K

levels_all = []
for k in range(3000):
    
    levels_all.append(q_t.level.values)
levels_all = np.reshape(levels_all,(23,30,100))
levels_all=-np.sort(-levels_all,axis=0)
levels_all = levels_all * units.Pa

lfc= []


'''
for pas in range(periodes):
    print(pas)
    metpy.calc.lfc(levels_all, t[pas,:,:,:], dewpoint[:,pas,:,:], parcel_temperature_profile=None, dewpoint_start=None, which='top')
    #lfc.append(metpy.calc.lfc(levels_all, t[pas,:,:,:], dewpoint[:,pas,:,:], parcel_temperature_profile=None, dewpoint_start=None, which='top'))
'''

for pas in range(periodes):
    print(pas)
    for lat in range(latitudes):
        for lon in range(longitudes):
                metpy.calc.el(levels, t[pas,:,lat,lon],dewpoint[:,pas,lat,lon],
                                          parcel_temperature_profile=None, which='top')
                
             
                lfc.append(metpy.calc.lfc(levels, t[pas,:,lat,lon],dewpoint[:,pas,lat,lon],
                                          parcel_temperature_profile=None, dewpoint_start=None, which='top'))


A = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
        775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
        350.,  300.,  250.,  225.,  200.] * units.hPa

B= list(reversed([214.65959, 218.65106, 224.071  , 233.07158, 241.4688 , 248.98245,
       253.22833, 258.705  , 264.039  , 268.9587 , 273.21835, 276.38126,
       278.87378, 279.0285 , 281.0672 , 283.27264, 284.97958, 285.1893 ,
       284.9882 , 286.0213 , 286.745  , 287.32773, 288.62555]))* units.K

C= list(reversed([304.1265, 304.1265, 304.1265, 304.1265, 304.1265, 304.1265,
       304.1265, 304.1265, 304.1265, 304.1265, 304.1265, 304.1265,
       304.1265, 304.1265, 304.1265, 304.1265, 304.1265, 304.1265,
       304.1265, 304.1265, 304.1265, 304.1265, 304.1265])) * units.K


metpy.calc.lfc([1000]* units.hPa, [288.62555]* units.K, [304.1265]* units.K, parcel_temperature_profile=None, dewpoint_start=None, which='top')

#metpy.calc.lfc(levels, , C, parcel_temperature_profile=None, dewpoint_start=None, which='top')

metpy.calc.el(A, B, C, parcel_temperature_profile=None, which='top')



lfc = metpy.calc.lfc(pressure.sp.values *  units.hPa, t2m_d2m.t2m.values *units.K, t2m_d2m.d2m.values *units.K, parcel_temperature_profile=None, dewpoint_start=None, which='top')






































lfc = metpy.calc.lfc(sp, t2m, d2m, parcel_temperature_profile=None, dewpoint_start=None, which='top')
el = metpy.calc.el(sp[0,0,0], t2m[0,0,0], d2m[0,0,0], parcel_temperature_profile=None, which='top')

lfc = xr.DataArray.to_dataset(xr.DataArray(lfc),name='lfc').rename({'dim_0': 'time','dim_1': 'lat','dim_2': 'lon'}).assign_coords({'lat' : data_model.lat.values,
                                                                                                                                   'lon' : data_model.lon.values,'time' : data_model.time.values})


# =============================================================================
# proxy cape cp horaire
# =============================================================================
cape_cp = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/cape_cp_era5_2018-2020.nc')
cape_cp = cape_cp.isel(time=slice(0,17520)).sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

proxy = cape_cp.cape * cape_cp.cp
proxy_horaire = xr.DataArray.to_dataset(proxy,name='capecp')
#data_processing.sauvegarde_donnees(proxy, list(proxy.lat.values), list(proxy.lon.values), list(proxy.time.values),list(proxy.keys())[0])

_,_,_,proxy,_ = data_processing.selectionDonnees(base,proxy_horaire)

# =============================================================================
# proxy cin
# =============================================================================
cin = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/cin_era5_2018-2019.nc')
cin = -cin.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

cin = cin.where(cin>-200)

base_cin = {'method_obs' : 'max','frequence':'1D'}

cin_journaliere = ajustement.resolution_temporelle_obs(base_cin,cin)

proxy_cin_horaire = proxy_horaire.capecp * (1 + cin.cin )

proxy_cin = ajustement.resolution_temporelle_obs(base_cin,proxy_cin_horaire)
proxy_cin = xr.DataArray.to_dataset(proxy_cin,name='capecpexp(-cin)')
#data_processing.sauvegarde_donnees(inv_exp_cin, list(inv_exp_cin.lat.values), list(inv_exp_cin.lon.values), list(inv_exp_cin.time.values),'inv_exp_cin')


# =============================================================================
# proxy divergence
# =============================================================================
'''
cape_cp = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/cape_cp_era5_2018-2020.nc')
cape_cp = cape_cp.isel(time=slice(0,17520)).sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

proxy = cape_cp.cape * cape_cp.cp
#proxy = xr.DataArray.to_dataset(proxy,name='capecp')
'''
#negatif pour une convergence
d = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/d_era5_2018-2019.nc')
#d= d.rename({'longitude' : 'lon','latitude' : 'lat'})
#d.reindex(lat=sorted(d.lat.values))
d= d.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))


base_d = {'method_obs' : 'min','frequence':'1D'}
d_journalier=ajustement.resolution_temporelle_obs(base_d,d)

#data_processing.sauvegarde_donnees(inv_exp_cin, list(inv_exp_cin.lat.values), list(inv_exp_cin.lon.values), list(inv_exp_cin.time.values),'inv_exp_cin')

proxy_d_horaire = xr.DataArray.to_dataset(-1 * proxy_horaire.capecp * d.d,name = 'capecpd')

base_proxy_d = {'methode_temporelle_proxy' : ['max'],'frequence':'1D','proxy':['capecpd']}
proxy_d=ajustement.resolution_temporelle_model(base_proxy_d,proxy_d_horaire)

base_proxy_d = {'methode_temporelle_proxy' : ['sum'],'frequence':'1D','proxy':['capecpd']}
proxy_dsum=ajustement.resolution_temporelle_model(base_proxy_d,proxy_d_horaire)
proxy_dsum.rename({'capecpd':'capecpdsum'})

#proxy_d = xr.DataArray.to_dataset(proxy_d, name = 'capecpd')

# =============================================================================
# cin divergence
# =============================================================================

cin_d =  d.where(d<0).d *cin.cin
cin_d = xr.DataArray.to_dataset(cin_d, name = 'cin_d')

base_cin_d = {'method_obs' : 'min','frequence':'1D'}
cin_d_journalier=ajustement.resolution_temporelle_obs(base_cin_d,cin_d)

# =============================================================================
# proxy vent
# =============================================================================

uv = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/u_v_700_era5_2018-2019.nc')
uv= uv.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))

u = xr.DataArray.to_dataset(uv.u, name = 'u')
v = xr.DataArray.to_dataset(uv.v, name = 'v')

base_u = {'method_obs' : 'mean','frequence':'1D'}
u_journalier=ajustement.resolution_temporelle_obs(base_u,u)
v_journalier=ajustement.resolution_temporelle_obs(base_u,v)

uv_journalier = u_journalier
uv_journalier['v'] = v_journalier.v

u_journalier=ajustement.resolution_temporelle_obs(base_u,u)



# =============================================================================
# proxy temperature
# =============================================================================
T850 = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/T850_era5_2018-2019.nc')
T500 = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/T500_era5_2018-2019.nc')


# =============================================================================
# proxy div flux d'humidité
# =============================================================================
all = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/all_era5_2018-2019.nc')

div_flux_humid = all['p84.162']
div_flux_humid = xr.DataArray.to_dataset(div_flux_humid, name = 'div_flux_humid')

base_flux_humid = {'method_obs' : 'mean','frequence':'1D'}
div_flux_humid_journalier=ajustement.resolution_temporelle_obs(base_flux_humid,div_flux_humid)






# =============================================================================
# etude par jour
# =============================================================================

carte.trace_comparaison(base_Canada, [da_obs.isel(time=150),proxy.isel(time=150),proxy_d.isel(time=150),da_obs.isel(time=150)])


carte.tracer(base_Canada, da_obs.isel(time=187), 'F','F')
carte.tracer(base_Canada, proxy.isel(time=187), list(proxy.keys())[0],list(proxy.keys())[0])
carte.tracer(base_Canada, proxy_d.isel(time=187), list(proxy_d.keys())[0],list(proxy_d.keys())[0])
carte.tracer(base_Canada, proxy_cin.isel(time=187), list(proxy_cin.keys())[0],list(proxy_cin.keys())[0])

carte.tracer(base_Canada, cin_journaliere.isel(time=187), list(cin_journaliere.keys())[0],list(cin_journaliere.keys())[0])


#carte avec vent 700
carte.tracer_avec_vent(base_Canada, da_obs.sel(time='2018-07-09'), 'F',uv_journalier.sel(time = '2018-07-09'),'F')
carte.tracer_avec_vent(base_Canada, proxy_d.sel(time='2018-07-09'), list(proxy_d.keys())[0],uv_journalier.sel(time = '2018-07-09'),'F')
carte.tracer_avec_vent(base_Canada, proxy.sel(time='2018-07-09'), list(proxy.keys())[0],uv_journalier.sel(time = '2018-07-09'),'F')


carte.tracer_avec_vent(base_Canada, da_obs.isel(time=188), 'F',uv.sel(time = '2018-07-08T00:00:00'),'F')
carte.tracer_avec_vent(base_Canada, proxy_d.isel(time=188), list(proxy_d.keys())[0],uv.sel(time = '2018-07-08T00:00:00'),'F')
carte.tracer_avec_vent(base_Canada, proxy.isel(time=188), list(proxy.keys())[0],uv.sel(time = '2018-07-08T00:00:00'),'F')



#cin min
carte.tracer(base_Canada, xr.ufuncs.exp(cin_journaliere.sel(time='2018-07-09')), list(cin_journaliere.keys())[0],list(cin_journaliere.keys())[0])

#d min
carte.tracer(base_Canada, d_journalier.isel(time=187), list(d_journalier.keys())[0],list(d_journalier.keys())[0])
#cin et d
carte.tracer(base_Canada, cin_d_journalier.isel(time=187), list(cin_d_journalier.keys())[0],list(cin_d_journalier.keys())[0])


#par heure
jour = 4488
jour_bis = 144
jourh =  '2018-07-07T00:00:00'

for k in range(48) :
    #carte.tracer(base_Canada, cin.isel(time=4464+k), list(cin.keys())[0],str(cin.isel(time=4464+k).time.values)[:13])
    carte.tracer(base_Canada, div_flux_humid.isel(time=4464+k), list(div_flux_humid.keys())[0],str(div_flux_humid.isel(time=4464+k).time.values)[:13])
    
    
    
for k in range(48) :
    carte.tracer_avec_vent(base_Canada,da_obs.isel(time=jour+k), list(da_obs.keys())[0],uv.isel(time = jour_bis + k),str(uv.isel(time = jour_bis + k).time.values)[:13])
    carte.tracer_avec_vent(base_Canada, proxy_d_horaire.isel(time=jour+k), list(proxy_d.keys())[0],uv.isel(time = jour_bis + k),str(uv.isel(time = jour_bis + k).time.values)[:13])

#par jour

for k in range(31):
    carte.tracer_avec_vent(base_Canada,da_obs.isel(time=181+k), list(da_obs.keys())[0],uv_journalier.isel(time =  k),str(uv_journalier.isel(time =  k).time.values)[:13])

    carte.tracer_avec_vent(base_Canada, proxy.isel(time=181+k), list(proxy.keys())[0],uv_journalier.isel(time =  k),str(uv_journalier.isel(time =  k).time.values)[:13])
    carte.tracer_avec_vent(base_Canada, proxy_d.isel(time=181+k), list(proxy_d.keys())[0],uv_journalier.isel(time =  k),str(uv_journalier.isel(time =  k).time.values)[:13])

for k in range(31):
    carte.tracer_avec_vent(base_Canada,da_obs.isel(time=181+k), list(da_obs.keys())[0],uv_journalier.isel(time =  k),str(uv_journalier.isel(time =  k).time.values)[:13])
    
for k in range(48) :
    carte.trace_comparaison(base_Canada,[da_obs.isel(time=181+k),proxy.isel(time=181+k),proxy_d.isel(time=181+k),proxy_dsum.isel(time=181+k)],str(proxy_d.time.values[181+k])[:10])






#vent
import cartopy.crs as ccrs
from cartopy.examples.arrows import sample_data
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, 75, 10, 60])
ax.stock_img()
ax.coastlines()

x, y, u, v, vector_crs = sample_data(shape=(10, 14))
ax.barbs(x, y, u, v, length=5,
sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
linewidth=0.95, transform=vector_crs)


















# =============================================================================
# carte de correlation mensuelles par point
# =============================================================================

proxy_new = proxy_d

for an in np.arange(2018,2020):
    for mois in liste_mois:
        F_mensuel = statistiques.data_chaque_mois(da_obs,'F', mois, an, da_model, lo_model, la_model)
        proxy_mensuel_new = statistiques.data_chaque_mois(proxy_new,list(proxy_new.keys())[0], mois, an, da_model, lo_model, la_model)
        proxy_mensuel = statistiques.data_chaque_mois(proxy,list(proxy.keys())[0], mois, an, da_model, lo_model, la_model)
        corr = xr.DataArray.to_dataset(statistiques.correlation_mensuelle_par_point(da_obs, 'F', proxy_new, list(proxy_new.keys())[0], an, mois), name='correlation new')
        
        carte.trace_comparaison_moyenne(base_Canada, [F_mensuel,corr,proxy_mensuel,proxy_mensuel_new])
        
        #carte.tracer_moyenne(base_Canada, F_mensuel, list(F_mensuel.keys())[0])
        #carte.tracer_moyenne(base_Canada, proxy_mensuel, list(proxy_mensuel.keys())[0])
        #carte.tracer(base_Canada, corr,'correlation',mois + ' ' + str(an) + ' ')
        
        #carte.tracer_moyenne(base_Canada, data_processing.mask_canada(F_mensuel,mask_canada), list(F_mensuel.keys())[0])
        #carte.tracer_moyenne(base_Canada, data_processing.mask_canada(proxy_mensuel,mask_canada), list(proxy_mensuel.keys())[0])
        #carte.tracer(base_Canada, data_processing.mask_canada(corr,mask_canada),'correlation',mois + ' ' + str(an) + ' ')
        #carte.trace_comparaison_correlation_F(base_Canada, da_obs, 'F', corr,mois + ' ' + str(an) + ' ')

# =============================================================================
# 
# =============================================================================



r_t = data_processing.ouvre_fichier('/bwk01_01/san/stage_UQAM-ESCER/data/model/q_t_0708-2019.nc')
r_t= r_t.sel(lat=slice(base_Canada['latS'],base_Canada['latN'])).sel(lon=slice(base_Canada['lonW'],base_Canada['lonE']))
r_t = r_t.sortby('level',ascending = False)



import metpy
from metpy.units import units
import metpy.calc as mpcalc


pressure = r_t.level.values * units.hPa
t = r_t.t.values * units.K
r =  r_t.r.values/100 


periode,levels,latitudes,longitudes = np.shape(t)


import xarray as xr
xrdp = xr.DataArray(dp,name='dp')

xrdp = xrdp.rename({'dim_0':'time','dim_1':'level','dim_2':'lat','dim_3':'lon'})
xrdp=xrdp.assign_coords({'time' : r_t.time.values, 'level' : r_t.level.values,'lat' : r_t.lat.values,'lon' : r_t.lon.values})

xrdp = xr.DataArray.to_dataset(xrdp)
periode,levels,latitudes,longitudes = np.shape(t)





periode = 300

lfc=np.zeros((periode,latitudes,longitudes))
lfc_T=np.zeros((periode,latitudes,longitudes))
el=np.zeros((periode,latitudes,longitudes))
el_T=np.zeros((periode,latitudes,longitudes))
for pas in [20]:
    print(pas)
    for lat in range(latitudes):
            for lon in range(longitudes):
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
                    
                    lfc[pas,lat,lon] = np.nan
                    lfc_T[pas,lat,lon] =np.nan
                    el[pas,lat,lon] =np.nan
                    el_T[pas,lat,lon] =np.nan
                    






