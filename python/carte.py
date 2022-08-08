#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:34:47 2022

@author: san
"""
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm

import statistiques

method_list = [
    "bilinear",
    "nearest_s2d",
    "nearest_d2s",
    "patch",
]
liste_mois =  ['J', 'F', 'M', 'A', 'M' , 'J',
             'J', 'A', 'S' , 'O','N',
             'D']
ecozones=['Boreal_Cordillera','Taiga_Shield_W','Boreal_Shield_W','Northern_Arctic','Taiga_Cordillera','Taiga_Plains',
          'Southern_Arctic','Boreal_Plains','Montane_Cordillera','Prairies','Pacific_Maritime','Arctic_Cordillera',
          'Taiga_Shield_E','Atlantic_Maritime','Boreal_Shield_E',
          'Mixedwood_Plains','Hudson_Plains','Boreal_Shield_S']

def bounds(dico):
    return [dico['lonW'], dico['lonE'], dico['latS'], dico['latN']]

def tracer_moyenne(dico,data,champs):
    resolution= abs(data.lat.values[0]-data.lat.values[1])
    
    #cmap = mpl.cm.jet
    #norm = BoundaryNorm(clevs, cmap.N, extend='both')
    
    fig = plt.figure(figsize=(28,16), frameon=True)  
    ############################
    ##### main image
    
    ax1 = plt.axes( projection=ccrs.Mercator(central_longitude=-80))
    ax1.set_title(champs+ '  resolution ' + str(resolution) +'°' +dico['methode_spatiale'][0],loc='center')
    #ax1 = fig.add_subplot( projection=ccrs.LambertConformal())
    ax1.set_extent(bounds(dico), crs=ccrs.PlateCarree())
    ax1.coastlines(resolution='auto', color='k')
    ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
    ax1.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
    ax1.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
    
    #ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted')    # couche frontieres
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières 
    #coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
    #                            facecolor='none', name='coastline')
    #ax.add_feature(coast, edgecolor='black')   
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.add_feature(cfeature.BORDERS)
        # Put a background image on for nice sea rendering.
    ax1.stock_img()
    
    datamean=data[champs].mean(dim='time').values
    #datamean = np.ma.masked_where(datamean <= 1.588127e-04, datamean)
    levels = np.linspace(0, np.nanmean(data[champs].values)*50, 11)
    datamean[abs(datamean)<0.00005]=np.nan
    mm = ax1.contourf(data.lon.values,\
      data.lat.values,\
      datamean,\
    #  vmin=0.,\
    #  vmax=40., \
      transform=ccrs.PlateCarree(),\
      levels=levels,\
      cmap="jet",\
      extend='max',
      alpha=0.7)
    
        
    #ax1.plot([-90,-56],[60,60],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-56],[41,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-90],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-56,-56],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    
    cb1 = plt.colorbar(mm, orientation='horizontal',shrink=0.5, pad=0.1)
    #cb1.set_label('CAPExCP proxy', size='x-large')
   
    #plt.savefig('./figure/capecp_'+str(year)+'.png', bbox_inches='tight', pad_inches=0.1)  
    plt.show()

def tracer_moyenne_echelle(dico,data,data_scale,champs,champs_scale):
    resolution= abs(data.lat.values[0]-data.lat.values[1])
    
    #cmap = mpl.cm.jet
    #norm = BoundaryNorm(clevs, cmap.N, extend='both')
    
    fig = plt.figure(figsize=(28,16), frameon=True)  
    ############################
    ##### main image
    
    ax1 = plt.axes( projection=ccrs.Mercator(central_longitude=-80))
    ax1.set_title(champs+ '  resolution ' + str(resolution) +'°' +dico['methode_spatiale'][0],loc='center')
    #ax1 = fig.add_subplot( projection=ccrs.LambertConformal())
    ax1.set_extent(bounds(dico), crs=ccrs.PlateCarree())
    ax1.coastlines(resolution='auto', color='k')
    ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
    ax1.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
    ax1.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
    
    #ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted')    # couche frontieres
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières 
    #coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
    #                            facecolor='none', name='coastline')
    #ax.add_feature(coast, edgecolor='black')   
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.add_feature(cfeature.BORDERS)
        # Put a background image on for nice sea rendering.
    ax1.stock_img()
    
    datamean=data[champs].mean(dim='time').values
    #datamean = np.ma.masked_where(datamean <= 1.588127e-04, datamean)
    levels = np.linspace(0, np.nanmean(data_scale[champs_scale].values), 11)
    
    datamean[datamean<0.00005]=np.nan
    mm = ax1.contourf(data.lon.values,\
      data.lat.values,\
      datamean,\
    #  vmin=0.,\
    #  vmax=40., \
      transform=ccrs.PlateCarree(),\
      levels=levels,\
      cmap="jet",\
      extend='max',
      alpha=0.7)
    
        
    #ax1.plot([-90,-56],[60,60],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-56],[41,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-90],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-56,-56],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    
    cb1 = plt.colorbar(mm, orientation='horizontal',shrink=0.5, pad=0.1)
    #cb1.set_label('CAPExCP proxy', size='x-large')
   
    #plt.savefig('./figure/capecp_'+str(year)+'.png', bbox_inches='tight', pad_inches=0.1)  
    plt.show()


def tracer(dico,data,champs):
    resolution= abs(data.lat.values[0]-data.lat.values[1])
    
    #cmap = mpl.cm.jet
    #norm = BoundaryNorm(clevs, cmap.N, extend='both')
    
    fig = plt.figure(figsize=(28,16), frameon=True)  
    ############################
    ##### main image
    
    ax1 = plt.axes( projection=ccrs.Mercator(central_longitude=-80))
    #ax1 = plt.axes( projection=ccrs.LambertConformal(central_longitude=-80))
    ax1.set_title(champs+ '  resolution ' + str(resolution) +'°' +dico['methode_spatiale'][0],loc='center')
    #ax1 = fig.add_subplot( projection=ccrs.LambertConformal())
    ax1.set_extent(bounds(dico), crs=ccrs.PlateCarree()) 
    ax1.coastlines(resolution='auto', color='k')
    ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
    ax1.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
    ax1.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
    
    #ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted')    # couche frontieres
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières 
    #coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
    #                            facecolor='none', name='coastline')
    #ax.add_feature(coast, edgecolor='black')   
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.add_feature(cfeature.BORDERS)
        # Put a background image on for nice sea rendering.
    ax1.stock_img()
    
    #datamean=np.nanmean(data[champs])
    #datamean = np.ma.masked_where(datamean <= 1.588127e-04, datamean)
    levels = np.linspace(data.min()[champs].values, np.nanmean(data[champs].values), 11)
    
    mm = ax1.contourf(data.lon.values,\
      data.lat.values,\
      data[champs].values,\
    #  vmin=0.,\
    #  vmax=40., \
      transform=ccrs.PlateCarree(),\
      levels=levels,\
      cmap="jet",\
      extend='both',
      alpha=0.7)
    
        
    #ax1.plot([-90,-56],[60,60],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-56],[41,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-90],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-56,-56],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    
    cb1 = plt.colorbar(mm, orientation='horizontal',shrink=0.5, pad=0.1)
   
   
    plt.savefig('carte-1')  
    plt.show()

def tracer_saison(dico,data,champs):
     resolution= abs(data.lat.values[0]-data.lat.values[1])
     clevs = np.linspace(0, np.mean(data[champs].values)*10, 11)
     #cmap = mpl.cm.jet
     #norm = BoundaryNorm(clevs, cmap.N, extend='both')
     
     fig=plt.figure(figsize=(10,6), frameon=True)   
     ax = plt.subplot(111, projection=ccrs.Orthographic(central_longitude=(dico['lonW']+dico['lonE'])/2, central_latitude=(dico['latS']+dico['latN'])/2))
     
     ax.set_extent(bounds(dico), crs=ccrs.PlateCarree())
     ax.set_title(champs+ '  resolution ' + str(resolution) +'°' +dico['methode_spatiale'][0],loc='center')
    
     ax.add_feature(cfeature.OCEAN.with_scale('50m'))
     ax.add_feature(cfeature.LAKES.with_scale('50m'))
     ax.add_feature(cfeature.LAND.with_scale('50m'))
     ax.add_feature(cfeature.BORDERS.with_scale('50m'))
     states_provinces = cfeature.NaturalEarthFeature(
             category='cultural',
             name='admin_1_states_provinces_lines',
             scale='50m',
             facecolor='none')
     
     ax.add_feature(states_provinces, edgecolor='gray')
        
     mm = ax.pcolormesh(data.lon.values,\
                       data.lat.values,\
                       data[champs].mean(dim='time').values,\
                       vmin=0,\
                       vmax=np.nanmean(data[champs].values)*10, \
                       transform=ccrs.PlateCarree(),\
                       cmap='jet' )
     #ax.contour(data.lon.values, data.lat.values, np.mean(data[champs].values,axis=0), levels=clevs, colors='black', linewidths=1, transform=ccrs.PlateCarree())
     
    
     
     cb_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
     cbar = plt.colorbar(mm, cax=cb_ax,extend='both')
     cbar.set_label("par km**2 par jour",horizontalalignment='center',rotation=90)
     ax.coastlines(resolution='110m');
     plt.savefig(champs+ '_resolution_' + str(resolution) +'.png' )
     plt.show()   

'''   
def trace_controle_methode_regrid(dico,dataset,champs):
    clevs = np.linspace(-1, 1, 11)
    #cmap = mpl.cm.seismic
    #norm = BoundaryNorm(clevs, cmap.N, extend='both')
    fig=plt.figure(figsize=(20,6))
    fig.suptitle('cape', fontsize=16)
    
    for k in range (len(method_list)):
        ax=fig.add_subplot(2,3,k+1,projection=ccrs.Orthographic(central_longitude=(dico['lonW']+dico['lonE'])/2, central_latitude=(dico['latS']+dico['latN'])/2))
        
        ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ax.add_feature(cfeature.LAKES.with_scale('50m'))
        ax.add_feature(cfeature.LAND.with_scale('50m'))
        ax.add_feature(cfeature.BORDERS.with_scale('50m'))
        states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='50m',
                facecolor='none')
        
        ax.add_feature(states_provinces, edgecolor='gray')
        
        mm = ax.pcolormesh(dataset[0].lon.values,\
                   dataset[1].lon.values,\
                   np.mean(dataset[k][champs].values,axis=0),\
                   vmin=0,\
                   vmax=500, \
                   transform=ccrs.PlateCarree(),\
                   cmap="jet")
        ax.coastlines(resolution='110m');
    plt.show()
 '''
'''
def trace_occurences_mensuelles(dataset_obs,liste_dataset_model,champs_obs,champs_model,dates,lon,lat,method):
    occurences_model =[]
    color=['--vb','--vy','--vg','--vr']
    count=0
    F= statistiques.nombre_occurence_mensuelle(dataset_obs, champs_obs[0], dates, lon, lat)
   
    if method == 'sum':
        for k in range (len(liste_dataset_model)):
            occurences_model.append(statistiques.nombre_occurence_mensuelle(liste_dataset_model[k], champs_model[k], dates, lon, lat))
    if method == 'mean':
        for k in range (len(liste_dataset_model)):
            occurences_model.append(statistiques.moyenne_occurence_mensuelle(liste_dataset_model[k], champs_model[k], dates, lon, lat))        
    x = np.linspace(0,11,num=12)
    n= int(len(liste_dataset_model)**(1/2)+1)
    
    fig, axs = plt.subplots(n, n)
   
    for i in range(n):
            for j in range (min(n,len(occurences_model))):
                    print(count)
                
                    axs[i, j].plot(x,occurences_model[count],color[count],label=champs_model[count])
                    ax2=axs[i, j].twinx() 
                    ax2.plot(F,'-vk',label='F')
                    axs[i, j].set_xticks(x, liste_mois, rotation=30, ha='center',size=5)
                    count=  min(count+1,len(champs_model)-1)
                    
        
    plt.suptitle('occurence mensuelle_' +method)
    fig.legend(loc="lower right")
    plt.show()
'''

def trace_occurences_mensuelles(dico_Canada,dico,dataset_obs_canada,dataset_obs,liste_dataset_model,champs_obs,champs_model,dates,lon,lat,method,domain):
    occurences_model =[]
    color=['--vb','--vy','--vg','--vr']
    count=0
    F= statistiques.nombre_occurence_mensuelle(dataset_obs, champs_obs[0], dates, lon, lat)
   
    if method == 'sum':
        for k in range (len(liste_dataset_model)):
            occurences_model.append(statistiques.nombre_occurence_mensuelle(liste_dataset_model[k], champs_model[k], dates, lon, lat))
    if method == 'mean':
        for k in range (len(liste_dataset_model)):
            occurences_model.append(statistiques.moyenne_occurence_mensuelle(liste_dataset_model[k], champs_model[k], dates, lon, lat))        
    x = np.linspace(0,11,num=12)
    n= int(len(liste_dataset_model)**(1/2)+1)
    
    #fig, axs = plt.subplots(n, n, constrained_layout=True)
    fig, axs = plt.subplots(n, n)
    for i in range(n):
            for j in range (min(n,len(occurences_model))):
                
                
                    
                
                if count < len(champs_model):    
                    
                    correlation = np.corrcoef(F,occurences_model[count])[0,1]
                    
                    axs[i, j].plot(x,occurences_model[count],color[count],label=champs_model[count])
                    axs[i, j].set_title('correlation : '+str(correlation)[:5],size=8)
                    ax2=axs[i, j].twinx() 
                    ax2.plot(F,'-vk',label='F')
                    axs[i, j].set_xticks(x, liste_mois, rotation=50, ha='center')
                    axs[i, j].tick_params(axis='x', which='major', labelsize=6)
                    axs[i, j].tick_params(axis='y', which='major', labelsize=4)
                    ax2.tick_params(axis='y', which='major', labelsize=4)
                    axs[i, j].legend(loc="upper right")
                    
                    
                    count=  count +1 
    
    axs[1, 1].axis('off')
    
   
    resolution= abs(dataset_obs_canada.lat.values[0]-dataset_obs_canada.lat.values[1])
    ax1 = plt.subplot(224, projection=ccrs.Orthographic(central_longitude=(-149.5-50.5)/2, central_latitude=(40.5+89.5)/2))
    
    #ax1 = plt.axes( projection=ccrs.Mercator(central_longitude=-80))
    ax1.set_title(champs_obs[0]+ '  resolution ' + str(resolution) +'°' ,loc='center', size=8)
    #ax1 = fig.add_subplot( projection=ccrs.LambertConformal())
    ax1.set_extent(bounds(dico_Canada), crs=ccrs.PlateCarree())
    ax1.coastlines(resolution='auto', color='k')
    ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
    ax1.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
    ax1.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
    
    #ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted')    # couche frontieres
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières 
    #coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',     # ajout de la couche cotière 
    #                            facecolor='none', name='coastline')
    #ax.add_feature(coast, edgecolor='black')   
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.add_feature(cfeature.BORDERS)
        # Put a background image on for nice sea rendering.
    ax1.stock_img()
    
    #datamean=np.nanmean(data[champs])
    #datamean = np.ma.masked_where(datamean <= 1.588127e-04, datamean)
    levels = np.linspace(0, np.nanmean(dataset_obs_canada[champs_obs[0]].values)*10, 11)
    
    mm = ax1.contourf(dataset_obs_canada.lon.values,\
      dataset_obs_canada.lat.values,\
      dataset_obs_canada[champs_obs[0]].mean(dim='time').values,\
    #  vmin=0.,\
    #  vmax=40., \
      transform=ccrs.PlateCarree(),\
      levels=levels,\
      cmap="jet",\
      extend='both',
      alpha=0.7)
    
    #ax1.plot([-90,-56],[60,60],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-56],[41,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-90,-90],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    #ax1.plot([-56,-56],[60,41],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    
      
    
    ax1.plot([dico['lonW'],dico['lonE']],[dico['latN'],dico['latN']],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    ax1.plot([dico['lonW'],dico['lonE']],[dico['latS'],dico['latS']],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    ax1.plot([dico['lonW'],dico['lonW']],[dico['latN'],dico['latS']],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    ax1.plot([dico['lonE'],dico['lonE']],[dico['latN'],dico['latS']],color='red',linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
    
    cb1 = plt.colorbar(mm, orientation='horizontal',shrink=0.5, pad=0.1)
    cb1.set_label('F')
    
    
                    
    
        
    
    plt.suptitle('occurence mensuelle_' +method+'_'+domain)
    fig.legend(loc="lower right")
    plt.savefig('Canada')
    plt.show()



def trace_occurences_mensuelles_ecozones(dico_Canada,dico,dataset_obs_canada,dataset_obs_ecozones,liste_dataset_model_ecozones,
                                         champs_obs,champs_model,dates,lon,lat,method):
    cape_ecozones=liste_dataset_model_ecozones[0]
    cp_ecozones=liste_dataset_model_ecozones[1]
    proxy_ecozones=liste_dataset_model_ecozones[2]
    for i in range(0,18):
        
        trace_occurences_mensuelles(dico_Canada,dico,dataset_obs_canada,dataset_obs_ecozones[i]
                                          [cape_ecozones[i],cp_ecozones[i],proxy_ecozones[i]],
                                          champs_obs,champs_model,dates,lon,lat,method,ecozones[i])


def trace_carte_correlation_mensuelle(dico,liste_mask_ecozones,dataset_obs_ecozones,dataset_model_ecozones,champs_obs,champs_model,dates,lon,lat,method='all'):
    correlation_ecozone=[]
    for i in range(0,18):
        correlation_ecozone.append(np.corrcoef(statistiques.nombre_occurence_mensuelle(dataset_obs_ecozones[i], champs_obs, 
                                                                                       dates, lon, lat),
                                   statistiques.moyenne_occurence_mensuelle(dataset_model_ecozones[i], champs_model, 
                                                                            dates, lon, lat))[0,1])
    A=liste_mask_ecozones[0] * correlation_ecozone[0]
    indices = [k for k in range(1,18)]
    if method != 'all':
        del indices[method]
    for i in indices:
        A= A+  liste_mask_ecozones[i]  * correlation_ecozone[i]
        
    tracer(dico,A.where(A != 0),'ecozone')

   
    
    