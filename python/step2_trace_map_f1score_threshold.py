#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:46:33 2022

@author: jonathandurand
"""

import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
#from carto import scale_bar
import xarray as xr
####https://uoftcoders.github.io/studyGroup/lessons/python/cartography/lesson/
#matplotlib.use('Agg')
import seaborn as sns; 
sns.set(style="white", color_codes=True)
sns.set(rc={'figure.figsize':(28,16)})
from mpl_toolkits.axes_grid1 import make_axes_locatable


var='capecp'
if var =="cape":   
    #CAPE
    seuil=np.arange(0,2500,50)
if var =="capecp" or var =='capecp_1':   
    #capecp seuil
    seuilcenter=np.arange(0.00001,1,0.01).tolist()
    seuilend=np.arange(2,130,10).tolist()
    seuil=seuilcenter+seuilend
if  var =="cp":
    #CP
    seuil=np.linspace(0,0.005,150)
if  var =="d":
    #CP
    seuil=np.linspace(-0.0005,0,100)
if  var =="gwd":
    #CP
    seuil=np.linspace(0,4000,100)
if  var =="proxy_d":
    #CP
 
    seuil=np.linspace(0,0.0008,100)

if  var =="proxy_gwd":
    
  
    
    seuil=np.linspace(0,100,100)
    
ds_all = xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/python/all_scores_"+var+"_2018-01-01.nc")
ds_auc=xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/python/auc_"+var+"_2018-01-01.nc")

#ds_all = xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/all_scores_capecp_1_2018-01-01.nc")
#ds_auc=xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/auc_capecp_1_2018-01-01.nc")


var_toplot=ds_auc.auc

#var_toplot=ds_auc.auc
tpr=ds_all.tpr.values
fpr=ds_all.fpr.values
tp=ds_all.tp.values
fp=ds_all.fp.values
fn=ds_all.fn.values
tn=ds_all.tn.values

f1best = np.zeros((len(ds_all.latitude),len(ds_all.longitude)),dtype=float)
f1best[f1best==0]=np.nan 
threshold = np.zeros((len(ds_all.latitude),len(ds_all.longitude)),dtype=float)
threshold[threshold==0]=np.nan 

for i in range(len(f1best[0])):
    print("i", i, len(f1best[0]))
    for y in range(len(f1best)):
        #print(i,y)
        #####Calcultate precision and other indices
        precision=tp[:,y,i]/(tp[:,y,i]+fp[:,y,i])
        recall=tp[:,y,i]/(tp[:,y,i]+fn[:,y,i])
        f1=2*(precision * recall) / (precision + recall)
        plt.plot(f1)
        ##get the intersection between precision and recall
        idx = np.argwhere(np.diff(np.sign(precision - recall))).flatten()
        idx=idx.tolist()
        try :
            if idx[0]==0:
                
                idx.pop(0)
            f1best[y,i]=f1[idx[0]]
           
            threshold[y,i]=seuil[idx[0]]
        except IndexError:
            f1best[y,i]=np.nan
            threshold[y,i]=np.nan
        if var=="cape" and threshold[y,i]<=50:
            threshold[y,i]=np.nan
        if var=="cp" and threshold[y,i]<=0.0001:
            threshold[y,i]=np.nan 
        if var=="capecp" and threshold[y,i]<=0.011:
            threshold[y,i]=np.nan
        if var=="capecp_1" and threshold[y,i]<=0.011:
            threshold[y,i]=np.nan
        if var=="d" and threshold[y,i]>=0:
            threshold[y,i]=np.nan
        if var=="gwd" and threshold[y,i]<=100:
            threshold[y,i]=np.nan
# #which indices are lat lon selected?
# ilon = list(ds_all.longitude.values).index(ds_all.sel(longitude=-75, method='nearest').longitude)
# ilat = list(ds_all.latitude.values).index(ds_all.sel(latitude=45.0, method='nearest').latitude)
# print(' lon index=',ilon,'\n','lat index=', ilat)



############################
##### main image
proj = ccrs.LambertConformal()
f, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection=proj))
h=ax.pcolormesh(ds_auc.longitude,ds_auc.latitude, f1best,vmin=0.0,vmax=1,cmap=plt.cm.jet,transform=ccrs.PlateCarree())

ax.coastlines()

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)


f.add_axes(ax_cb)
plt.colorbar(h, cax=ax_cb)

ax.set_extent([-145,-50,30,75])
ax.coastlines(resolution='auto', color='k')
ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
# ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
ax.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
ax.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac

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
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.BORDERS)
    # Put a background image on for nice sea rendering.
#ax1.stock_img()

ax.set_title("F1 best score for "+var, fontsize=20)  
#f.savefig('./figure/f1best_score_'+var+'.png',bbox_inches='tight',dpi=200)
plt.show()
plt.close()

############################
##### main image
proj = ccrs.LambertConformal()
f, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection=proj))
if var=="cp":
    h=ax.pcolormesh(ds_auc.longitude,ds_auc.latitude, threshold,vmin=0.000,vmax=0.002,cmap=plt.cm.jet,transform=ccrs.PlateCarree())
if var=="cape":
    h=ax.pcolormesh(ds_auc.longitude,ds_auc.latitude, threshold,vmin=0.000,vmax=800,cmap=plt.cm.jet,transform=ccrs.PlateCarree())
if var=="capecp":
    h=ax.pcolormesh(ds_auc.longitude,ds_auc.latitude, threshold,vmin=0.000,vmax=0.6,cmap=plt.cm.jet,transform=ccrs.PlateCarree())
ax.coastlines()

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)


f.add_axes(ax_cb)
plt.colorbar(h, cax=ax_cb)

ax.set_extent([-145,-50,30,75])
ax.coastlines(resolution='auto', color='k')
ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
# ax1.add_feature(cfeature.OCEAN.with_scale('50m'))      # couche ocean
ax.add_feature(cfeature.LAND.with_scale('50m'))       # couche land
ax.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac

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
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.BORDERS)
    # Put a background image on for nice sea rendering.
#ax1.stock_img()

ax.set_title("Best threshold for "+var, fontsize=20)  
#f.savefig('./figure/best_threshold_'+var+'.png',bbox_inches='tight',dpi=200)