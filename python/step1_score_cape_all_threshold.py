#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:16:51 2022

@author: jonathandurand
"""


import glob
import xarray as xr
import numpy as np
import os.path
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import ajustement
import data_processing
time_start="2018-01-01"
time_end="2019-12-31"

var="cape"
#CAPE
seuil=np.arange(0,2500,50)
#CP
#seuil=np.linspace(0,0.005,150)
#CAPECP
# seuilcenter=np.arange(0.00001,1,0.01).tolist()
# seuilend=np.arange(2,130,10).tolist()
# seuil=seuilcenter+seuilend

## LOAD FILES
filename_var= glob.glob("/home/bun-kim/Documents/stage_UQAM-ESCER/data/model/d950_era5_2018_1.nc")
#ds_var = xr.open_mfdataset(filename_var)
ds_cape_cp= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/cape_cp_era5_2018-2020.nc').sel(time=slice(time_start,time_end))
#cape
var='cape'
ds_var= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/cape_cp_era5_2018-2020.nc').sel(time=slice(time_start,time_end))
base_cape = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_cape, ds_var)
seuil=np.arange(0,2500,50)

#cp
var='cp'
ds_var= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/cape_cp_era5_2018-2020.nc').sel(time=slice(time_start,time_end))
base_cape = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_cape, ds_var)
seuil=np.linspace(0,0.005,150)


#proxy 
var='capecp'
ds_capecp= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/capecp_era5_2018-2019.nc').sel(time=slice(time_start,time_end))
base_proxy = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_proxy, ds_capecp)
seuilcenter=np.arange(0.00001,1,0.01).tolist()
seuilend=np.arange(2,130,10).tolist()
seuil=seuilcenter+seuilend

#proxy décalé
decalage = 1
var='capecp_1'
ds_capecp= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/capecp_era5_2018-2019.nc').sel(time=slice(time_start,time_end))
'''
temp1 = ds_cape_cp.cape.values[:-decalage,:,:]
temp1 =xr.DataArray(temp1,name='cape_'+str(decalage))
temp1 = temp1.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
temp1=temp1.assign_coords({'time' : ds_cape_cp.time.values[:-decalage], 'lat' : ds_cape_cp.lat.values,'lon' : ds_cape_cp.lon.values})
temp1 = xr.DataArray.to_dataset(temp1)

temp2 = ds_cape_cp.cp.values[decalage:,:,:]
temp2 =xr.DataArray(temp2,name='cp_'+str(decalage))
temp2 = temp2.rename({'dim_0':'time','dim_1':'lat','dim_2':'lon'})
temp2=temp2.assign_coords({'time' : ds_cape_cp.time.values[:-decalage], 'lat' : ds_cape_cp.lat.values,'lon' : ds_cape_cp.lon.values})
temp2 = xr.DataArray.to_dataset(temp2)
'''

ds_var = ds_cape_cp.cape.isel(time = slice(0,17519)) * ds_cape_cp.cp.isel(time = slice(1,17520))
ds_var = xr.DataArray.to_dataset(ds_var, name='capecp_'+str(decalage))


base_proxy = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_proxy, ds_var)
seuilcenter=np.arange(0.00001,1,0.01).tolist()
seuilend=np.arange(2,130,10).tolist()
seuil=seuilcenter+seuilend


#proxyd950
var = 'proxy_d'
ds_var = xr.open_mfdataset(
    '/bwk01_01/san/stage_UQAM-ESCER/data/model/d_era5_2018-2019.nc').sel(time=slice(time_start, time_end))


ds_var = ds_cape_cp.cape * ds_cape_cp.cp * ds_var.d
ds_var = xr.DataArray.to_dataset(ds_var, name='proxy_d')

base_cape = {'method_obs': 'min', 'frequence': '1D'}
ds_var = ajustement.resolution_temporelle_obs(
    base_cape, ds_var).where(ds_var.proxy_d < 0)
ds_var = -ds_var
seuil = np.linspace(0, 0.0008, 100)

#gwd
var='gwd'
ds_var= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/gwd_era5_2018-2019_1.nc').sel(time=slice(time_start,time_end))
base_cape = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_cape, ds_var)
seuil=np.linspace(0,1000,100)

#proxy_gwd
var='proxy_gwd'
ds_var= xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/model/gwd_era5_2018-2019_1.nc').sel(time=slice(time_start,time_end))
ds_var = ds_cape_cp.cape * ds_cape_cp.cp * ds_var.gwd
ds_var = xr.DataArray.to_dataset(ds_var, name='proxy_gwd')

base_cape = {'method_obs':'max','frequence':'1D'}
ds_var = ajustement.resolution_temporelle_obs(base_cape, ds_var)
seuil=np.linspace(0,10,100)






ds_foudre = xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/observation/WWLLN_2010-2019.nc').sel(Time=slice(time_start,time_end))
ds_tz = xr.open_mfdataset('/bwk01_01/san/stage_UQAM-ESCER/data/mask/ecozone_mask_1.nc') 
ds_tz=ds_tz.sel(lat=slice(40, 84), lon=slice(212-360, 310-360))
ds_tz=ds_tz.ecozones.values
eco=[0,1,2,4,5,7,8,9,10,12,13,14,15,16,17]
### PROCESS & SUBSET VAR FILES
ds_var=ds_var.resample(time='d').max()
ds_var=ds_var.sel(lat=slice(40, 84), lon=slice(212-360, 310-360),time=slice(time_start,time_end))   
ds_foudre=ds_foudre.sel(lat=slice(40, 84), lon=slice(212-360, 310-360),Time=slice(time_start,time_end))     
##select only may to october
summer = ds_var.time.dt.month.isin(range(5, 11))
ds_var = ds_var.sel(time=summer)
summer = ds_foudre.Time.dt.month.isin(range(5, 11))
ds_foudre= ds_foudre.sel(Time=summer)


#Change the threshold for the VAR, initiate lists

data_all = xr.Dataset( coords={'longitude': ([ 'longitude'], ds_var.variables['lon'][:]),
                                      'latitude': (['latitude',], ds_var.variables['lat'][:]),
                                      'seuil': seuil})
data_auc = xr.Dataset( coords={'longitude': ([ 'longitude'], ds_var.variables['lon'][:]),
                                      'latitude': (['latitude',], ds_var.variables['lat'][:])})

fpr_list=[]
tpr_list=[]

tp = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
tp[tp==0]=np.nan 
tn = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
tn[tn==0]=np.nan 
fp = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
fp[fp==0]=np.nan 
fn = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
fn[fn==0]=np.nan 


auc = np.zeros((len(ds_var.lat),len(ds_var.lon)),dtype=float)
auc[auc==0]=np.nan 
fpr = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
fpr[fpr==0]=np.nan 
tpr = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
tpr[tpr==0]=np.nan 
tnr = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
tnr[tnr==0]=np.nan 
ppv = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
ppv[ppv==0]=np.nan 
npv = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
npv[npv==0]=np.nan 
fnr = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
fnr[fnr==0]=np.nan 
fdr = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
fdr[fdr==0]=np.nan 
acc = np.zeros((len(seuil),len(ds_var.lat),len(ds_var.lon)),dtype=float)
acc[acc==0]=np.nan 


#### SELECT VAR INSIDE XARRAY DATASET
ds_var=ds_var[var].values
#ds_var=ds_var.cp.values*ds_var.cape.values

ds_foudre=ds_foudre.F.values
ds_foudre = np.where(ds_foudre > 1, 1, 0)


for i in range(len(ds_var[0][0])):
    print("i", i, len(ds_var[0][0]))
    for y in range(len(ds_var[0])):
        
# for i in range(73,74,1):
#     print("i", i, len(ds_var[0][0]))
#     for y in range(5,6,1):
        
        if ds_tz[y,i] in eco:
           
        #if np.isnan(ds_tz[y,i]) == False:
            fpr_list=[]
            tpr_list=[]
            for s in range(len(seuil)):
              
                #print('s',s)
                ds_var_mat = np.where(ds_var >= seuil[s], 1, 0)
             
                predictions=ds_var_mat[:,y,i]
             
                actuals=ds_foudre[:,y,i]
                        
                cf_train_matrix = confusion_matrix(actuals, predictions,labels=[1,0])
               
                #print("Confusion matrix \n", cf_train_matrix)
                
                #plt.figure(figsize=(10,8))
                #sns.heatmap(cf_train_matrix, annot=True, fmt='d')
                    
                TP = cf_train_matrix[0][0]
                
                if np.array_equal(predictions,actuals):
                    FN=0
                    FP=0
                    TN=0
                    
                    fn[s,y,i]=FN
                    tp[s,y,i]=TP
                    fp[s,y,i]=FP
                    tn[s,y,i]=TN
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    TPR = TP/(TP+FN)
                    # Precision or positive predictive value
                    PPV = TP/(TP+FP)

                    # Overall accuracy
                    ACC = (TP+TN)/(TP+FP+FN+TN)
                    fn[s,y,i]=FN
                    tp[s,y,i]=TP
                    fp[s,y,i]=FP
                    tn[s,y,i]=TN

                    TPR=TP/(TP+FN)
                    FPR=0
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    tpr[s,y,i] = TP/(TP+FN)
                    # Specificity or true negative rate
                    tnr[s,y,i] = 0
                    # Precision or positive predictive value
                    ppv[s,y,i] = TP/(TP+FP)
                    # Negative predictive value
                    npv[s,y,i] = 0
                    # Fall out or false positive rate
                    fpr[s,y,i] = 0
                    # False negative rate
                    fnr[s,y,i] = 0
                    # False discovery rate
                    fdr[s,y,i] = 0
                    # Overall accuracy
                    acc[s,y,i] = (TP+TN)/(TP+FP+FN+TN)                    

                    
                else:  
                    FN = cf_train_matrix[0][1]
                    FP = cf_train_matrix[1][0]
                    TN = cf_train_matrix[1][1]
                
                    fn[s,y,i]=FN
                    tp[s,y,i]=TP
                    fp[s,y,i]=FP
                    tn[s,y,i]=TN
                    
                    TPR=TP/(TP+FN)
                    FPR=FP/(FP+TN)
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    tpr[s,y,i] = TP/(TP+FN)
                    # Specificity or true negative rate
                    tnr[s,y,i] = TN/(TN+FP) 
                    # Precision or positive predictive value
                    ppv[s,y,i] = TP/(TP+FP)
                    # Negative predictive value
                    npv[s,y,i] = TN/(TN+FN)
                    # Fall out or false positive rate
                    fpr[s,y,i] = FP/(FP+TN)
                    # False negative rate
                    fnr[s,y,i] = FN/(TP+FN)
                    # False discovery rate
                    fdr[s,y,i] = FP/(TP+FP)
                    # Overall accuracy
                    acc[s,y,i] = (TP+TN)/(TP+FP+FN+TN)
                
                fpr_list.append(FPR)
                tpr_list.append(TPR)
            
                #print(cf_train_matrix)
            fpr_list.insert(0,0)
            fpr_list.insert(len(fpr_list),1)
            tpr_list.insert(0,0)
            tpr_list.insert(len(tpr_list),1)
            
            sorted_index = np.argsort(fpr_list)
            fpr_list_sorted =  np.array(fpr_list)[sorted_index]
            tpr_list_sorted = np.array(tpr_list)[sorted_index]
            #auc[y,i]=integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
            #auc=roc_auc_score(actuals, predictions)
            auc[y,i]=metrics.auc(fpr_list_sorted, tpr_list_sorted)

data_all["tp"] = (['seuil','latitude', 'longitude'],  tp)
data_all["tn"] = (['seuil','latitude', 'longitude'],  tn)
data_all["fn"] = (['seuil','latitude', 'longitude'],  fn)
data_all["fp"] = (['seuil','latitude', 'longitude'],  fp)
data_all["fpr"] = (['seuil','latitude', 'longitude'],  fpr)
data_all["tpr"] = (['seuil','latitude', 'longitude'],  tpr)
data_all["tnr"] = (['seuil','latitude', 'longitude'],  tnr)
data_all["ppv"] = (['seuil','latitude', 'longitude'],  ppv)
data_all["npv"] = (['seuil','latitude', 'longitude'],  npv)
data_all["fdr"] = (['seuil','latitude', 'longitude'],  fdr)
data_all["fnr"] = (['seuil','latitude', 'longitude'],  fnr)
data_all["acc"] = (['seuil','latitude', 'longitude'],  acc)


data_auc["auc"] = (['latitude', 'longitude'],  auc)


comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in data_all.data_vars}
outfile="./all_scores_"+str(var)+"_"+time_start+".nc"
if os.path.exists(outfile):
    os.remove(outfile)
data_all.to_netcdf(path=outfile,format="NETCDF4_CLASSIC",encoding=encoding)

encoding = {var: comp for var in data_auc.data_vars}
outfile="./auc_"+str(var)+"_"+time_start+".nc"
if os.path.exists(outfile):
    os.remove(outfile)
data_auc.to_netcdf(path=outfile,format="NETCDF4_CLASSIC",encoding=encoding)