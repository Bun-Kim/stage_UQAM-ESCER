#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 00:36:56 2022

@author: jonathandurand
"""

import matplotlib.pylab as plt
import numpy as np
#from carto import scale_bar
import xarray as xr
####https://uoftcoders.github.io/studyGroup/lessons/python/cartography/lesson/
#matplotlib.use('Agg')
import seaborn as sns; 
sns.set(style="white", color_codes=True)
sns.set(rc={'figure.figsize':(28,16)})


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

var='capecp_1'
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
if  var =="gwd":
    #CP
    seuil=np.linspace(0,4000,100)
if  var =="d":
    #CP
    seuil=np.linspace(-0.0005,0,100)
if  var =="proxy_d":
    #CP
 
    seuil=np.linspace(0,0.0008,100)

if  var =="proxy_gwd":
    
    
    
    seuil=np.linspace(0,10,100)
    

ds_cape =xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/all_scores_capecp_1_2018-01-01.nc")
ds_cape_auc=xr.open_mfdataset("/bwk01_01/san/stage_UQAM-ESCER/auc_capecp_1_2018-01-01.nc")
# ds_cp = xr.open_mfdataset("all_scores_cp.nc")
# ds_cp_auc=xr.open_mfdataset("auc_cp.nc")

#var_toplot=ds_auc.auc
tpr=ds_cape.tpr.sel(latitude=45, longitude=-75,method="nearest").values
fpr=ds_cape.fpr.sel(latitude=45, longitude=-75,method="nearest").values
tp=ds_cape.tp.sel(latitude=45, longitude=-75,method="nearest").values
fp=ds_cape.fp.sel(latitude=45, longitude=-75,method="nearest").values
fn=ds_cape.fn.sel(latitude=45, longitude=-75,method="nearest").values
tn=ds_cape.tn.sel(latitude=45, longitude=-75,method="nearest").values

#which indices are lat lon selected?
ilon = list(ds_cape.longitude.values).index(ds_cape.sel(longitude=-75, method='nearest').longitude)
ilat = list(ds_cape.latitude.values).index(ds_cape.sel(latitude=45.0, method='nearest').latitude)
print(' lon index=',ilon,'\n','lat index=', ilat)

auc=ds_cape_auc.auc.sel(latitude=45, longitude=-75,method="nearest").values

# cp_tpr=ds_cp.tpr.sel(latitude=45, longitude=-75,method="nearest").values
# cp_fpr=ds_cp.fpr.sel(latitude=45, longitude=-75,method="nearest").values
# auc_cp=ds_cp_auc.auc.sel(latitude=45, longitude=-75,method="nearest").values


##calculate J_youden score
cutoff_youdens_j(fpr,tpr,seuil)

# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (seuil[ix], gmeans[ix]))
#####Calcultate precision and other indices
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1=2*(precision * recall) / (precision + recall)
##get the intersection between precision and recall
idx = np.argwhere(np.diff(np.sign(precision - recall))).flatten()
idx=idx.tolist()
if idx[0]==0:
    idx.pop(0)
f1best=f1[idx[0]]


fig,(ax1, ax2,ax3)=plt.subplots(1,3,figsize=(18,6))
ax1.plot(fpr, tpr,linestyle='--', marker='o', markersize=4, color='darkorange', lw = 2, label='ROC - AUC :'+str("{:.2f}".format(auc)), clip_on=False)
ax1.scatter(fpr[ix], tpr[ix], marker='o',s=120, color='black', label='Best')
# plt.plot(cp_fpr, cp_tpr,linestyle='--', marker='o', markersize=4,color='darkblue', lw = 2, label='ROC CP - AUC: '+str(auc_cp), clip_on=False)
ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax1.axis(xmin=0.0,xmax=1.0)
ax1.axis(ymin=0.0,ymax=1.0)
# for i, txt in enumerate(seuil):
#     ax.annotate(txt, (fpr_list[i]+0.001, tpr_list[i]+0.001))
#plt.text(fpr_list+0.3, tpr_list+0.3, str(seuil), fontsize=9)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC curve')
ax1.legend(loc="lower right")

ax2.plot(recall, precision,linestyle='--', marker='o', markersize=4, color='darkred', lw = 2, label='Recall-Precision', clip_on=False)
#ax2.scatter(fpr[ix], tpr[ix], marker='o',s=120, color='black', label='Best')
# plt.plot(cp_fpr, cp_tpr,linestyle='--', marker='o', markersize=4,color='darkblue', lw = 2, label='ROC CP - AUC: '+str(auc_cp), clip_on=False)
ax2.plot([1, 0], [1, 0], color='navy', linestyle='--')
ax2.axis(xmin=0.0,xmax=1.0)
ax2.axis(ymin=0.0,ymax=1.0)
# for i, txt in enumerate(seuil):
#     ax.annotate(txt, (fpr_list[i]+0.001, tpr_list[i]+0.001))
#plt.text(fpr_list+0.3, tpr_list+0.3, str(seuil), fontsize=9)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('PR curve')
ax2.legend(loc="lower right")

ax3.plot(seuil, f1,linestyle='--', marker='o', markersize=4, color='darkgreen', lw = 2, label='F1 score = '+str("{:.2f}".format(f1best)))
ax3.plot(seuil, recall,linestyle='--', marker='o', markersize=4, color='darkred', lw = 2, label='recall')
ax3.plot(seuil, precision,linestyle='--', marker='o', markersize=4, color='darkblue', lw = 2, label='precision')
threshold=seuil[idx[0]]
ax3.plot([threshold, threshold], [0, 1], color='black', linestyle='--', label='Best threshold = '+str("{:.3f}".format(threshold)))

#ax2.scatter(fpr[ix], tpr[ix], marker='o',s=120, color='black', label='Best')
# plt.plot(cp_fpr, cp_tpr,linestyle='--', marker='o', markersize=4,color='darkblue', lw = 2, label='ROC CP - AUC: '+str(auc_cp), clip_on=False)
#ax3.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax3.axis(xmin=0.0,xmax=seuil[-1])
if var=="capecp":
    ax3.axis(xmin=0.0,xmax=1)
ax3.axis(ymin=0.0,ymax=1.0)
# for i, txt in enumerate(seuil):
#     ax.annotate(txt, (fpr_list[i]+0.001, tpr_list[i]+0.001))
#plt.text(fpr_list+0.3, tpr_list+0.3, str(seuil), fontsize=9)
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Metrics')
ax3.set_title('Threshold/scores')
ax3.legend(loc="lower right")
plt.suptitle(str(var))
plt.show()
plt.savefig('score_'+var+'.png',bbox_inches='tight',dpi=200)