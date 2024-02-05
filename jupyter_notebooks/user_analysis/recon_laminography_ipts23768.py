# Copyright (C) 2017 by S. V. Venkatakrishnan (venkatakrisv@ornl.gov)
# All rights reserved. BSD 3-clause License.
# This file is part of the tomoLam package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

import numpy as np
from matplotlib import pyplot as plt
#import pyqtgraph as pg
from readNeutronData import readRadiographTifDir,readAverageTiffDir
from tomoORNL.reconEngine import *
import tomopy
import dxchange
from tomopy.misc.corr import median_filter
import time
from tomoORNL.corrections import apply_proj_tilt
from tomoORNL.utils import plot_cross_section

data_path='/HFIR/CG1D/IPTS-23768/raw/ct_scans/2024_01_22_laminography_test'
bright_path='/HFIR/CG1D/IPTS-23768/raw/ob/2024_01_22_laminography_test'
dark_path='/HFIR/CG1D/IPTS-23768/raw/dc/2024_01_21_45s'
fbp_fname='/netdisk/imaging/results/HFIR/IPTS-23768/fbp'
mbir_fname='/netdisk/imaging/results/HFIR/IPTS-23768/mbir'
write_output=True
display_output=False
z_start=500
z_numSlice=1000
alpha=np.array([60.0])
rot_center=1024 + 200.0 #1032.5
det_tilt=0 #0.88 
tot_col=2048
num_col=1600 
view_subsamp=4
det_x=1.0
det_y=1.0
vol_xy=1.0
vol_z=1.0
f_c=0.5
end_ang_idx=1162
zero_idx=0
oneeighty_idx=581//view_subsamp
MAX_CORE=240
disp_slice_idx=z_numSlice//2

rec_params={}
rec_params['num_iter']=100
rec_params['gpu_index']=[0,1,2,3,4,5,6,7]
rec_params['MRF_P']=1.2
rec_params['MRF_SIGMA']=0.75
rec_params['debug']=False
rec_params['verbose']=True
rec_params['stop_thresh']=0.01

rec_params['filt_type']='Ram-Lak'
rec_params['filt_cutoff']=f_c

#Miscalibrations - detector offset, tilt
miscalib={}
miscalib['delta_u']=(tot_col/2 - rot_center)
miscalib['delta_v']=0
miscalib['phi']=0 #-det_tilt*np.pi/180

brights=readAverageTiffDir(bright_path,z_start,z_numSlice)
darks=readAverageTiffDir(dark_path,z_start,z_numSlice)
count_data,angles=readRadiographTifDir(data_path,z_start,z_numSlice)
print(angles)


print('Applying median filter to remove gammas')
filt_size = 7 #window of median filter 
count_data= median_filter(count_data,size=filt_size,axis=0,ncore=MAX_CORE)

print(count_data.shape)
print(brights.shape)
print(darks.shape)

norm_data = -np.log((count_data - darks) / (brights - darks))
count_data[np.isnan(norm_data)]=0 #remove the bad values
count_data[np.isinf(norm_data)]=0 #remove the bad values 
norm_data[np.isnan(norm_data)]=0
norm_data[np.isinf(norm_data)]=0

import tomopy
norm_data = tomopy.remove_stripe_fw(norm_data,level=5,ncore=MAX_CORE)

#print('Tilt axis correction ..')
#count_data=apply_proj_tilt(count_data,det_tilt,ncore=MAX_CORE)
#norm_data=apply_proj_tilt(norm_data,det_tilt,ncore=MAX_CORE)

print(norm_data.max())
print(norm_data.min())

####Crop Data along the detector column axis #####
count_data= count_data[:,:,tot_col//2-num_col//2:tot_col//2+num_col//2]
norm_data = norm_data[:,:,tot_col//2-num_col//2:tot_col//2+num_col//2]
######End of data cropping #######

norm_data=norm_data.swapaxes(0,1)
count_data=count_data.swapaxes(0,1)

#For data set 1
print('Subsetting data ..')
ang_idx = range(0,end_ang_idx,view_subsamp)
count_data=count_data[:,ang_idx,:]
norm_data = norm_data[:,ang_idx,:] #Subset to run on GPU 
angles = angles[ang_idx]

print('Shape of data array')
print(count_data.shape)

proj_data = norm_data 

det_row,num_angles,det_col=proj_data.shape
proj_dims=np.array([det_row,num_angles,det_col])

proj_params={}
proj_params['type'] = 'par'
proj_params['dims']= proj_dims
proj_params['angles'] = angles*np.pi/180
proj_params['forward_model_idx']=2
proj_params['alpha']=alpha*np.pi/180
proj_params['pix_x']=det_x
proj_params['pix_y']=det_y

vol_params={}
vol_params['vox_xy']=vol_xy
vol_params['vox_z']=vol_z
vol_params['n_vox_z']=int(det_row*det_y/vol_z)
vol_params['n_vox_y']=int(det_col*det_x/vol_xy)
vol_params['n_vox_x']=int(det_col*det_x/vol_xy)

rot_center_est=tomopy.find_center_pc(np.squeeze(norm_data[:,zero_idx,:]), np.squeeze(norm_data[:,oneeighty_idx,:]), tol=0.5)
print('Estimated center of rotation %f' %rot_center_est)

#plt.imshow(np.squeeze(proj_data[:,zero_idx,:]),cmap='gray')

t=time.time()
print('Starting TomoPy-GR..')
#rec_gridrec_tp = tomopy.recon(proj_data, proj_params['angles'], center=det_col//2-miscalib['delta_u'], sinogram_order=True,algorithm='gridrec')
t_gr=time.time()-t
print('Time for Gridrec %f' % t_gr)

t=time.time()
print('Starting ASTRA-FBP..')
rec_fbp=np.float32(analytic(proj_data,proj_params,miscalib,vol_params,rec_params))
t_fbp=time.time()-t
print('Time for FBP %f' % t_fbp)
if write_output == True:
    dxchange.write_tiff_stack(rec_fbp, fname=fbp_fname, start=z_start,overwrite=True)

print('Starting MBIR..')
rec_mbir=np.float32(MBIR(proj_data,count_data,proj_params,miscalib,vol_params,rec_params))
if write_output == True:
    dxchange.write_tiff_stack(rec_mbir, fname=mbir_fname, start=z_start,overwrite=True)

if display_output == True:
    #Display results
    plot_cross_section(rec_fbp,plt_title='FBP')
    plot_cross_section(rec_mbir,plt_title='MBIR')
#   plt.imshow(rec_fbp[disp_slice_idx],cmap='gray')
#   plt.imshow(rec_mbir[disp_slice_idx],cmap='gray')
#   plt.show()





