# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:16:49 2022

@author: ZCS
"""

import os
import pydicom
import numpy as np
from pydicom import dcmwrite
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

root_path = r'E:\Data_Aneurysm\resample'

print(os.listdir(root_path)[:])

def interpolate(volumeImage, newSpacing):
	
	resampleFilter = sitk.ResampleImageFilter()
	resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor) 	
    #mask用近邻差值sitk.sitkNearestNeighbor
    #data用线性差值sitk.sitkLinear
	resampleFilter.SetOutputDirection(volumeImage.GetDirection())
	resampleFilter.SetOutputOrigin(volumeImage.GetOrigin())

	newSpacing = np.array(newSpacing, float)
	newSize = volumeImage.GetSize() / newSpacing * volumeImage.GetSpacing()
	newSize = newSize.astype(np.int)

	resampleFilter.SetSize(newSize.tolist())
	resampleFilter.SetOutputSpacing(newSpacing)
	newVolumeImage = resampleFilter.Execute(volumeImage)

	return newVolumeImage

count = 0                    
for i in os.listdir(root_path)[:]:
    count+=1
    print(count)
    path1 = root_path + '/' + i
    mask_path = path1 + '/mask_nt.nii.gz'
    itk_img=sitk.ReadImage(mask_path)
    resampled_data = interpolate(itk_img, (0.2,0.2,0.2))
    sitk.WriteImage(resampled_data,path1+'/mask_resample_NearestNeighbor_masknt0.2.nii.gz')
  



