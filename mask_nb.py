# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 01:23:10 2022

@author: ZCS
"""

import radiomics
from radiomics import featureextractor
import pandas as pd 
import os
import numpy as np
import SimpleITK as sitk
from radiomics.imageoperations import checkMask
root_path = r'E:\Data_Aneurysm\resample'

print(os.listdir(root_path)[:])
                  
for i in os.listdir(root_path)[:]:
    path1 = root_path + '/' + i
    # data_path = path1 + '/data.nii'
    mask_path = path1 + '/mask_resample_NearestNeighbor_mask0.2.nii.gz'
    itk_img=sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(itk_img)
    # itk_img1=sitk.ReadImage(data_path)
    
    # img_arr1 = sitk.GetArrayFromImage(itk_img1)
    # origin = itk_img1.GetOrigin()
    # dire = itk_img1.GetDirection()
    # space = itk_img1.GetSpacing()
    img_arr[img_arr==2] =1
    img_arr[img_arr==3] =0
    img_arr[img_arr==4] =0
    img_arr[img_arr==5] =1
    
    print(np.max(img_arr))
    out = sitk.GetImageFromArray(img_arr)
    # out.SetOrigin(origin)
    # out.SetDirection(dire)
    # out.SetSpacing(space)
    
    sitk.WriteImage(out,path1+'/msak_nt0.2.nii.gz')