# -*- coding: utf-8 -*-

import radiomics
from radiomics import featureextractor
import pandas as pd 
import os
import numpy as np
import SimpleITK as sitk
from radiomics.imageoperations import checkMask

root_path = r'E:\Data_Aneurysm\all_data'

print(os.listdir(root_path)[:])
df=pd.DataFrame()                    
for i in os.listdir(root_path)[:]:
    path1 = root_path + '/' + i
    data_path = path1 + '/data.nii'
    mask_path = path1 + '/mask.nii.gz'
    itk_img=sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(itk_img)
    itk_img1=sitk.ReadImage(data_path)
    
    img_arr1 = sitk.GetArrayFromImage(itk_img1)
    origin = itk_img1.GetOrigin()
    dire = itk_img1.GetDirection()
    space = itk_img1.GetSpacing()
  
    img_arr[img_arr==2] =1
    img_arr[img_arr==3] =0
    img_arr[img_arr==4] =0
    img_arr[img_arr==5] =1
    #img_arr[img_arr==2] =1
    print(i)
    out = sitk.GetImageFromArray(img_arr)
    out.SetOrigin(origin)
    out.SetDirection(dire)
    out.SetSpacing(space)
    
    sitk.WriteImage(out,path1+'/mask1.nii.gz')
    mask_path = path1 + '/mask1.nii.gz'
    # itk_img=sitk.ReadImage(mask_path)
    # img_arr = sitk.GetArrayFromImage(itk_img)
    # print(np.max(img_arr))
    settings={}
    settings['binWidth']=25
    settings['resampledPixelSpacing']=[0.3,0.3,0.3]
    settings['interpolator']=sitk.sitkNearestNeighbor
    settings['normalize']=True
    extractor=featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableImageTypes(Original={},LoG={"sigma":[4.0]},Wavelet={})                                               
    featureVector = extractor.execute(data_path,mask_path)        
    df_add = pd.DataFrame.from_dict(featureVector.values()).T    
    df_add.columns=featureVector.keys()                           
    df = pd.concat([df,df_add])                                
    
df.to_excel(root_path  + '0224.xlsx')



