#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  WRITE CSV FOR PET/CT TRAINING   ############################################# 

def write_training_csv():
    import csv
    import glob
    import numpy as np

    filename = "/media/storage/projet_LYSA_TEP_3.5/TEP_CT_training_filenames.csv"
    path_Mask = "/media/storage/projet_LYSA_TEP_3.5/RAWDATA/PetMask"
    path_PetSUV = "/media/storage/projet_LYSA_TEP_3.5/RAWDATA/PetSuv"
    path_CTUH = "/media/storage/projet_LYSA_TEP_3.5/RAWDATA/CtUh"

    PET_scans_ids = np.sort(glob.glob(path_PetSUV+'/*')) 
    CT_scans_ids = np.sort(glob.glob(path_CTUH+'/*')) 
    masks_ids = np.sort(glob.glob(path_Mask+'/*')) 

    with open(filename,"w") as file:
        filewriter = csv.writer(file)
        for PET_scan_id,CT_scan_id,mask_id in zip(PET_scans_ids,CT_scans_ids,masks_ids):
                filewriter.writerow((PET_scan_id,CT_scan_id,mask_id))

###########################  WRITE CSV FOR PET/CT PREDICTION   ############################################# 

def write_prediction_csv():
    import csv
    import glob
    import numpy as np

    csv_filename = "/media/storage/projet_LYSA_TEP_3.5/TEP_CT_filenames.csv"
    path_PetSUV = "/media/storage/projet_LYSA_TEP_3.5/RAWDATA/PetSuv"
    path_CtUh = "/media/storage/projet_LYSA_TEP_3.5/RAWDATA/CtUh"

    PET_ids = np.asarray(np.sort(glob.glob(path_PetSUV+'/*nii')))
    CT_ids = np.asarray(np.sort(glob.glob(path_CtUh+'/*nii')))

    with open(csv_filename,"w") as file:
        filewriter = csv.writer(file)
        for PET_id,CT_id in zip(PET_ids,CT_ids):
                filewriter.writerow((PET_id,CT_id))
        
if __name__=='__main__':
    
    import sys
    if sys.argv[1]=='TRAINING' or sys.argv[1]=='PREDICTION':
        mode = sys.argv[1]
    else:
        mode = 'TRAINING' #default
    
    if mode=='TRAINING':
        write_training_csv()
        
    if mode=='PREDICTION':
        write_prediction_csv()

    