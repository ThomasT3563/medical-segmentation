
# NB : Modality_TRAINING_PET_CT and Modality_PREDICTION_PET_CT are highly redondant
#      it is possible and personnaly recommended to later develop a unique class PET CT 
#      modality that handle dependencies in Prediction and Training modes

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages
from copy import copy

import numpy as np
import SimpleITK as sitk
import os
import random
from math import pi
from os.path import splitext,basename

from global_tools.tools import display_loading_bar
from global_tools.transformations import isometric_resample

class Modality_TRAINING_PET_CT(object):
    """
        Modality used to train a PET/CT model
    
        Functions required:
            - PREPROCESS
            - DATA_AUGMENTATION
            - VISUALISATION_MIP_PREPROCESS
            - get_GENERATOR
            - PREDICT_MASK
            - VISUALISATION_MIP_PREDICTION
    """
    
    # globals definitions
    labels_names = ['Background','Lymphoma',]
    labels_numbers = [0,1]
    number_channels = 2
    
    ####################################################################################################
    
    def PREPROCESS(self,data_set_ids,path_output,
                   output_shape=None,pixel_size=None,
                   resample=True,normalize=True):
        """
            Perform preprocessing and save new datas from a dataset
            
            data_set_ids : [(PET_id_1,CT_id_1,MASK_id_1),(PET_id_2,CT_id_2,MASK_id_2)...]
        """
        n_patients = len(data_set_ids)
    
        preprocessed_data_set_ids = []

        for i,data_set_id in enumerate(data_set_ids):

            # display a loading  bar
            display_loading_bar(iteration=i,length=n_patients,add_char=basename(data_set_id[0])+'    ')

            # load data set
            self.PET_id,self.CT_id,self.MASK_id = data_set_id
            
            self.PET_img  = sitk.ReadImage(self.PET_id ,sitk.sitkFloat32)
            self.CT_img   = sitk.ReadImage(self.CT_id  ,sitk.sitkFloat32)
            self.MASK_img = sitk.ReadImage(self.MASK_id,sitk.sitkUInt8)

            if normalize:
                self.PREPROCESS_normalize()

            if resample:
                self.PREPROCESS_resample_CT_to_TEP()
                self.PREPROCESS_resample_TEPCT_to_CNN(output_shape[::-1],pixel_size[::-1]) #reorder to [x,y,z]

            # save preprocess data
            new_PET_id = path_output+'/'+splitext(basename(data_set_id[0]))[0]+'.nii'
            new_CT_id   = path_output+'/'+splitext(basename(data_set_id[1]))[0]+'.nii'
            new_MASK_id   = path_output+'/'+splitext(basename(data_set_id[2]))[0]+'.nii'

            preprocessed_data_set_ids.append((new_PET_id,new_CT_id,new_MASK_id))

            self.PREPROCESS_save(new_filenames=preprocessed_data_set_ids[i])
        
        # clear
        del self.PET_id, self.CT_id, self.MASK_id
        del self.PET_img, self.CT_img, self.MASK_img
        
        return preprocessed_data_set_ids
    
    def PREPROCESS_normalize(self,):
        """ called by PREPROCESS """
        
        # NB : possibility to add threshold values to hide artefacts
        
        # normalization TEP
        self.PET_img = sitk.ShiftScale(self.PET_img,shift=0.0, scale=1./10.)
        # normalization CT
        self.CT_img = sitk.ShiftScale(self.CT_img,shift=1000, scale=1./2000.)
        # normalization MASK
        self.MASK_img = sitk.Threshold(self.MASK_img, lower=0.0, upper=1.0, outsideValue=1.0)
        return None
    
    def PREPROCESS_resample_CT_to_TEP(self):
        """ called by PREPROCESS """
        
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter() 
        transformation.SetOutputDirection(self.PET_img.GetDirection())
        transformation.SetOutputOrigin(self.PET_img.GetOrigin())
        transformation.SetOutputSpacing(self.PET_img.GetSpacing())
        transformation.SetSize(self.PET_img.GetSize())
        transformation.SetInterpolator(sitk.sitkBSpline)

        # apply transformations on CT IMG
        self.CT_img = transformation.Execute(self.CT_img)
        
        return None
    
    def PREPROCESS_resample_TEPCT_to_CNN(self,new_shape,new_spacing):
        """ called by PREPROCESS """
        
        # compute transformation parameters
        new_Origin = self.compute_new_Origin(new_shape,new_spacing)
        new_Direction = self.PET_img.GetDirection()
        
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter() 
        transformation.SetOutputDirection(new_Direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(new_spacing)
        transformation.SetSize(new_shape)

        # apply transformations on PET IMG
        transformation.SetInterpolator(sitk.sitkBSpline)
        self.PET_img = transformation.Execute(self.PET_img)
        # apply transformations on CT IMG
        transformation.SetInterpolator(sitk.sitkBSpline)
        self.CT_img = transformation.Execute(self.CT_img)
        # apply transformations on MASK
        transformation.SetInterpolator(sitk.sitkNearestNeighbor)
        self.MASK_img = transformation.Execute(self.MASK_img)
        return None
    
    def compute_new_Origin(self,new_shape,new_spacing):
        """ called by PREPROCESS_resample """
        
        origin = np.asarray(self.PET_img.GetOrigin())
        shape = np.asarray(self.PET_img.GetSize())
        spacing = np.asarray(self.PET_img.GetSpacing())
        new_shape = np.asarray(new_shape)
        new_spacing = np.asarray(new_spacing)
        
        return tuple(origin+0.5*(shape*spacing-new_shape*new_spacing))

    def PREPROCESS_save(self,new_filenames):
        """ called by PREPROCESS """

        sitk.WriteImage(self.PET_img,new_filenames[0])
        sitk.WriteImage(self.CT_img,new_filenames[1])
        sitk.WriteImage(self.MASK_img,new_filenames[2])
        return None
    
    ####################################################################################################
    
    def DATA_AUGMENTATION(self,data_set_ids,augmentation_factor):
        """
            Perform data augmentation operations on a dataset
                - Data are augmented "augmentation_factor" times
            
            data_set_ids : [(PET_id_1,CT_id_1,MASK_id_1),(PET_id_2,CT_id_2,MASK_id_2)...]
        """
        
        n_patients = len(data_set_ids)
        new_data_set_ids = []

        for i,data_set_id in enumerate(data_set_ids):

            # load files
            PET_id,CT_id,MASK_id = data_set_id

            PET_img  = sitk.ReadImage(PET_id,sitk.sitkFloat32)
            CT_img   = sitk.ReadImage(CT_id,sitk.sitkFloat32)
            MASK     = sitk.ReadImage(MASK_id,sitk.sitkUInt8)
            
            # display a loading  bar
            display_loading_bar(iteration=i,length=n_patients,add_char=basename(MASK_id)+'             ')

            for factor in range(1,augmentation_factor):

                # generate random deformation
                def_ratios = self.DATA_AUGMENTATION_DeformationRatios()

                # apply same transformation to PET / CT / Mask
                new_PET_img  = self.DATA_AUGMENTATION_AffineTransformation(image=PET_img ,
                                                                           interpolator=sitk.sitkBSpline,
                                                                           deformations=def_ratios)
                new_CT_img   = self.DATA_AUGMENTATION_AffineTransformation(image=CT_img,
                                                                           interpolator=sitk.sitkBSpline,
                                                                           deformations=def_ratios)
                new_MASK_img = self.DATA_AUGMENTATION_AffineTransformation(image=MASK,
                                                                           interpolator=sitk.sitkNearestNeighbor,
                                                                           deformations=def_ratios)

                # generates names
                new_PET_id = splitext(PET_id)[0]+'_augm'*factor+'.nii'
                new_CT_id = splitext(CT_id)[0]+'_augm'*factor+'.nii'
                new_Mask_id = splitext(MASK_id)[0]+'_augm'*factor+'.nii'

                # saves
                sitk.WriteImage(new_PET_img,new_PET_id)
                sitk.WriteImage(new_CT_img,new_CT_id)
                sitk.WriteImage(new_MASK_img,new_Mask_id)

                new_data_set_ids.append((new_PET_id,new_CT_id,new_Mask_id))
        
        return new_data_set_ids
    
    def DATA_AUGMENTATION_DeformationRatios(self):
        """ Called by DATA_AUGMENTATION """
        
        deformation = {'Translation':(random.uniform(-20,20),random.uniform(-20,20),random.uniform(-20,20)),
                       'Scaling'    :(random.uniform(0.8,1.2),random.uniform(0.8,1.2),random.uniform(0.8,1.2)),
                       'Rotation'   :random.uniform((-pi/15),(pi/15))}
        return deformation
    
    def DATA_AUGMENTATION_AffineTransformation(self,image,interpolator,deformations):
        """ Called by DATA_AUGMENTATION """
        
        center = tuple(np.asarray(image.GetOrigin()) + 0.5*np.asarray(image.GetSize())*np.asarray(image.GetSpacing()))
        
        transformation = sitk.AffineTransform(3)
        transformation.SetCenter(center)
        transformation.Scale(deformations['Scaling'])
        transformation.Rotate(axis1=0,axis2=2,angle=deformations['Rotation'])
        transformation.Translate(deformations['Translation'])
        reference_image = image
        default_value = 0.0                                 
        
        return sitk.Resample(image,reference_image,transformation,interpolator,default_value)
    
    ####################################################################################################
    
    def VISUALISATION_MIP_PREPROCESS(self,path_output,data_set_ids,filename=None):
        """
            Generates a MIP in .pdf file from data set filenames 
            
            data_set_ids : [(PET_id_1,CT_id_1,MASK_id_1),(PET_id_2,CT_id_2,MASK_id_2)...]
        """
        if filename is None:
                filename = path_output+"/PETCTMASK_MIP_"+time.strftime("%m%d%H%M%S")+".pdf"
        else:
            filename = path_output+'/'+filename

        # generates folder 
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        n_patients = len(data_set_ids)

        transparency = 0.7

        color_CT = plt.cm.gray
        color_PET = plt.cm.plasma
        color_MASK = copy(plt.cm.Greys)
        color_MASK.set_bad('white', 0.0)

        with PdfPages(filename) as pdf:

            # loop on files to get MIP visualisation
            for i,data_set_id in enumerate(data_set_ids):

                display_loading_bar(iteration=i,length=n_patients,add_char=basename(data_set_id[0])+'    ')

                # load imgs
                PET_id,CT_id,Mask_id = data_set_id
                PET_scan = sitk.GetArrayFromImage(sitk.ReadImage(PET_id))
                CT_scan = sitk.GetArrayFromImage(sitk.ReadImage(CT_id))
                Mask = sitk.GetArrayFromImage(sitk.ReadImage(Mask_id))

                # for TEP visualisation
                PET_scan = np.where(PET_scan>1.0,1.0,PET_scan)
                PET_scan = np.where(PET_scan<0.0,0.0,PET_scan)

                # for CT visualisation
                CT_scan = np.where(CT_scan>1.0,1.0,CT_scan)
                CT_scan = np.where(CT_scan<0.0,0.0,CT_scan)

                # for correct visualisation
                PET_scan = np.flip(PET_scan,axis=0)
                CT_scan = np.flip(CT_scan,axis=0)
                Mask = np.flip(Mask,axis=0)

                # stacked projections           
                PET_scan = np.hstack((np.amax(PET_scan,axis=1),np.amax(PET_scan,axis=2)))
                CT_scan = np.hstack((np.amax(CT_scan,axis=1),np.amax(CT_scan,axis=2)))
                Mask = np.hstack((np.amax(Mask,axis=1),np.amax(Mask,axis=2)))

                ############################# PLOT ###############################
                f = plt.figure(figsize=(15, 10))
                f.suptitle(splitext(basename(PET_id))[0], fontsize=15)

                plt.subplot(121)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.axis('off')
                plt.title('PET/CT',fontsize=20)

                plt.subplot(122)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.imshow(np.where(Mask,0,np.nan),cmap=color_MASK,origin='lower')
                plt.axis('off')
                plt.title('PET/CT + Segmentation',fontsize=20)

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                ##################################################################
    
    ####################################################################################################
    def get_GENERATOR(self,data_set_ids):
        """
            Return an object data set filenames generating an iterator object for tensorflow DataSet creation
            
            data_set_ids : [(PET_id_1,CT_id_1,MASK_id_1),(PET_id_2,CT_id_2,MASK_id_2)...]
        """
        
        class PET_CT_MASK_generator(object):

            def __init__(self,data_set_ids):

                self.data_set_ids = data_set_ids   

                self.number_of_images = len(data_set_ids)

                (PET_scans_id,CT_scans_id,MASKS_id) = data_set_ids[0]
                
                # temporary load
                PET_scan=tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(PET_scans_id,sitk.sitkFloat32)),dtype=tf.float32)
                CT_scan=tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(CT_scans_id,sitk.sitkFloat32)),dtype=tf.float32)

                tmp_PET_CT = tf.stack((PET_scan,CT_scan),axis=-1)
                tmp_MASK = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(MASKS_id,sitk.sitkUInt8 )),dtype=tf.uint8)
                tmp_MASK = tf.expand_dims(tmp_MASK,axis=-1)

                self.types = (tmp_PET_CT.dtype, tmp_MASK.dtype)
                self.shapes = (tmp_PET_CT.shape, tmp_MASK.shape)

                del PET_scan
                del CT_scan
                del tmp_PET_CT
                del tmp_MASK

            def call(self):
                """
                    yield generator of tf.Tensor as (Scan,Mask)==(Output,Labels)
                """
                for it in range(self.number_of_images):
                    
                    (PET_scans_id,CT_scans_id,MASKS_id) = data_set_ids[it]
                    
                    PET_scan = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(PET_scans_id,sitk.sitkFloat32 )),dtype=tf.float32)
                    CT_scan = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(CT_scans_id,sitk.sitkFloat32 )),dtype=tf.float32)

                    scan = tf.stack((PET_scan,CT_scan),axis=-1)
                    mask = tf.expand_dims(tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(MASKS_id,sitk.sitkUInt8  )),dtype=tf.uint8  ),axis=-1)
                    yield (scan,mask)
        
        return PET_CT_MASK_generator(data_set_ids)
    
    ####################################################################################################
    
    def PREDICT_MASK(self,data_set_ids,path_predictions,model):
        """
            Perform and save prediction on a data set filenames
            
            data_set_ids : [(PET_id_1,CT_id_1,MASK_id_1),(PET_id_2,CT_id_2,MASK_id_2)...]
        """
        
        # generates folder 
        if not os.path.exists(path_predictions):
            os.makedirs(path_predictions)

        filenames_predicted_masks = []
        n_patients = len(data_set_ids)

        for i,data_set_id in enumerate(data_set_ids):

            # display a loading  bar
            display_loading_bar(iteration=i,length=n_patients,add_char=basename(data_set_id[0])+'    ')

            PET_id,CT_id,Mask_id = data_set_id
            
            # load files and prepare model input
            PET_scan = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(PET_id,sitk.sitkFloat32)),dtype=tf.float32)
            CT_scan  = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(CT_id,sitk.sitkFloat32)),dtype=tf.float32)
            MultimodScan = tf.stack((PET_scan,CT_scan),axis=-1)
            MultimodScan = tf.expand_dims(MultimodScan,axis=0)

            MultimodScan_Dataset = tf.data.Dataset.from_tensor_slices(MultimodScan).batch(1)

            prediction = model.predict(MultimodScan_Dataset)

            prediction = tf.argmax(prediction,axis=-1) # output from a multiclass softmax
            prediction = tf.squeeze(prediction).numpy()

            # conversion in unsigned int 8 to store mask with less memory requirement
            mask = np.asarray(prediction,dtype=np.uint8)

            new_filename = path_predictions+"/pred_"+splitext(basename(Mask_id))[0]+'.nii'
            filenames_predicted_masks.append(new_filename)
            sitk.WriteImage(sitk.GetImageFromArray(mask),new_filename)

        return filenames_predicted_masks 
    
    ####################################################################################################
    
    def POSTPROCESS(self):
        """ not used in training """
        pass
    
    ####################################################################################################
    
    def VISUALISATION_MIP_PREDICTION(self,path_output,data_set_ids,pred_ids,filename=None):
        """
            Generate MIP projection of PET/CT files with its predicted mask

            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """
    
        if filename is None:
                filename = path_output+"/PETCT_MIP_"+time.strftime("%m%d%H%M%S")+".pdf"
        else:
            filename = path_output+'/'+filename

        # generates folder 
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        n_patients = len(data_set_ids)

        transparency = 0.7

        color_CT = plt.cm.gray
        color_PET = plt.cm.plasma
        color_MASK = copy(plt.cm.Greys)
        color_MASK.set_bad('white', 0.0)

        with PdfPages(filename) as pdf:

            ############################ BOXPLOTS GENERATION #############################################

            # get boxplots
            num_class = len(self.labels_numbers)
            # initialize values board
            Conf = np.zeros((num_class,num_class),dtype=np.int32)
            TruePos = []
            GTsum = []
            PRsum = []

            TruePos,GTsum,PRsum = self.compute_metrics(data_set_ids,pred_ids)
            
            accuracy_TAB = TruePos/PRsum
            dice_TAB = (2*TruePos+0.1) / (GTsum+PRsum+0.1)
            
            f = plt.figure(figsize=(15, 10))
            f.suptitle('Metrics evaluation', fontsize=15)

            plt.subplot(121)
            plt.boxplot(accuracy_TAB[:,:],sym='x',whis=5,labels=self.labels_names)
            accuracy_median_tumor = np.median(accuracy_TAB[:,1])
            plt.ylim(0.1)
            plt.title("Accuracy Boxplot : tumor=%5.3f" % accuracy_median_tumor,fontsize=15)

            plt.subplot(122)
            plt.boxplot(dice_TAB[:,:],sym='x',whis=5,labels=self.labels_names)
            dice_median_tumor = np.median(dice_TAB[:,1])
            plt.ylim(0.1)
            plt.title("Dice Boxplot : tumor=%5.3f" % dice_median_tumor,fontsize=15)
            
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            
            ############################# MIP GENERATION ################################################
            
            # loop on files to get MIP visualisation
            for i,(DataSet_id,Pred_id) in enumerate(zip(data_set_ids,pred_ids)):

                display_loading_bar(iteration=i,length=n_patients,add_char=basename(DataSet_id[0])+'    ')

                # load imgs
                PET_id,CT_id,Mask_id = DataSet_id
                PET_scan = sitk.GetArrayFromImage(sitk.ReadImage(PET_id))
                CT_scan = sitk.GetArrayFromImage(sitk.ReadImage(CT_id))
                MASK = sitk.GetArrayFromImage(sitk.ReadImage(Mask_id))
                PRED = sitk.GetArrayFromImage(sitk.ReadImage(Pred_id))

                # for TEP visualisation
                PET_scan = np.where(PET_scan>1.0,1.0,PET_scan)
                PET_scan = np.where(PET_scan<0.0,0.0,PET_scan)

                # for CT visualisation
                CT_scan = np.where(CT_scan>1.0,1.0,CT_scan)
                CT_scan = np.where(CT_scan<0.0,0.0,CT_scan)

                # for correct visualisation
                PET_scan = np.flip(PET_scan,axis=0)
                CT_scan = np.flip(CT_scan,axis=0)
                MASK = np.flip(MASK,axis=0)
                PRED = np.flip(PRED,axis=0)

                # stacked projections           
                PET_scan = np.hstack((np.amax(PET_scan,axis=1),np.amax(PET_scan,axis=2)))
                CT_scan = np.hstack((np.amax(CT_scan,axis=1),np.amax(CT_scan,axis=2)))
                MASK = np.hstack((np.amax(MASK,axis=1),np.amax(MASK,axis=2)))
                PRED = np.hstack((np.amax(PRED,axis=1),np.amax(PRED,axis=2)))

                ##################################################################
                f = plt.figure(figsize=(15, 10))
                f.suptitle(splitext(basename(PET_id))[0], fontsize=15)

                plt.subplot(121)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.imshow(np.where(MASK,0,np.nan),cmap=color_MASK,origin='lower')
                plt.axis('off')
                plt.title('Ground Truth',fontsize=20)

                plt.subplot(122)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.imshow(np.where(PRED,0,np.nan),cmap=color_MASK,origin='lower')
                plt.axis('off')
                plt.title('Prediction',fontsize=20)

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                ##################################################################
    
    
    def compute_metrics(self,data_set_ids,pred_ids):
        """ Called by VISUALISATION_MIP_PREDICTION """
        
        num_class = len(self.labels_names)
        
        TruePositive = []
        Sum_GroundTruth = []
        Sum_Prediction = []
        
        for (_,_,mask_id),pred_id in zip(data_set_ids,pred_ids):
            
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(mask_id,sitk.sitkUInt8))
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_id,sitk.sitkUInt8))
            
            TP = []
            GT = []
            PR = []
            
            TP_mask = ground_truth==prediction
            
            for label in self.labels_numbers:
                
                label_mask = ground_truth==label
                
                TP.append(np.sum(np.logical_and(TP_mask,label_mask)))
                GT.append(np.sum(label_mask))
                PR.append(np.sum(prediction==label))
            
            TruePositive.append(TP)
            Sum_GroundTruth.append(GT)
            Sum_Prediction.append(PR)

        # final conversion of board of values
        TruePositive = np.stack(TruePositive)
        Sum_GroundTruth = np.stack(Sum_GroundTruth)
        Sum_Prediction = np.stack(Sum_Prediction)
        
        return TruePositive,Sum_GroundTruth,Sum_Prediction
    
    ####################################################################################################


class Modality_PREDICTION_PET_CT(object):
    """
        Modality used for prediction on PET/CT images
        Requires a trained model
    
        Functions required:
            - PREPROCESS
            - PREDICT_MASK
            - POSTPROCESS
            - GENERATES_MIP_PREDICTION
    """
    
    # definition des globals
    labels_names = ['Background','Lymphoma',]
    labels_numbers = [0,1]
    number_channels = 2
    
    ####################################################################################################
    
    def PREPROCESS(self,data_set_ids,path_output,
                   output_shape=None,pixel_size=None,
                   resample=True,normalize=True):
        """
            Perform preprocessing and save new datas from a dataset
            
            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """
        n_patients = len(data_set_ids)
    
        preprocessed_data_set_ids = []

        for i,data_set_id in enumerate(data_set_ids):

            # display a loading  bar
            display_loading_bar(iteration=i,length=n_patients,add_char=basename(data_set_id[0])+'     ')

            # load data set
            self.PET_id,self.CT_id = data_set_id
            self.PET_img  = sitk.ReadImage( self.PET_id ,sitk.sitkFloat32)
            self.CT_img   = sitk.ReadImage( self.CT_id  ,sitk.sitkFloat32)

            if normalize:
                self.PREPROCESS_normalize()

            if resample:
                self.PREPROCESS_resample_CT_to_TEP()
                self.PREPROCESS_resample_TEPCT_to_CNN(output_shape[::-1],pixel_size[::-1]) #reorder to [x,y,z]

            # save preprocess data
            new_PET_id = path_output+'/'+splitext(basename(self.PET_id))[0]+'.nii'
            new_CT_id  = path_output+'/'+splitext(basename(self.CT_id))[0]+'.nii'

            preprocessed_data_set_ids.append((new_PET_id,new_CT_id,))

            self.PREPROCESS_save(new_filenames=preprocessed_data_set_ids[i])
        
        # clear
        del self.PET_id, self.CT_id
        del self.PET_img, self.CT_img
        
        return preprocessed_data_set_ids
    
    def PREPROCESS_normalize(self,):
        """ called by PREPROCESS """
        
        # normalization TEP
        self.PET_img = sitk.ShiftScale(self.PET_img,shift=0.0, scale=1./10.)
        # normalization CT
        self.CT_img = sitk.ShiftScale(self.CT_img,shift=1000, scale=1./2000.)
    
    def PREPROCESS_resample_CT_to_TEP(self):
        """ called by PREPROCESS """
        
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter() 
        transformation.SetOutputDirection(self.PET_img.GetDirection())
        transformation.SetOutputOrigin(self.PET_img.GetOrigin())
        transformation.SetOutputSpacing(self.PET_img.GetSpacing())
        transformation.SetSize(self.PET_img.GetSize())
        transformation.SetInterpolator(sitk.sitkBSpline)

        # apply transformations on CT IMG
        self.CT_img = transformation.Execute(self.CT_img)
        
        return None
    
    def PREPROCESS_resample_TEPCT_to_CNN(self,new_shape,new_spacing):
        """ called by PREPROCESS """
        
        # compute transformation parameters
        new_Origin = self.compute_new_Origin(new_shape,new_spacing)
        new_Direction = self.PET_img.GetDirection()
        
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter() 
        transformation.SetOutputDirection(new_Direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(new_spacing)
        transformation.SetSize(new_shape)

        # apply transformations on PET IMG
        transformation.SetInterpolator(sitk.sitkBSpline)
        self.PET_img = transformation.Execute(self.PET_img)
        # apply transformations on CT IMG
        transformation.SetInterpolator(sitk.sitkBSpline)
        self.CT_img = transformation.Execute(self.CT_img)
    
    def compute_new_Origin(self,new_shape,new_spacing):
        """ called by PREPROCESS_resample """
        
        origin = np.asarray(self.PET_img.GetOrigin())
        shape = np.asarray(self.PET_img.GetSize())
        spacing = np.asarray(self.PET_img.GetSpacing())
        new_shape = np.asarray(new_shape)
        new_spacing = np.asarray(new_spacing)
        
        return tuple(origin+0.5*(shape*spacing-new_shape*new_spacing))
    
    def PREPROCESS_data_augmentation(self):
        """ not used in TEP/CT """
        pass
    
    def PREPROCESS_save(self,new_filenames):
        """ called by PREPROCESS """

        sitk.WriteImage(self.PET_img,new_filenames[0])
        sitk.WriteImage(self.CT_img,new_filenames[1])
    
    ####################################################################################################
    
    def PREDICT_MASK(self,data_set_ids,path_predictions,trained_model):
        """
            Perform and save prediction on a data set filenames
            
            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """
        
        # generates folder 
        if not os.path.exists(path_predictions):
            os.makedirs(path_predictions)

        filenames_predicted_masks = []
        n_patients = len(data_set_ids)

        for i,data_set_id in enumerate(data_set_ids):

            # display a loading  bar
            display_loading_bar(iteration=i,length=n_patients,add_char=basename(data_set_id[0])+'    ')

            PET_id,CT_id = data_set_id
            
            # load files and prepare model input
            PET_scan = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(PET_id,sitk.sitkFloat32 )),dtype=tf.float32)
            CT_scan = tf.constant(sitk.GetArrayFromImage(sitk.ReadImage(CT_id,sitk.sitkFloat32 )),dtype=tf.float32)

            scan = tf.stack((PET_scan,CT_scan),axis=-1)
            scan = tf.expand_dims(scan,axis=0)

            scan_dataset = tf.data.Dataset.from_tensor_slices(scan).batch(1)

            prediction = trained_model.predict(scan_dataset)

            prediction = tf.argmax(prediction,axis=-1) # output from a multiclass softmax
            prediction = tf.squeeze(prediction).numpy()

            # conversion in unsigned int 8 to store mask with less memory requirement
            mask = np.asarray(prediction,dtype=np.uint8)

            new_filename = path_predictions+"/pred_"+splitext(basename(data_set_id[0]))[0]+'.nii'
            filenames_predicted_masks.append(new_filename)
            sitk.WriteImage(sitk.GetImageFromArray(mask),new_filename)

        return filenames_predicted_masks 
    
    ####################################################################################################
    
    def POSTPROCESS(self,path_output,
                    data_set_ids,prediction_ids,
                    input_pixel_size,resize=True):
        """
            Perform postprocessing on a prediction mask, based on data its corresponding data set
            
            TODO :
            /!\ by default postprocess to PET size
            
            TODO :
            /!\ add all header parameters when generating postprocess mask
            
            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """

        new_filenames = []
        n_patient = len(data_set_ids)

        for i,(data_set_id,pred_id) in enumerate(zip(data_set_ids,prediction_ids)):

            display_loading_bar(iteration=i,length=n_patient,add_char=basename(data_set_id[0])+'     ')
            
            # load data set
            PET_id,CT_id = data_set_id
            PET_img  = sitk.ReadImage(PET_id,sitk.sitkFloat32)
            
            # gather input parameters
            PET_shape = list(PET_img.GetSize())
            PET_shape.reverse() # axis order is reversed when img->array
            PET_pixelsize = list(PET_img.GetSpacing())
            PET_pixelsize.reverse() # axis order is reversed when img->array

            # ordered as [z,y,x]
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_id,sitk.sitkUInt8))

            if resize:
                prediction = isometric_resample(prediction,
                                                input_pixel_size=input_pixel_size,
                                                output_shape=PET_shape,
                                                output_pixel_size=PET_pixelsize,
                                                interpolation_order=0)

            # save postprocessed data
            new_filename = path_output+'/'+splitext(basename(pred_id))[0]+'.nii'
            # save spacing a.k.a pixel size in the file
            sitk_img = sitk.GetImageFromArray(prediction)
            sitk_img.SetSpacing(PET_pixelsize[::-1])
            sitk.WriteImage(sitk_img,new_filename)

            new_filenames.append(new_filename)
            
        return new_filenames
    
    ####################################################################################################
    
    def GENERATES_MIP_PREDICTION(self,path_output,data_set_ids,pred_ids,filename=None):
        """
            Generate MIP projection of PET/CT files with its predicted mask

            data_set_ids : [(PET_id_1,CT_id_1),(PET_id_2,CT_id_2)...]
        """
    
        if filename is None:
                filename = path_output+"/PETCT_MIP_"+time.strftime("%m%d%H%M%S")+".pdf"
        else:
            filename = path_output+'/'+filename

        # generates folder 
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        n_patients = len(data_set_ids)

        transparency = 0.7

        color_CT = plt.cm.gray
        color_PET = plt.cm.plasma
        color_MASK = copy(plt.cm.Greys)
        color_MASK.set_bad('white', 0.0)

        with PdfPages(filename) as pdf:

            # loop on files to get MIP visualisation
            for i,(DataSet_id,Pred_id) in enumerate(zip(data_set_ids,pred_ids)):

                display_loading_bar(iteration=i,length=n_patients,add_char=basename(DataSet_id[0])+'    ')

                # load imgs
                PET_id,CT_id = DataSet_id
                PET_scan = sitk.GetArrayFromImage(sitk.ReadImage(PET_id))
                CT_scan = sitk.GetArrayFromImage(sitk.ReadImage(CT_id))
                PRED = sitk.GetArrayFromImage(sitk.ReadImage(Pred_id))

                # for TEP visualisation
                PET_scan = np.where(PET_scan>1.0,1.0,PET_scan)
                PET_scan = np.where(PET_scan<0.0,0.0,PET_scan)

                # for CT visualisation
                CT_scan = np.where(CT_scan>1.0,1.0,CT_scan)
                CT_scan = np.where(CT_scan<0.0,0.0,CT_scan)

                # for correct visualisation
                PET_scan = np.flip(PET_scan,axis=0)
                CT_scan = np.flip(CT_scan,axis=0)
                PRED = np.flip(PRED,axis=0)

                # stacked projections           
                PET_scan = np.hstack((np.amax(PET_scan,axis=1),np.amax(PET_scan,axis=2)))
                CT_scan = np.hstack((np.amax(CT_scan,axis=1),np.amax(CT_scan,axis=2)))
                PRED = np.hstack((np.amax(PRED,axis=1),np.amax(PRED,axis=2)))

                ##################################################################
                f = plt.figure(figsize=(15, 10))
                f.suptitle(splitext(basename(PET_id))[0], fontsize=15)

                plt.subplot(121)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.axis('off')
                plt.title('PET/CT',fontsize=20)

                plt.subplot(122)
                plt.imshow(CT_scan,cmap=color_CT,origin='lower')
                plt.imshow(PET_scan,cmap=color_PET,alpha=transparency,origin='lower')
                plt.imshow(np.where(PRED,0,np.nan),cmap=color_MASK,origin='lower')
                plt.axis('off')
                plt.title('Prediction',fontsize=20)

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                ##################################################################
    
    ####################################################################################################
