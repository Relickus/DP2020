import os, re, json
from juputils import jutils as jutils
import nibabel as nib
import numpy as np
import logging
import functools
from juputils import filenames
from src import myexceptions
# TODO: substitute 1 and 2 for constants.LEG_LEFT / _RIGHT

class DS:        
    """
    Dataset loading utilities. Works with file naming convention used in 'femury_extracted_v3' folder.
    
    path: string with rel. or abs. path to data folder. 
    
    !Note: the data under 'path' directory must not be structured into folders.
    """
    def __init__(self,path, patients=None, ignore_files_path=None):        
        self.path=os.path.abspath(path)
        self.data_contents = list(filter(self.valid_file,next(os.walk(os.path.abspath(path)))[2]))
        if patients is not None:
            self.data_contents = list(filter(lambda f: self.patient_of(f) in patients, self.data_contents))

        
        if ignore_files_path is not None:
            ignore_files = self.load(ignore_files_path)
            toremove = []
            
            for dat in self.data_contents:
                for impl in ignore_files:
                    if self.patient_of(impl) == self.patient_of(dat) and self.leg_of(dat) == self.leg_of(impl):
                        toremove.append(dat)        
            
            #print(len(self.data_contents),"-",(len(toremove)))
            self.data_contents = list(set(self.data_contents) - set(toremove))
            #print(len(self.data_contents))
            logging.info(f"Created dataset with {len(self.patients)} patients.")
            assert len(self.patients) == len(set(self.patients))
    
    @property
    def patients(self):
        """
        Return list of IDs of all patients in the dataset.
        """
        p = re.compile(pattern=r'(S[0-9]{2,})_')    
        return list(set([p.search(string=x).group(1) for x in filter(p.search,self.data_contents)]))
    @property
    def boundingboxes(self):
        """
        Return list of all bounding box filenames in the dataset.
        """
        return [bb for bb in self.data_contents if ".pkl" in bb]
        
    @property
    def labelmasks(self):
        """
        Return list of all labelmask filenames in the dataset.
        """
        return [mask for mask in self.data_contents if "labelmask" in mask]
    
    @property
    @functools.lru_cache()
    def femurs(self):
        """
        Return list of all femur filenames in the dataset.
        """
        return [fem for fem in self.data_contents if ".nii.gz" in fem and "image" in fem]
        
    
    def femurs_of(self,patient):
        """
        Return list of all femur filenames belonging to the patient.
        Left leg always first.
        """
        return [filenames.femur_file(patient,1),filenames.femur_file(patient,2)]
    
    def patient_of(self, filename):
        """
        Extract and return ID of the patient from the given filename.
        """
        rp = re.compile(pattern=r'(S[0-9]{2,})_')
        match = rp.search(string=filename)
        if match is None:
            raise myexceptions.BadFilename()
        else:
            return match.group(1)
        
    
    def leg_of(self,filename):
        """
        Extract and return the leg (integer) in the filename.
        """
        if "_1_" in filename:
            return 1
        elif "_2_" in filename:
            return 2
        else:
            raise myexceptions.BadFilename
        
    def bb_of(self,femur):
        """
        Return the bounding box pickle filename related to the femur filename.
        """
        rp = re.compile(pattern=r'(S[0-9]{2,})_')
        patient = rp.search(string=femur).group(1)
        leg = self.leg_of(femur)
        
        return filenames.bb_file(patient,leg)
    
    def mask_of(self,femur):
        """
        Return the labelmask filename related to the femur filename.
        """
        patient = self.patient_of(femur)
        leg = self.leg_of(femur)
         
        return filenames.mask_file(patient,leg)
    
    def load(self,filename):
        """
        Load either niftii (.nii.gz) file or pickle (.pkl) file given by the filename
        """
        path = os.path.join(self.path,filename)
        if ".nii.gz" in filename:
            return nib.load(path).get_fdata()
        elif ".npy" in filename:
            return np.load(filename)
        elif ".pkl" in filename:
            res = np.load(path,allow_pickle=True)
            return jutils.bb2arr(res)
        else:
            return None

    def valid_file(self,filename):
        try:
            self.leg_of(filename)
            self.patient_of(filename)
        except myexceptions.BadFilename:
            return False

        return True
        
    def inspect(self,lidx,femur=None,patient=None,leg=None,bb=True):
        """
        Draw the given layer of the left/right femur belonging to the given patient. 
        Optionally also load and draw its bounding box.
        """

        if femur is not None:
            if patient is not None:
                print("You have to pass patient and leg or a femur,not both. ")
                return
            else:
                patient = self.patient_of(femur)
                leg = self.leg_of(femur)

        elif femur is None:
            if patient is None or leg is None or lidx is None:
                print("You have to pass (patient,leg and lidx) or (femur,lidx). ")
                return 
            else:
                femur=self.femurs_of(patient)[leg-1]
    
    
        bbf=jutils.bb2arr(self.load(self.bb_of(femur)))
        print(bbf.shape,bbf[lidx,:])
        
        bbrect = jutils.get_rect(bbf[lidx,:])
        femur=self.load(femur)
        fig=jutils.imshowdraw(femur[:,:,lidx],bbrect)
        return fig



class DSTrimmed(DS):
    def __init__(self, path, trim_dict_path, patients=None, ignore_files_path=None):
        super().__init__(path, patients, ignore_files_path)

        with open(trim_dict_path,"r") as t:
            self.trim_dict = json.load(t)
        
    def load(self,filename):
        """
        Load either niftii (.nii.gz) file or pickle (.pkl) file given by the filename and trim the result.
        """
        
        res = super().load(filename)
        
        if ".nii.gz" in filename:
            fkey = self.patient_of(filename) + "_"+str(self.leg_of(filename))+"_"
            cl,cr,origlen = self.trim_dict[fkey]
            res = res[:,:,cl:cr]
            return res

        elif ".npy" in filename:
            return res

        elif ".pkl" in filename:
            fkey = self.patient_of(filename) + "_"+str(self.leg_of(filename))+"_"
            cl,cr,origlen = self.trim_dict[fkey]            
            res = res[cl:cr,:]
            return res
        else:
            return None
