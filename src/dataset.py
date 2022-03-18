import logging, os
import numpy as np
import sys
import torch
from collections import defaultdict

sys.path.append(os.path.abspath("./src"))

from src import constants, myexceptions, transformations as tr
from juputils import jutils

# number of channels of data
C = 1

class Femur:
    def __init__(self,fdata,mdata,bbs,patient,leg):
        assert np.argmax(fdata.shape) == 2
        self.fdata = fdata
        self.mdata = mdata
        self.bbs = bbs
        self.patient = patient
        self.leg = leg
        self.label = constants.LABEL_NEGATIVE

    @property
    def shape(self):
        return self.fdata.shape

class DataPoint:
    @property
    def members(self):
        return self.__dict__.keys()

class Patch(DataPoint):
    def __init__(self, data,y, rel_pos, metadata):
        self.data = data
        self.tensor = data
        self.y = torch.ones(1)*y
        self.rel_pos = torch.ones(1)*rel_pos
        self.meta = metadata
    def __repr__(self):
        return str(self)
    def __str__(self):
        return f'Patch: patient|leg: {self.meta["patient"]}|{self.meta["leg"]}, label:{self.y}, pstart:{self.meta["pstart"]}'

    @property
    def size(self):
        return self.data.shape

class Bag(DataPoint):
    def __init__(self,patches, Y, ys=None, metadata=None):
        self.patches = patches
        self.tensor = torch.stack([p.data for p in self.patches])
        self.Y = torch.ones(1)*Y
        self.ys = torch.from_numpy(np.array(ys)).float()
        self.meta = metadata

    def append(self, patch):
        self.patches.append(patch)

    def __len__(self):
        return len(self.patches)
    def __repr__(self):
        return str(self)
    def __str__(self):
        return f'Bag: patient|leg: {self.meta["patient"]}|{self.meta["leg"]}, label:{self.Y}'

    @property
    def size(self):
        return len(self)

class Batch(DataPoint):
    def __init__(self, items):
        self.items = items
        #self.tensor = torch.stack([p.data for p in items])

    def __len__(self):
        return len(self.items)
    def __repr__(self):
        return str(self)
    def __str__(self):
        res = f"BATCH of size {self.size}:\n"
        return res + "\n".join([str(item) for item in self.items])


    @property
    def size(self):
        return len(self)

    @property
    def tensors(self):
        return patch_collate(self.items)


#=============================================================================


class PatchGenerator:
    def __init__(self, ds, attach_joint):
        self.ds = ds
        self.attach_joint=attach_joint
        self.femuridx = 0
        self.femur = self.load(self.ds.femurs[self.femuridx])
        
        self.numdata = self._calculate_numdata()
        
    def __len__(self):
        return self.numdata
    
    def next_femur(self):
        
        try:
            self.femuridx += 1
            self.femur = self.load(self.ds.femurs[self.femuridx])
        except IndexError:
            raise myexceptions.DatasetExhausted
        except FileNotFoundError:
            raise myexceptions.NoSuchFemur
        except Exception:
            raise Exception

    def reset(self):
        self.femuridx=0
        self.femur = self.load(self.ds.femurs[self.femuridx])
            
    @property
    def femur(self):
        return self._femur
    
    @femur.setter
    def femur(self,f):
        self._femur=f
        self.p_start=0
            
    def load(self, femur):
        return self.load_femur(self.ds.patient_of(femur),self.ds.leg_of(femur))

    def load_femur(self, patient, leg):
    
        try:
            femur = self.ds.femurs_of(patient)[leg-1]
        except IndexError:
            raise myexceptions.NoSuchFemur
        except Exception:
            raise Exception
            
        mask = self.ds.mask_of(femur)
        bbs = self.ds.bb_of(femur)

        f = self.ds.load(femur)
        m = self.ds.load(mask)
        bbs = jutils.bb2arr(self.ds.load(bbs))

        return Femur(f,m,bbs,patient,leg)
    
    def _curr_patch(self):
        p_start = self.p_start - constants.PATCH_D + constants.PATCH_OVERLAY
        fp = self.femur.fdata[:,:, p_start : p_start+constants.PATCH_D]
        return fp
        
    def get_bag(self, idx):
        
        try:
            femur = self.ds.femurs[idx]
            self.femur = self.load(femur)
        except IndexError as e:
            raise e
        print("loaded femur on idx",idx)
        bag = []
        meta = {"patient":self.femur.patient,"leg":self.femur.leg}
        
        while True:
            try:
                inst = self.next_patch()
                bag.append(inst)
            except myexceptions.FemurExhausted:
                break
            except Exception as ex:
                raise ex
                
        return bag, meta
        
    def next_patch(self):
        if self.femur is None:
            raise myexceptions.FemurExhausted
            

        if (self.p_start + constants.PATCH_D) <= self.femur.shape[2]:
            fp = self.femur.fdata[:,:, self.p_start : self.p_start+constants.PATCH_D]
            mp = self.femur.mdata[:,:, self.p_start : self.p_start+constants.PATCH_D]
            bbp =  self.femur.bbs[self.p_start : self.p_start+constants.PATCH_D,:]
            
            old_pstart = self.p_start
            rel_pos = jutils.get_rel_pos(old_pstart,self.femur.shape[2])
            
            self.p_start = self.p_start + constants.PATCH_D - constants.PATCH_OVERLAY

            meta = {"patient":self.femur.patient,"leg":self.femur.leg,"pstart":old_pstart}

            return rel_pos, fp, mp, bbp, meta
        
    
        if self.attach_joint:
            # add the very last patch of the femur
            fp = self.femur.fdata[:,:, -constants.PATCH_D:]
            mp = self.femur.mdata[:,:, -constants.PATCH_D:]
            bbp =  self.femur.bbs[-constants.PATCH_D:,:]
            
            old_pstart = self.femur.shape[2]-constants.PATCH_D
            rel_pos = jutils.get_rel_pos(old_pstart,self.femur.shape[2])
            
            meta = {"patient":self.femur.patient,"leg":self.femur.leg,"pstart":old_pstart}

            self.femur=None
                        
            return rel_pos, fp,mp,bbp, meta
        
        else:
            raise myexceptions.FemurExhausted
            
        
    def _calculate_numdata(self):
        tot_patches = 0
        for femur in self.ds.femurs:
            f = self.ds.load(femur)
            tot_patches += self.get_numpatches(f,self.attach_joint)
        
        self.numdata = tot_patches
        return tot_patches
    
    def get_numpatches(self,femur,attach_joint):
        K=constants.PATCH_D
        W=femur.shape[2]
        S=(constants.PATCH_D-constants.PATCH_OVERLAY)
        res = ((W - K) // S) +1
        if attach_joint:
            res +=1
        return res


class ADataset(torch.utils.data.Dataset):

    def __init__(self, pg, patch_prevalence, mask_by, augmentor, transforms, mode):
        self.pg = pg
        self.augmentor = augmentor
        self.transforms = transforms
        self.mask_by = np.array(constants.MASK_ALL).ravel() if mask_by is None else np.array([mask_by]).ravel()
        self.patch_prevalence = patch_prevalence
        self.mode=mode


    def prepare_patch(self, fp, mp, bbp ):

        log = None
        plabel = constants.LABEL_NEGATIVE
        #crop
        fp = jutils.crop_patch(fp,bbp,constants.PATCH_W,constants.PATCH_H)
        mp = jutils.crop_patch(mp,bbp,constants.PATCH_W,constants.PATCH_H)
        
        #clip air / stents
        fp = jutils.clip_values(fp,constants.CLIP_MIN,constants.CLIP_MAX)
        

        #augment 
        if self.augmentor is not None:
            if np.random.uniform() < self.patch_prevalence:
                fp,log = self.augmentor(fp,mp, return_log=True) 
                plabel = constants.LABEL_POSITIVE
                
        # masking 
        fp,_ = jutils.mask_patch(fp,mp,self.mask_by)
        
        
        #flip/rotate
        if self.transforms is not None:
            for transform in self.transforms:
                fp = transform(fp)
                
        #standardize
        fp = tr.standardize(fp)

        # transform patch to torch.tensor of dims C x D x H x W
        fp = tr.patch2tensor(fp)

        return fp, plabel, log

    def prepare_bag(self, bag, bagmeta, augment_femur):

        bag_patches=[]
        ys = []
        augm_log = defaultdict(lambda: [])
        tr_log = defaultdict(lambda: [])

        for inst_num,insttup in enumerate(bag):
            rel_pos, fp, mp, bbp, pmeta = insttup

            plabel = constants.LABEL_NEGATIVE

            #crop
            fp = jutils.crop_patch(fp,bbp,constants.PATCH_W,constants.PATCH_H)
            mp = jutils.crop_patch(mp,bbp,constants.PATCH_W,constants.PATCH_H)
            
            #clip air / stents
            fp = jutils.clip_values(fp,constants.CLIP_MIN,constants.CLIP_MAX)
            
            
            #augment 
            if self.augmentor is not None:
                if augment_femur:
                    if np.random.uniform() < self.patch_prevalence:
                        fp,log = self.augmentor(fp,mp, return_log=True) 
                        plabel = constants.LABEL_POSITIVE
                        augm_log[inst_num].append(log)
            

            # masking 
            fp,_ = jutils.mask_patch(fp,mp,self.mask_by)
            
            #flip/rotate
            if self.transforms is not None:
                for transform in self.transforms:
                    fp,log = transform(fp,return_log=True)
                    tr_log[inst_num].append(log)
                    
            #standardize
            fp = tr.standardize(fp)

            # transform patch to torch.tensor of dims C x D x H x W
            fp = tr.patch2tensor(fp)


            bag_patches.append(fp)
            ys.append(plabel)

        bagmeta.update(augm_log)
        bagmeta.update(tr_log)

        Y = constants.LABEL_POSITIVE if np.any(ys) else constants.LABEL_NEGATIVE
        bagobj = Bag(bag_patches, Y=Y, ys=ys, metadata=bagmeta)

        return bagobj

class SVDataset(ADataset):
    def __init__(self,pg,patch_prevalence,mask_by,augmentor,transforms,mode="debug"):
        super().__init__(pg, patch_prevalence, mask_by, augmentor,transforms,mode)
        self.numdata = self._calculate_numdata()

    def __len__(self):
        return self.numdata
    
    def __getitem__(self,idx):

        try:
            rel_pos, fp, mp, bbp, meta = self._next_patch()

            fp, plabel, log = self.prepare_patch(fp, mp, bbp)
            meta["log"] = log

            if self.mode == "run":
                return fp, plabel, rel_pos
            elif self.mode == "debug":
                p = Patch(fp,plabel,rel_pos,metadata=meta)
                return p
        except Exception as e:
            raise e

    def start_over(self):
        logging.info("SV dataset start over.")
        self.pg.reset()

    def get_patch(self):

        try:
            rel_pos, fp, mp, bbp, meta = self._next_patch()
            fp, plabel, log = self.prepare_patch(fp, mp, bbp)
            meta["log"] = log
            return Patch(fp,plabel,rel_pos,metadata=meta)

        except Exception as e:
            raise e
    
    def _next_patch(self):
        try:
            return self.pg.next_patch()
        except myexceptions.FemurExhausted:
            try:
                self.pg.next_femur()
                logging.debug("NEW FEMUR")
                return self._next_patch()
            except myexceptions.DatasetExhausted as e:
                raise e
                #self.pg.reset()
                #return self._next_patch()
        

    def _calculate_numdata(self):
        return self.pg.numdata

class MILDataset(ADataset):
    def __init__(self, pg, patch_prevalence, femur_prevalence, mask_by, augmentor, transforms,mode="debug"):
        super().__init__(pg, patch_prevalence, mask_by, augmentor,transforms,mode)
        self.numdata = self._calculate_numdata()
        self.femur_prevalence = femur_prevalence
        
    def _calculate_numdata(self):
        return len(self.pg.ds.femurs)
    
    def __len__(self):
        return self.numdata
    
    def __getitem__(self,idx):
        
        if self.pg.attach_joint:
            print("POZOR, V MIL NECHCES ATTACH JOINT")
        
        try:
            bag, bmeta = self._get_bag(idx)
        except IndexError as iex:
            raise iex
        
        fres,ys = [],[]

        augment_femur = True if np.random.uniform() < self.femur_prevalence else False

        bagobj = self.prepare_bag(bag,bmeta,augment_femur)

        if self.mode == "debug":
            return bagobj

        elif self.mode == "run":
            rel_pos_arr = []

            fres,ys,Y,rel_pos_arr = bagobj.patches,bagobj.ys,bagobj.Y,[tup[0] for tup in bagobj.patches]

            return fres,ys, Y, rel_pos_arr


        # turn fres (list of tensors) to a tensor
        #fres = torch.stack(fres)

        #return fres, y, Y, rel_pos_arr
        
            
    def _get_bag(self, idx):
        try:
            bag, meta = self.pg.get_bag(idx)
            return bag, meta
            
        except IndexError as iex:
            raise iex
        except Exception as e:
            raise e


class MILDataloader:
    def __init__(self,dataset,batch_size):
        self.ds = dataset
        self.bs = batch_size

    def __iter__(self):
        self.bag_idx=0
        self.exhausted = False
        return self

    def __len__(self):
        numbatches = len(self.ds) // self.bs

        # if exact integer division, dont add the extra batch
        if numbatches == int(numbatches):
            return numbatches
        else:
            return numbatches + 1

    def __next__(self):

        if self.exhausted:
            raise StopIteration

        try:
            bag_batch = []

            for _ in range(0,self.bs):
                bag_batch.append(self.ds[self.bag_idx])
                self.bag_idx += 1

            return Batch(bag_batch)

        except IndexError:
            print("DATASET END")
            self.exhausted = True
            if len(bag_batch) == 0:
                raise StopIteration
            else:
                return Batch(bag_batch)


class SVDataloader:
    def __init__(self,dataset,batch_size):
        self.ds = dataset
        self.bs = batch_size

    def __iter__(self):
        self.ds.start_over()
        self.p_idx=0
        self.exhausted = False
        return self

    def __len__(self):
        numbatches = len(self.ds) // self.bs

        # if exact integer division, dont add the extra batch
        if numbatches == int(numbatches):
            return numbatches
        else:
            return numbatches + 1

    def __next__(self):
        if self.exhausted:
            raise StopIteration

        try:
            batch = []

            for _ in range(0,self.bs):
                batch.append(self.ds[self.p_idx])
                self.p_idx += 1

            return Batch(batch)

        except myexceptions.DatasetExhausted:
            print("DATASET END")
            self.exhausted = True
            if len(batch) == 0:
                raise StopIteration
            else:
                return Batch(batch)

def patch_collate(lst):
    data_res = torch.stack([p.data for p in lst])
    y_res = torch.squeeze(torch.stack([p.y for p in lst]))
    rp_res = torch.squeeze(torch.stack([p.rel_pos for p in lst])) 
    return (data_res,y_res,rp_res)
