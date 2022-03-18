import numpy as np
from skimage import draw
import logging
from scipy.ndimage import distance_transform_edt
from src import constants
from juputils import jutils


class Ellipsoid(object):
    def __init__(self, level, radial_colors=True, normalize_output=True):
        self.radial_colors = radial_colors
        self.normalize_output = normalize_output

        self.level = level
        assert level in constants.AUGM["ellipsoid"].keys()
        self.factor_interval = constants.AUGM["ellipsoid"][level]

    def __call__(self, img_patch, marrow_patch, ldims=None, return_log=False):
        """
        Adds a single ellipsoid lesion to a random position in a given bone marrow.
        Collision-proof.
        """
        factor = np.random.uniform(low=self.factor_interval[0],high=self.factor_interval[1])

        #logging.info(f"augmenting in:{str(self)}, factor: {factor} ")

        # choose a random dims if not given
        if ldims is None:
            ldims = np.random.randint(low=2,high=4,size=3)

        # BUGFIX: some marrow masks are not fully contained in the image -> distance transform didnt work properly
        newmarpatch = marrow_patch.copy()
        newmarpatch = jutils.frame_marpatch(newmarpatch)

        #shrink bone marrow mask in order to avoid collisions with the ellisoid
        d=distance_transform_edt(newmarpatch)
        max_clearance = np.max(ldims)
        newmarpatch = np.where(d<=max_clearance,0,marrow_patch)
        newmarpatch[:,:,:max_clearance] = 0
        newmarpatch[:,:,-max_clearance:] = 0
        newmarpatch = np.around(newmarpatch) #BUGFIX: some marrow masks had values 0.999999 instead of 1s
        x_m,y_m,z_m = np.where(newmarpatch==1)

        # choose a random center of the lesion
        ridx=np.random.randint(len(x_m))
        L_center = x_m[ridx],y_m[ridx],z_m[ridx]
        #logging.debug(f"Lcenter: {L_center}")

        # boundaries for insertion of the ellipsoid 3D array
        xmin,xmax = L_center[0] - ldims[0], L_center[0] + ldims[0]
        ymin,ymax = L_center[1] - ldims[1], L_center[1] + ldims[1]
        zmin,zmax = L_center[2] - ldims[2], L_center[2] + ldims[2]

        # draw a 3D ellipsoid, cut its border dims which are useless
        ellipsoid = draw.ellipsoid(*ldims,levelset=self.radial_colors)[1:-1,1:-1,1:-1]

        if self.radial_colors:
            ellipsoid = jutils.normalize_0_1(ellipsoid)
            # for a correct radial color gradient we need to invert the values
            ellipsoid = jutils.invert(ellipsoid)

        # copy parameter image and normalize
        nimgcopy = jutils.normalize_0_1(img_patch)

        # insert the ellipsoid 3D array to the CT image array
        nimgcopy[xmin:xmax+1,ymin:ymax+1,zmin:zmax+1] += ellipsoid*factor

        # adding the ellipsoid could break the 0-1 normality of the nimgcopy -> normalize again
        if self.normalize_output:
            nimgcopy = jutils.normalize_0_1(nimgcopy)


        if return_log:
            unitlog = {}
            unitlog[str(self)] = (L_center,factor)
            return nimgcopy, unitlog
        else:
            return nimgcopy

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

class Noise(object):
    def __init__(self, level, normalize_output=True):
        self.normalize_output = normalize_output
        self.level = level
        assert level in constants.AUGM["noise"].keys()
        self.factor_interval = constants.AUGM["noise"][level]


    def __call__(self, img_patch, marrow_patch, return_log=False):
        """
        Adds a random noise augmentation to a marrow volume of a given patch
        """
        factor = np.random.uniform(low=self.factor_interval[0], high=self.factor_interval[1])
        
        #logging.info(f"augmenting in:{str(self)}, factor: {factor} ")
        unitlog = {}
        if return_log:
            unitlog[str(self)] = factor

        marrow_patch = np.around(marrow_patch)
        nimgpatch = jutils.normalize_0_1(img_patch)

        noise=np.random.rand(*marrow_patch.shape)*factor

        noise_patch = (noise*marrow_patch) + nimgpatch

        if self.normalize_output:
            noise_patch = jutils.normalize_0_1(noise_patch)

        if return_log:
            return noise_patch, unitlog
        else:
            return noise_patch

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

class Augmentor(object):

    def __init__(self, func, odds=None, multiple_augm=True):
        """
        Args:
            func (callable or [callables]): A single (or array of) augmentations from this module.
            odds (int/float or [int/floats]): Probabili(ty/ties) or odds of augmenting each of the funcs. Need not sum up to 1 if viewed as odds.
            multiple_augm (bool): Flag which states if user wants to enable multiple chained augmentations.
                These are then chosen with probabilities given by odds parameter.
                In a case of uniform probs (odds), the distribution is multiple
        """
        self._odds = np.array(np.clip(odds,0,1)) if odds is not None else None
        self._func = np.array(func) if func else None
        self._multiple_augm = multiple_augm
        self._prob = (self._odds / np.sum(self._odds)) if self._odds is not None else None

        if odds is not None and (len(self._odds) != len(self._func)):
            logging.warning(f'Not equal lengths of odds in Augmentor:')
            logging.warning(f'odds:{self._odds.shape},func:{self._func.shape}')
            raise "Augmentor exception"

    def __call__(self, img_patch, mask, return_log=False):

        if self._func is None:
            logging.warning(f"Empty augm. function list!")
            return
                
        marrow_mask = np.where(mask==constants.MASK_MARROW,1,0)
        
        assert len(np.unique(marrow_mask)) == 2
        assert np.min(marrow_mask)==0
        
        np.random.shuffle(self._func)
        f_len = len(self._func) if self._multiple_augm == True else None
        augs = np.random.choice(self._func, size=f_len, replace=self._multiple_augm, p=self._prob)
        augs = list(set(augs))
        #logging.debug(f"f: {self._func}, augs:{augs}")

        log = []
        img_res = img_patch.copy()
        for a in augs:
            if return_log:
                img_res, unitlog = a(img_patch=img_res, marrow_patch=marrow_mask, return_log=return_log)
                log.append( unitlog )
            else:
                img_res = a(img_patch=img_res, marrow_patch=marrow_mask, return_log=return_log)


        #logging.debug(f"augmented patch, fx:{augs}")
        assert np.any(img_res - img_patch) == True

        if return_log:
            return img_res, log
        else:
            return img_res
