# PLOTTING/DRAWING
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import numpy as np
from src import constants
import logging

def plot_intensities(intarr,hline_at=None,vertlines=None, title=None,titlepre="",titlesuf=""):
    """
    Plot maximal intensities of each layer of a femur file f.
    Optionally draw a horizontal line at a given level.
    """
    
    plt.scatter(np.arange(len(intarr)),intarr)
    
    if hline_at is not None:
        plt.axhline(hline_at,label=str(hline_at),c="r")
        
        
    if vertlines is not None:
        for vertline in vertlines:
            if vertline < 0:
                vertline = len(intarr)+vertline
            plt.axvline(x=vertline,c="r")
            
    plt.ylabel("Layer maximum HU")
    plt.xlabel("Layer idx")

    
    with np.printoptions(precision=2):
        if title is None:
            title = titlepre
            title += "\nMedian intensity: {0:.3f}".format(np.median(intarr))
            title += "\n"+titlesuf
            title = title.strip()
        
        plt.title(title)
        
    plt.legend()
    plt.show()
    

def plot_bbs(bbs,hline_at=None,vertlines=None, title=None,titlepre="",titlesuf=""):
    """
    Plot BBoxes with optional horizontal and vertical lines.
    
    bbs: dict/result of bb2arr
    line_at: int/float
    vertlines: iterable
    """
        
    if type(bbs)==dict:
        bbs=bb2arr(bbs)
    
    stats = bb_stats(bbs)
    plt.plot(np.arange(bbs.shape[0]),bbs[:,2],label="Width")
    plt.plot(np.arange(bbs.shape[0]),bbs[:,3],label="Height")
    plt.plot(np.arange(bbs.shape[0]),stats["Sqrtarea"],label="Sqrt BB area")
    
    if hline_at is not None:
        label = "{0:.3f}".format(hline_at)
        plt.axhline(hline_at,label=label,c="r")
    
    if vertlines is not None:
        for vertline in vertlines:
            if vertline < 0:
                vertline = len(bbs)+vertline
            plt.axvline(x=vertline,c="r")
            
    plt.ylabel("BB dimension")
    plt.xlabel("Layer idx")

    
    with np.printoptions(precision=2):
        if title is None:
            title = titlepre
            title += "\nMedian Sqrtarea: {0:.3f}".format(np.median(stats["Sqrtarea"]))
            title += "\n"+titlesuf
            title = title.strip()
        
        plt.title(title)    
    
    plt.legend()
    plt.show()
    
        
        
# ----------------------------------------------------------------------------------

def imshowdraw(img, *patches):
    """
    Draw img, optinally overlay it with drawings (=matplotlib.patches objects) given in arguments.
    """
    fig,ax = plt.subplots(1)
    ax.imshow(img,cmap="gray")
   
    for ptch in patches:
        ax.add_patch(ptch)

    return fig
        
def get_rect(bb,color="r",alpha=1,style="-"):
    """
    Create rectangle patch given by BB dimensions. 
    """
    y,x,h,w=bb
    return pch.Rectangle(xy=(x,y),width=w,height=h,linewidth=1,edgecolor=color,facecolor='none',alpha=alpha,linestyle=style)

def get_circ(xmid,ymid,radius=0.5):
    """
    Create circle patch given by center x,y and radius.
    Useful for drawing single points, i.e. to draw the middle of a BB.
    """
    return pch.Circle(xy=(xmid,ymid),radius=radius,linewidth=2,edgecolor='y',facecolor='none')

def show_slices(img3d,slice_indeces,saveas=None):
    
    r = max(1, (np.ceil(len(slice_indeces)/3).astype(int)))
    c = min(3, len(slice_indeces))

    h=5*r
    w=6*min(len(slice_indeces),3)

    #testutils.figsize(w,h)
    fig, axes = plt.subplots( r, c ,figsize=(w,h))
    plt.subplots_adjust(hspace=0.1)

    i=0
    for lidx in slice_indeces:
        slc = img3d[:,:,lidx]
        if len(slice_indeces) > 3:
            axes[i//3,i%3].imshow(slc,cmap="gray")
        elif len(slice_indeces) > 1:
            axes[i].imshow(slc,cmap="gray")
        else:
            axes.imshow(slc,cmap="gray")
        i+=1

    if saveas is not None:
        fig.savefig(saveas)

    plt.show()


def cropshow(patch,bbp,W,H,clipvalues=True):
    # FOCUS
    lmid = bbp.shape[0]//2
    bbpm = bbp[lmid,:]

    # GEOMETRY
    rm = get_rect(bbpm)
    circ = get_circ(*get_midpoint(bbpm))

    # CROP
    pcrop = crop_patch(patch,bbp,W,H)
    bbmax = adjust_bb(patch,bbp,W,H)
    rmax = get_rect(bbmax)
    imshowdraw(patch[:,:,lmid],rm,rmax,circ)

    if clipvalues:
        pcrop = clip_values(pcrop, constants.CLIP_MIN,constants.CLIP_MAX)

    imshowdraw(pcrop[:,:,lmid])
    
    return pcrop

# =================================================================
# OTHER FUNCTIONS

def bb_stats(bbs):
    """
    Return structured statistics about the given bounding boxes.
    
    bbs: dict or result of bb2arr func.
    """
    if type(bbs)==dict:
        bbs=bb2arr(bbs)
    
    bbsqrtarea = np.sqrt(bbs[:,2]*bbs[:,3])     
    medianw,medianh=np.median(bbs[:,2]),np.median(bbs[:,3])
    meanw,meanh=np.mean(bbs[:,2]),np.mean(bbs[:,3])
    maxw,maxh,minw,minh = bbs[:,2].max(),bbs[:,3].max(),bbs[:,2].min(),bbs[:,3].min()
    
    return {"Sqrtarea":bbsqrtarea,"Median":{"W":medianw,"H":medianh},"Mean":{"W":meanw,"H":meanh},"Max":{"W":maxw,"H":maxh},"Min":{"W":minw,"H":minh}}
    
    
def bb2arr(bb_dict):
    """
    Transform BB dictionary to numpy array.
    """
    if type(bb_dict) == np.ndarray or type(bb_dict)==list:
        return bb_dict
    elif type(bb_dict) == dict:
        return np.array(list(bb_dict.values())).squeeze()

    raise Exception(f"Bad bb_dict argument of type {type(bb_dict)}")

def get_midpoint(bb):
    """
    Return the coordinates of middle point of the given BB.
    """
    y,x,h,w=bb
    mid = x + (w//2), y + (h//2)
    return mid


def patch_gen(data,attach_joint, PATCH_D, PATCH_OVERLAY):
    """
    Yield patches of depth PATCH_D drawn from data file (e.g. femur or labelmask).
    """
    p_start=0
    assert np.argmax(data.shape)==2
    
    while (p_start + PATCH_D) <= data.shape[2]:
        patch = data[:, :,p_start : p_start+PATCH_D]
        yield patch, p_start
        p_start = p_start + PATCH_D - PATCH_OVERLAY

    if attach_joint:
        # add the very last patch of the femur
        yield data[:,:,-PATCH_D:], data.shape[2]-PATCH_D
        
def get_intensities(data,func):
    intarr=[]
    
    for lidx in range(data.shape[2]):
        intarr.append(func(data[:,:,lidx]))
    return intarr

def printprog(curr,total,granularity=5):
    if ((100*curr)//total)%granularity==0:
        print("\rDone {0:.0f}%".format(100*curr/total),end="")
        

def maxtracker():
    import math
    n = math.inf*-1
    def update(val=math.inf*-1):
        nonlocal n
        if val > n:
            n=val
        return n
    return update


# =================================================================
# Trimming and implant utilities

def trim_indeces(f,limit_broken,overrun_spikes):
    # go from the middle towards one end
    slc_idx = f.shape[2]//2

    while not limit_broken(f,slc_idx,"left",overrun_spikes):
        slc_idx -= 1
        if slc_idx<=0:
            break
    cutleftidx = slc_idx

    # go from the middle towards the second end
    slc_idx = f.shape[2]//2
    while not limit_broken(f,slc_idx,"right",overrun_spikes):
        slc_idx += 1
        if slc_idx>=f.shape[2]:
            break
            
    cutrightidx = slc_idx
    
    return cutleftidx,cutrightidx


def find_implants(ds,overrun_spikes,print_progress,LOOKAHEAD):
    """
    Robust method to find bone implants only (with overrun_spikes=False) or all implants (otherwise)
    """
    ifemurs=[]
    maxval = 2500

    for i,femur in enumerate(ds.femurs):
        f=ds.load(femur)
        
        cl,cr=trim_indeces(f,intensity_limit(maxval,LOOKAHEAD=LOOKAHEAD),overrun_spikes=overrun_spikes)
        if cl != 0 or cr != f.shape[2]:
            ifemurs.append(femur)
        
        if print_progress:
            printprog(i,len(ds.femurs))
        
        
    return ifemurs


# functional closures as trimming criteria

def sqrtarea_limit(maxvalue,stats,LOOKAHEAD):
    def fun(f,lidx,direction,overrun_spikes):
        nonlocal maxvalue, stats, LOOKAHEAD
        broken=stats["Sqrtarea"][lidx] > maxvalue

        if not broken:
            return False

        if not overrun_spikes:
            return broken

        #look behind spike
        if direction=="left":
            lookaheadidx = max(lidx-LOOKAHEAD,0)
        elif direction=="right":
            lookaheadidx = min(lidx+LOOKAHEAD,f.shape[2]-1)

        broken=stats["Sqrtarea"][lookaheadidx] > maxvalue
        return broken
    
    return fun

def intensity_limit(maxvalue,LOOKAHEAD):
    def fun(f,lidx,direction,overrun_spikes):

        nonlocal maxvalue, LOOKAHEAD
        maxint = f[:,:,lidx].max()
        broken = maxint > maxvalue

        if not broken:
            return False

        if not overrun_spikes:
            return broken

        if direction=="left":
            lookaheadidx = max(lidx-LOOKAHEAD,0)
        elif direction=="right":
            lookaheadidx = min(lidx+LOOKAHEAD,f.shape[2]-1)

        broken = f[:,:,lookaheadidx].max() > maxvalue
        return broken
    
    return fun

def get_bb(midx,midy,h,w):
    """
    Get BBox in the standart format from its middle point and dims.
    """
    x=midx - w//2
    y=midy - h//2
    return y,x,h,w

def adjust_bb(patch,bbp,cropw,croph):
    midpoint = get_midpoint(bbp[bbp.shape[0]//2,:])
    bbcrop = list(get_bb(*midpoint,croph,cropw))

    maxy,maxx=patch.shape[:2]

    if maxy < cropw or maxx < croph:
        raise Exception("Picture too small for crop of W,H")
        return

    bbcrop[0]=max(bbcrop[0],0)
    bbcrop[1]=max(bbcrop[1],0)

    bbcrop[0]=min(bbcrop[0],maxy)        
    bbcrop[1]=min(bbcrop[1],maxx)

    if bbcrop[0]+bbcrop[2] > maxy:
        bbcrop[0] = maxy - croph

    if bbcrop[1]+bbcrop[3] > maxx:
        bbcrop[1] = maxx - cropw

    return bbcrop
    
def crop_patch(patch,bbp,cropw,croph):
    """
    Crop out a patch given a cropping BBox. 
    """
    bbcrop = adjust_bb(patch,bbp,cropw,croph)
    patch_crop = patch[bbcrop[0]:bbcrop[0]+bbcrop[2],bbcrop[1]:bbcrop[1]+bbcrop[3],:] 
    return patch_crop
    
def clip_values(x, clipmin,clipmax):
    """
    Clear air and implant residues by limiting values.
    """
    #logging.debug(f"CLIP AIR {x.min()},  -> {max(clipmin,x.min())}")
    #logging.debug(f"CLIP MAX {x.max()},  -> {min(clipmax,x.max())}")

    x = np.clip(x,clipmin,clipmax)
    return x

def get_rel_pos(pstart,flen):
    """
    Mean relative pos. between patch start and patch end.
    """
    return (pstart + pstart + constants.PATCH_D)/(2*flen)

def mask_patch(patch, mask, mask_indeces=[]):
    
    mcopy = np.copy(mask)    
    for mi in mask_indeces:
        mcopy[mcopy==mi] = 99
        
    maskedimg = np.where(mcopy==99,patch,0)
    mindeces = np.where(mcopy==99)
    return maskedimg, mindeces
        

def normalize_0_1(img):
    return (img - img.min()) / (img.max() - img.min())

def invert(img):
    return np.abs(img - img.max())

def frame_marpatch(marpatch):
    """
    Creates a 1vx wide frame of non-marrow on the given marrow mask image.
    This fixes broken augmentations on some malformed CT images.
    """
    marpatch[:,:,[-1,0]] = 0
    marpatch[:,[-1,0],:] = 0
    marpatch[[-1,0],:,:] = 0

    return marpatch


def split_dataset(dataset,ratio):
    """
    Split dataset by a given ratio.
    """
    assert len(ratio.split(":"))==2
    r1,r2 = int(ratio.split(":")[0]),int(ratio.split(":")[1])
    unit= len(dataset)/(r1+r2)
    #print(len(dataset),"/",r1+r2,"=",unit)
    set1 = np.random.choice(dataset,size=(np.around(unit*r1)).astype(int),replace=False)
    set2 = set(dataset)-set(set1)
    assert len(set2) != 0, "Split dataset left one set empty!"
        
    return set1,set2


def prob_gen(amin=0,amax=1):
    def fun():
        return np.random.uniform(low=amin,high=amax)
    return fun



class EarlyStop(object):
    def __init__(self, patience):
        self.patience = patience
        self.bad_epochs = 0

    def __call__(self, val_loss, best_val, epoch):
        if best_val is None or val_loss < best_val:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            logging.info(f"Early stopping on epoch {epoch}.")
            return True
        else:
            return False
