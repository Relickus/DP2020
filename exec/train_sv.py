import sys,os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from model import MyNetwork
import argparse
from datetime import datetime
import logging
import torch
import constants
import numpy as np

# Arg parser
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# mandatory argument
parser.add_argument("train_path", type=str, help="Absolute or relative path to directory with CT samples of healthy patients.")
parser.add_argument("patch_pos", type=float, help="Probility of a patch being positive.")

#optional arguments
parser.add_argument("--implants",default=constants.IMPLANTS_PATH, type=str, help="Absolute or relative path to the .npy file with implants array.")
parser.add_argument("--trim_dict",default=constants.TRIMDICT_PATH, type=str, help="Absolute or relative path to the trim dictionary file.")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--log", default="INFO", type=str, help="Degree of logger verbosity {none,info (default),debug}")
parser.add_argument("--out_filename",   default="./SUP/sess_"+ str(datetime.now().strftime("%b%d_%H_%M_%S")),
                                        type=str,
                                        help="Specify filename to save the model.")
parser.add_argument("--print_each_step", default=1, type=int, help="Print training stats each N-th training step.")
parser.add_argument("--device", default=None, type=str, help="Computation device {cuda:0 (default),cpu}.")
parser.add_argument("--train_val", default="5:1", type=str, help="Ratio of train:val data. Default: 5:1")
parser.add_argument("--attach_joint", default=True, type=bool, help="Whether to also process load -1th patch of every femur.")
parser.add_argument("--mask", default=constants.MASK_ALL, type=int, help="Mask {0=nothing,1=bone,2=marrow} when loading data.")
parser.add_argument("--augmentation", default="real", type=str, help="Agresivity of augmentation {weak, mild,real,hard}")
parser.add_argument("--rngseed", default=None, type=int, help="RNG seed (int)")

args = parser.parse_args()

import settings
settings.init(args)

import metrics, augmentations as au
import transformations as tr
from juputils import jds,jutils
from dataset import SVDataset, PatchGenerator, SVDataloader
import saveutils,printer

# arrays for plotting values

TLOSS_EPOCH_ARR=[]
VLOSS_EPOCH_ARR=[]

# -------------------------------------------------------------------------

def validate(net, crit, dl_val):
    
    on_validation_start()

    net.eval()
    avg_loss_epoch = metrics.AVG()
    with torch.no_grad():
        for step, batch in enumerate(dl_val):

            input_b, label_b, _ = batch.tensors

            input_b = input_b.to(settings.ARGS["device"])
            label_b = label_b.to(settings.ARGS["device"])
        
            logging.info( f'curr BS: {label_b.shape}')


            preds_b = net(input_b)

            if preds_b.shape != label_b.shape:
                preds_b=preds_b.view(label_b.shape)

            loss = crit(preds_b, label_b)
            avg_loss_epoch.update(loss.item())
            logging.info(f'Validation step - sup. loss: {loss:.4f}')

                
    on_validation_end(avg_loss_epoch)
    net.train()
    return avg_loss_epoch.item()


def on_validation_end(*metrics):
    printer.on_validation_end(*metrics)

def on_validation_start():
    printer.on_validation_start()

def on_step_end(step, **kwargs):
    printer.on_step_end(step, **kwargs)

def on_epoch_start(epoch, **kwargs):
    printer.on_epoch_start(epoch, **kwargs)

def on_epoch_end(epoch, **kwargs):
    printer.on_epoch_end(epoch, **kwargs)


def train(net, optimizer, crit, dl_train, avg_loss_tot):
    net.train()
    avg_loss_epoch = metrics.AVG()


    for step, batch in enumerate(dl_train):
        
        input_b, label_b, _ = batch.tensors

        input_b = input_b.to(settings.ARGS["device"])
        label_b = label_b.to(settings.ARGS["device"])

        # Forward pass
        preds_b = net(input_b)
        # Compute loss (need to reshape preds to match label batch shape)

        if preds_b.shape != label_b.shape:
            preds_b=preds_b.view(label_b.shape)

        loss = crit(preds_b, label_b)

        logging.info( f'curr BS: {label_b.shape}')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss_tot.update(loss.item())
        avg_loss_epoch.update(loss.item())

        if (step+1) % args.print_each_step == 0:
            logging.info(f'Step:[{step+1}/{len(dl_train)}], Train loss: [curr:{loss.item():.4f}, epoch avg:{avg_loss_epoch.item():.4f}, total avg:{avg_loss_tot.item():.4f}]')

    TLOSS_EPOCH_ARR.append(avg_loss_epoch.item())

def save_best_model(net, val_loss, best_val, epoch):
    # prevents saving useless models in first N epochs
    save_from_epoch = 5

    if best_val is None or val_loss < best_val:
        if epoch > save_from_epoch:
            saveutils.save_session(args.out_filename, net, val_loss, epoch)
            logging.info(f"Model saved with val.loss: {val_loss:.4f}")



def train_validate(net, optimizer, crit, dl_train, dl_val):
    logging.info(f'Training for {settings.ARGS["epochs"]} epochs...')

    train_loss_avg = metrics.AVG()
    best_loss = None
    early_stop = jutils.EarlyStop(patience=20)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=7,factor=0.2,verbose=True)

    for epoch in range(settings.ARGS["epochs"]):
        on_epoch_start(epoch)

        train(net,optimizer,crit,dl_train,train_loss_avg)
        val_loss = validate(net,crit,dl_val)
        VLOSS_EPOCH_ARR.append(val_loss)

        save_best_model(net, val_loss, best_loss, epoch)

        if early_stop(val_loss, best_loss, epoch):
            break

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss

        if scheduler:
            scheduler.step(val_loss)

        logging.info(f'Val. loss:{val_loss}, Best loss:{best_loss}')


def model_to_device(net):
    net = net.to(settings.ARGS["device"])

    logging.info(f'Model on device: {next(net.parameters()).device}')

    if settings.ARGS["device"] == "cuda":
        logging.info(torch.cuda.get_device_name(0))
        logging.info(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0)}')

    return net

def create_datasets(args):

    logging.info("Creating datasets...")

    ds = jds.DS(path=args.train_path, ignore_files_path=args.implants)
    train_pat, val_pat = jutils.split_dataset(ds.patients, args.train_val)
    #test_pat = set(ds.patients) - (set(train_pat).union(set(val_pat)))

    ds_train = jds.DSTrimmed(path=args.train_path,trim_dict_path=args.trim_dict,patients=train_pat,ignore_files_path=args.implants)
    ds_val   = jds.DSTrimmed(path=args.train_path,trim_dict_path=args.trim_dict,patients=val_pat,ignore_files_path=args.implants)
    #ds_test  = jds.DSTrimmed(path=args.train_path,trim_dict_path=args.trim_dict,patients=test_pat,ignore_files_path=args.implants)
    
    aug = au.Augmentor(func=[au.Ellipsoid(level=args.augmentation),au.Noise(level=args.augmentation)])
    trans = [tr.Rotate(),tr.Flip()]

    pgtr = PatchGenerator(ds_train,args.attach_joint)
    pgval=PatchGenerator(ds_val,args.attach_joint)
    #pgtest=PatchGenerator(ds_test,args.attach_joint)
    
    sv = {}
    sv["train"] = SVDataset(pg=pgtr,mask_by=args.mask,patch_prevalence=args.patch_pos,augmentor=aug, transforms=trans)
    sv["val"]   = SVDataset(pg=pgval,mask_by=args.mask,patch_prevalence=args.patch_pos,augmentor=aug, transforms=trans)
    #sv["test"]  = SVDataset(pg=pgtest,mask_by=args.mask,patch_prevalence=args.patch_pos,augmentor=aug, transforms=None)
    
    saveutils.save_split(args.out_filename,{"train":train_pat,"val":val_pat})

    return sv

if __name__ == "__main__":

    # Network setup
    net = MyNetwork()
    net = model_to_device(net)
    # Model optimizer & criterion
    crit = torch.nn.BCELoss().to(settings.ARGS["device"])
    optimizer = torch.optim.Adam(net.parameters())


    # Train/val dataset split
    sv = create_datasets(args)
    logging.info(f'TRAIN|VAL patients lens:{len(sv["train"].pg.ds.patients)}|{len(sv["val"].pg.ds.patients)}')

    # DataLoaders
    dl_train =SVDataloader(dataset=sv["train"],batch_size=args.batch_size)
    dl_val = SVDataloader(dataset=sv["val"],batch_size=args.batch_size)
    #dl_test = SVDataloader(dataset=sv["test"],batch_size=args.batch_size)
    
    train_validate(net, optimizer, crit, dl_train, dl_val)


    # dump values for plotting
    np.array(TLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "tloss_epoch_arr"))
    np.array(VLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "vloss_epoch_arr"))
