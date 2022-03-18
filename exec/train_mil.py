import sys,os
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

import argparse
from datetime import datetime
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import pdb
from src import constants
import numpy as np

# Arg parser
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# mandatory argument
parser.add_argument("train_path", type=str, help="Absolute or relative path to directory with CT samples of healthy patients.")
parser.add_argument("patch_pos", type=float, help="Probility of a patch being positive.")
parser.add_argument("femur_pos", type=float, help="Probability of a femur being positive.")


#optional arguments
parser.add_argument("--implants",default=constants.IMPLANTS_PATH, type=str, help="Absolute or relative path to the .npy file with implants array.")
parser.add_argument("--trim_dict",default=constants.TRIMDICT_PATH, type=str, help="Absolute or relative path to the trim dictionary file.")

parser.add_argument("--train_val", default="5:1", type=str, help="Ratio of train:val data. Default: 6:1")
parser.add_argument("--attach_joint", default=False, type=bool, help="Whether to also process load -1st patch of every femur.")
parser.add_argument("--mask", default=constants.MASK_ALL, type=int, help="Mask {0=nothing,1=bone,2=marrow} when loading data.")
parser.add_argument("--augmentation", default="real", type=str, help="Agresivity of augmentation {weak, mild,real,hard}")

parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--batch_size", default=8, type=int, help="Number of bags (femurs) in a batch (processed before optimizer step).")
parser.add_argument("--train_a", default=False, type=bool, help="Whether to make Nand.A trainable.")
parser.add_argument("--nand_a", default=10, type=int, help="Hyperparam. a for the NoisyAnd layer.")
parser.add_argument("--nand_b", default=None, type=float, help="Initial value of NoisyAnd b hyperparameter.")
parser.add_argument("--clip_b", default=False, type=bool, help="Clips nand.b to [0,1]")
parser.add_argument("--validate_each", default=1, type=int, help="Validate each n-th epoch.")
parser.add_argument("--log", default="INFO", type=str, help="Degree of logger verbosity {none,info (default),debug}")
parser.add_argument("--out_filename",   default="./MIL/sess_"+ str(datetime.now().strftime("%b%d_%H_%M_%S")),
                                        type=str,
                                        help="Specify filename to save the model.")
parser.add_argument("--print_each_step", default=1, type=int, help="Print training stats each N-th training step.")
parser.add_argument("--device", default=None, type=str, help="Computation device {cuda:0,cpu}. If None->automatic best choice.")
parser.add_argument("--early_stop_patience", default=20, type=int, help="Patience of EarlyStopping regularizer.")
parser.add_argument("--scheduler_patience", default=7, type=int, help="Patience of ReduceLROnPlateau scheduler.")
parser.add_argument("--rngseed", default=None, type=int, help="RNG seed (int)")
parser.add_argument("--transfer_from", default=None, type=str, help="Path to directory with saved trained model.")
parser.add_argument("--tensorboard", default=False, type=bool, help="Enables tensorboard runtime statistics.")
args = parser.parse_args()


import settings
settings.init(args)

from src import augmentations as au, transformations as tr
from juputils import jds,jutils
from model import MILNetwork
from dataset import MILDataloader,MILDataset,PatchGenerator
import printer, metrics,saveutils

writer = SummaryWriter()


# arrays for plotting values
NAND_B_EPOCH_ARR=[]
NAND_A_EPOCH_ARR=[]

MIL_TLOSS_EPOCH_ARR=[]
SUP_TLOSS_EPOCH_ARR=[]

MIL_VLOSS_EPOCH_ARR=[]
SUP_VLOSS_EPOCH_ARR=[]

VAL_INFO_ARR=[]

# -------------------------------------------------------------------------


def model_to_device(net):
    net = net.to(settings.ARGS["device"])

    logging.info(f'Model on device: {next(net.parameters()).device}')

    if settings.ARGS["device"] == "cuda":
        logging.info(torch.cuda.get_device_name(0))
        logging.info(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0)}')

    return net

def validate_mil(net, crit, dl_val):
    printer.on_validation_start()
    net.eval()
    val_loss_mil = metrics.AVG(title="over bag labels")
    val_loss_sup = metrics.AVG(title="over patch labels")
    with torch.no_grad():
        for batch_num, batch in enumerate(dl_val):
            # batch of N bags
            for bag_num, bag in enumerate(batch.items):

                input_b = bag.tensor.to(settings.ARGS["device"])
                Y = bag.Y.to(settings.ARGS["device"])
                ys = bag.ys.to(settings.ARGS["device"])

                Y_pred, ys_pred = net(input_b)
                mil_loss = crit(Y_pred, Y)

                val_loss_mil.update(mil_loss.item())

                sup_loss = crit(ys_pred,ys)
                val_loss_sup.update(sup_loss.item())
                
                print(bag, ". Y_pred: ",Y_pred, "milloss: ",mil_loss)
                print("ys,ys_pred: ","\n",ys,"\n",ys_pred)
                print("bag_meta:")
                print(bag.meta)

                logging.info(f'Val step end. MIL val loss: {mil_loss.item():.4f}')
                print("-"*30)

                VAL_INFO_ARR.append((str(bag),Y_pred,ys_pred,mil_loss))


    printer.on_validation_end(val_loss_mil, val_loss_sup)

    net.train()
    return val_loss_mil.item(),val_loss_sup.item()


def train_mil(net, crit, optimizer, dlb_train):

    mil_loss_epoch = metrics.AVG()
    sup_loss_epoch = metrics.AVG()

    for batch_num, batch in enumerate(dlb_train):
        net.train()
        mil_loss, sup_loss = 0,0
        optimizer.zero_grad()
        mil_loss_batch = metrics.AVG()

        for bagnum,bag in enumerate(batch.items):

            input_b = bag.tensor
            Y = bag.Y
            ys = bag.ys

            input_b = input_b.to(settings.ARGS["device"])
            Y = Y.to(settings.ARGS["device"])
            ys = ys.to(settings.ARGS["device"])

            # Forward pass. Return both bag label prediction AND patch label predictions.
            Y_pred, ys_pred = net(input_b)

            logging.debug( f'input: shape: {input_b.shape}, device: {input_b.device}' )
            #logging.debug( f'label: shape: {Y.shape}, device: {Y.device}' )
            #logging.debug( f'preds: shape: {Y_pred.shape}, device: {Y_pred.device}' )

            if ys.shape != ys_pred.shape:
                #logging.warning(f"Reshape called in training: p:{ys_pred.shape},l:{ys.shape}")
                ys_pred=ys_pred.view(ys.shape)

            mil_loss = crit(Y_pred, Y)

            mil_loss_epoch.update(mil_loss.item())
            mil_loss_batch.update(mil_loss.item())

            sup_loss = crit(ys_pred, ys)

            sup_loss_epoch.update(sup_loss.item())

            # Only perform backward -> cumulate gradients
            mil_loss.backward()

            if (bagnum+1) % args.print_each_step == 0:
                logging.info(f'Step:[{batch_num*(bagnum+1)}/{dlb_train.bs * len(dlb_train)}], MIL Train losses|last: {mil_loss.item():.4f}, batch avg:{mil_loss_batch.item():.4f}, epoch avg:{mil_loss_epoch.item():.4f}]')

                logging.debug(f"----------------")
                logging.debug(f"Y|Y_pred: [{Y.item()}|{Y_pred.item():.4f}] -> {mil_loss.item():.6f}")
                logging.debug(f"ys: {ys.data}")
                logging.debug(f"ys_pred: {ys_pred.data}")
                logging.debug(f"epoch sup.loss: {sup_loss.item()}")
                logging.debug(f"nand B|grad: [{net.nand.b.item():.5f}|{net.nand.b.grad.item():.5f}]")
                #logging.debug(f"biases|grad:{net.basenet.fc1.bias.grad.data[:4]}|{net.basenet.fc1.bias.data[:4]}")

                logging.debug(f"----------------")

        # optimizer step after BATCH number of predictions on accumulated gradient
        print("================\nOPTIMIZER STEP\n=====================")
        optimizer.step()
        printer.on_batch_end(batch_num,mil_loss_batch=mil_loss_batch.item(),mil_loss_epoch=mil_loss_epoch.item(),noisyand_b=net.nand.b.item())

    return mil_loss_epoch.item(), sup_loss_epoch.item()

def save_model(net, val_loss, epoch):
    # prevents saving useless models in first N epochs
    saveutils.save_session(args.out_filename, net, val_loss,epoch)
    logging.info(f"Model saved with val.loss: {val_loss:.4f}")


def train_validate(net, optimizer, crit, dl_train, dl_val):
    logging.info(f'Training for {settings.ARGS["epochs"]} epochs...')

    train_loss_avg = metrics.AVG()
    best_loss = 1e4
    #early_stop = jutils.EarlyStop(patience=args.early_stop_patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.scheduler_patience,factor=0.2,verbose=True)

    for epoch in range(settings.ARGS["epochs"]):
        printer.on_epoch_start(epoch)

        # -------------- MIL cycle -----------------
        mil_loss_epoch,sup_loss_epoch=train_mil(net,crit,optimizer,dl_train)
        train_loss_avg.update(mil_loss_epoch)

        NAND_B_EPOCH_ARR.append(net.nand.b.item())
        MIL_TLOSS_EPOCH_ARR.append(mil_loss_epoch)
        SUP_TLOSS_EPOCH_ARR.append(sup_loss_epoch)


        if settings.ARGS["tensorboard"]:
            writer.add_scalar("AVG_train_loss",train_loss_avg.item(),epoch)
            writer.add_scalar("MIL_epoch_loss",mil_loss_epoch,epoch)
            writer.add_scalar("SUP_epoch_loss",sup_loss_epoch,epoch)
            writer.add_scalar("NAND_B", net.nand.b.item(),epoch)
        # ------------------------------------------

        # Validation
        if epoch % args.validate_each == 0:
            val_mil_loss,val_sup_loss = validate_mil(net,crit,dl_val)

            MIL_VLOSS_EPOCH_ARR.append(val_mil_loss)
            SUP_VLOSS_EPOCH_ARR.append(val_sup_loss)

            if settings.ARGS["tensorboard"]:
                writer.add_scalar("Val_sup_loss", val_sup_loss,epoch)
                writer.add_scalar("Val_mil_loss", val_mil_loss,epoch)


            if epoch > constants.SAVE_FROM_EPOCH: 
                if val_mil_loss < best_loss:
                    best_loss = val_mil_loss
                    save_model(net, val_mil_loss, epoch)

                #if early_stop(val_mil_loss, best_loss, epoch):
                #    break

                #if scheduler:
                #    scheduler.step(val_mil_loss)

        printer.on_epoch_end(epoch,mil_loss_epoch=mil_loss_epoch,sup_loss_epoch=sup_loss_epoch,train_loss_avg=train_loss_avg.item())
        logging.info(constants.SEPARATOR)
        logging.info(f"Curr val loss| Best loss:{val_mil_loss}|{best_loss}" )
        logging.info(constants.SEPARATOR)

    # save last model
    save_model(net, val_mil_loss, epoch)

def create_datasets(args):

    logging.info("Creating datasets...")

    ds = jds.DS(path=args.train_path, ignore_files_path=args.implants)
    train_pat, val_pat = jutils.split_dataset(ds.patients, args.train_val)
    #test_pat = set(ds.patients) - (set(train_pat).union(set(val_pat)))


    ds_train = jds.DSTrimmed(path=args.train_path,trim_dict_path=args.trim_dict,patients=train_pat,ignore_files_path=args.implants)
    ds_val   = jds.DSTrimmed(path=args.train_path,trim_dict_path=args.trim_dict,patients=val_pat,ignore_files_path=args.implants)
    
    aug = au.Augmentor(func=[au.Ellipsoid(level=args.augmentation),au.Noise(level=args.augmentation)])
    trans = [tr.Rotate(),tr.Flip()]

    pgtr = PatchGenerator(ds_train,args.attach_joint)
    pgval=PatchGenerator(ds_val,args.attach_joint)
    
    milds = {}
    milds["train"] = MILDataset(pg=pgtr,patch_prevalence=args.patch_pos,femur_prevalence=args.femur_pos,mask_by=args.mask,augmentor=aug, transforms=trans)
    milds["val"]   = MILDataset(pg=pgval,patch_prevalence=args.patch_pos,femur_prevalence=args.femur_pos,mask_by=args.mask,augmentor=aug, transforms=trans)
    
    saveutils.save_split(args.out_filename,{"train":train_pat,"val":val_pat})

    return milds

def load_basenet(saved_model, model_class, model_file="ModelClass.py"):
    import importlib.util

    model_class_file = os.path.join(os.path.dirname(os.path.abspath(saved_model)),model_file)

    if not os.path.exists(model_class_file):
        raise Exception(f"Model class file '{model_class_file}' doesnt exists.")

    spec = importlib.util.spec_from_file_location(model_class, model_class_file)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    net = f.__dict__[model_class]()
    net.load_state_dict(torch.load(saved_model,map_location=settings.ARGS["device"]))
    net = net.to(torch.device(settings.ARGS["device"]))

    return net

if __name__ == "__main__":

    # Network setup
    if args.transfer_from is not None:
        basenet = load_basenet(args.transfer_from,"MyNetwork")
        net = MILNetwork(a=args.nand_a,b_init=args.nand_b, clip_b=args.clip_b,transfer_from_model=basenet, train_a=args.train_a)
    else:
        net = MILNetwork(a=args.nand_a,b_init=args.nand_b, clip_b=args.clip_b,train_a=args.train_a)

    net = model_to_device(net)

    # Model optimizer & criterion
    crit = torch.nn.BCELoss().to(settings.ARGS["device"])
    optimizer = torch.optim.Adam(net.parameters())

    # Train/val dataset split
    milds = create_datasets(args)
    logging.info(f'TRAIN|VAL patients lens:{len(milds["train"].pg.ds.patients)}|{len(milds["val"].pg.ds.patients)}')

    # DataLoaders
    dl_train = MILDataloader(dataset=milds["train"],batch_size=args.batch_size)
    dl_val = MILDataloader(dataset=milds["val"],batch_size=args.batch_size)

    train_validate(net, optimizer, crit, dl_train, dl_val)


    # dump values for plotting
    np.array(NAND_B_EPOCH_ARR).dump(os.path.join(args.out_filename, "nand_b_arr"))
    if args.train_a: np.array(NAND_A_EPOCH_ARR).dump(os.path.join(args.out_filename, "nand_a_arr"))
    np.array(MIL_TLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "mil_tloss_epoch_arr"))
    np.array(MIL_VLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "mil_vloss_epoch_arr"))
    np.array(SUP_TLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "sup_tloss_epoch_arr"))
    np.array(SUP_VLOSS_EPOCH_ARR).dump(os.path.join(args.out_filename, "sup_vloss_epoch_arr"))

    np.array(VAL_INFO_ARR).dump(os.path.join(args.out_filename, "val_info_arr"))


