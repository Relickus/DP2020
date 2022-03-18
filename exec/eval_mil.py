#!/usr/bin/env python3
import logging
import torch
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import constants
import sys, json
sys.path.append(os.path.abspath("."))

# --------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("dataset_healthy", type=str, help="Absolute or relative path to directory of testing dataset.")
parser.add_argument("model", type=str, help="Absolute or relative path to the saved model.")
#parser.add_argument("patch_pos", type=float, help="Probability distribution of positive samples.")

parser.add_argument("--argsfile", default=None, type=str,help="Absolute or relative path to the args file of the experiment which created the tested model.")
parser.add_argument("--implants", default=constants.IMPLANTS_PATH, type=str, help="Absolute or relative path to implants file.")
parser.add_argument("--trim_dict", default=constants.TRIMDICT_PATH,type=str, help="Absolute or relative path to trim dictionary file.")
parser.add_argument("--model_class", type=str, default="ModelClass.py", help="Absolute or relative path to Model.py class or just a name Model.py class (without path if same as model_class path).")
parser.add_argument("--mask", default=None, type=int, help="Mask {0=nothing,1=bone,2=marrow} when loading data.")

parser.add_argument("--log", default="INFO", type=str, help="Degree of logger verbosity {none,info (default),debug}")
parser.add_argument("--saveresult", default=True, type=bool, help="Create directory with pickled results.")
testargs = parser.parse_args()

from result import Result
from src import constants,augmentations as au, transformations as tr
from juputils import jds,jutils
from src.dataset import PatchGenerator,MILDataset,MILDataloader

SESSIONPATH = os.path.dirname(testargs.model)

argsfilepath = testargs.argsfile if testargs.argsfile is not None else os.path.join(SESSIONPATH,"argsfile")
with open(argsfilepath,"r") as af:
    ARGSFILE = json.load(af)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
AUGMENTATION = ARGSFILE["augmentation"]
MASK_BY = ARGSFILE["mask"] if testargs.mask is None else testargs.mask
BATCH_SIZE = 1
PATCH_POS = ARGSFILE["patch_pos"]
FEMUR_POS = ARGSFILE["femur_pos"]

if os.path.exists(os.path.join(testargs.dataset_healthy,"train")) or os.path.exists(os.path.join(os.path.dirname(testargs.dataset_healthy),"train")):
    print("\t !!Before testing unbuild the dataset!!")
    quit()
# ---------------------------------
def get_test_patients():

    with open("train_val_patients","r") as tvp:
        trainvalsplit = json.load(tvp)

    ds = jds.DS(path=testargs.dataset_healthy)

    train = set(trainvalsplit["train"])
    val = set(trainvalsplit["val"])
    test = set(ds.patients) - (train.union(val))

    logging.info(f"Number of test patients: {len(test)}")

    return test

def init_logging():
    log_numeric = getattr(logging, testargs.log.upper(), logging.CRITICAL)
    form = '%(levelname)s: %(message)s'
    logging.basicConfig(level=log_numeric, format=form,
                        handlers=[logging.StreamHandler()])


TEST_PATIENTS = get_test_patients()

#-----------------------------------------------------------------------

def evaluate(net, dl_eval):
    net.eval()
    result1 = Result()
    result2 = Result()
    with torch.no_grad():
        for step, batch in enumerate(dl_eval):
            for bag in batch.items:
                input_b, Y, ys = bag.tensor, bag.Y, bag.ys

                input_b = input_b.to(DEVICE)
                Y = Y.to(DEVICE)
                ys = ys.to(DEVICE)

                Y_pred, ys_pred = net(input_b)
                result1.update(Y,Y_pred)
                result2.update(ys,ys_pred)

            print(f"Evaluation step [{step}/{len(dl_eval)}]")

    return result1, result2

def load_dataset(args):

    logging.info("Creating testset...")

    ds_test  = jds.DSTrimmed(path=testargs.dataset_healthy,patients=TEST_PATIENTS,trim_dict_path=testargs.trim_dict,ignore_files_path=testargs.implants)
    
    aug = au.Augmentor(func=[au.Ellipsoid(level=AUGMENTATION),au.Noise(level=AUGMENTATION)])
    #trans = [tr.Rotate(),tr.Flip()]

    pgtest=PatchGenerator(ds_test,attach_joint=False)
    mil = {}
    mil["test"] = MILDataset(pg=pgtest,mask_by=MASK_BY,patch_prevalence=PATCH_POS,femur_prevalence=FEMUR_POS,augmentor=aug, transforms=None) 
    
    dl_eval =MILDataloader(dataset=mil["test"],batch_size=BATCH_SIZE)

    return dl_eval

def load_model(args):
    import importlib.util

    model_name = "MILNetwork"

    model_class_file = os.path.join(os.path.dirname(os.path.abspath(testargs.model)),testargs.model_class)

    if not os.path.exists(model_class_file):
        raise Exception(f"Model class file '{model_class_file}' doesnt exists.")

    spec = importlib.util.spec_from_file_location(model_name, model_class_file)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    net = f.__dict__[model_name]()
    net.load_state_dict(torch.load(testargs.model,map_location=DEVICE))
    net = net.to(torch.device(DEVICE))

    return net

if __name__ == "__main__":

    modelpath = os.path.abspath(testargs.model)
    datapath = os.path.abspath(testargs.dataset_healthy)

    pardir = os.path.dirname(os.path.abspath(testargs.model))
    statsdir = os.path.join(pardir,f"evalstats_Ppos{PATCH_POS}_Fpos{FEMUR_POS}_A{AUGMENTATION}")

    if os.path.exists(statsdir):
        raise Exception("Stats directory already exists. Remove it first.")

    if not os.path.exists(modelpath):
        raise Exception('You entered an invalid or nonexistent path to a saved model.')

    if not os.path.exists(datapath):
        raise Exception('You entered an invalid or nonexistent path to directory with healthy samples.')
    else:
        logging.info(f'Data directory entered: {datapath}')

    # load saved model
    net = load_model(testargs)
    
    # create an evaluation dataset
    dl_eval = load_dataset(testargs)

    logging.info(f'Model device: {net.device}')
    logging.info(f"Num patients for eval: {len(dl_eval.ds.pg.ds.patients)}")
    
    logging.info(f"Num femurs: {len(dl_eval.ds)}")
    resultmil,resultsup = evaluate(net,dl_eval)

    if testargs.saveresult:
        if not os.path.exists(statsdir):
            os.mkdir(statsdir)

            np.array(resultmil._y_pred).dump(os.path.join(statsdir, "_y_pred_mil"))
            np.array(resultmil._y_true).dump(os.path.join(statsdir, "_y_true_mil"))

            np.array(resultsup._y_pred).dump(os.path.join(statsdir, "_y_pred_sup"))
            np.array(resultsup._y_true).dump(os.path.join(statsdir, "_y_true_sup"))


            roc_curve = np.array(resultmil.roc_curve())
            roc_curve.dump(os.path.join(statsdir, "roc_curve_mil"))

            roc_auc = np.array(resultmil.roc_auc())
            roc_auc.dump(os.path.join(statsdir, "roc_auc_mil"))

            pr_curve = np.array(resultmil.pr_curve())
            pr_curve.dump(os.path.join(statsdir, "pr_curve_mil"))
