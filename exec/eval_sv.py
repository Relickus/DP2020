#!/usr/bin/env python3
import sys,os
sys.path.append(os.path.abspath("."))
import argparse,json,torch,numpy as np

from src import constants

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

import logging
from src import settings
settings._init_logging(testargs,fileoutput=False)

from result import Result
from src import augmentations as au
from juputils import jds
from src.dataset import PatchGenerator,SVDataset,SVDataloader

SESSIONPATH = os.path.dirname(testargs.model)

argsfilepath = testargs.argsfile if testargs.argsfile is not None else os.path.join(SESSIONPATH,"argsfile")
with open(argsfilepath,"r") as af:
    ARGSFILE = json.load(af)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
AUGMENTATION = ARGSFILE["augmentation"]
MASK_BY = ARGSFILE["mask"] if testargs.mask is None else testargs.mask
BATCH_SIZE = 1
PATCH_POS = ARGSFILE["patch_pos"]

if os.path.exists(os.path.join(testargs.dataset_healthy,"train")) or os.path.exists(os.path.join(os.path.dirname(testargs.dataset_healthy),"train")):
    print("\t !!Before testing unbuild the dataset!!")
    quit()

# --------------------------------------------------------------------------------------------------------

def get_test_patients():

    with open(os.path.join(SESSIONPATH,"train_val_patients"),"r") as tvp:
        trainval = json.load(tvp)

    ds = jds.DS(path=testargs.dataset_healthy)

    train = set(trainval["train"])
    val = set(trainval["val"])
    test = set(ds.patients) - (train.union(val))

    logging.info(f"Number of test patients: {len(test)}")

    # assert disjunctive sets
    assert len(train.intersection(val)) == 0
    assert len(train.intersection(test)) == 0
    assert len(test.intersection(val)) == 0

    return test

def init_logging():
    log_numeric = getattr(logging, testargs.log.upper(), logging.CRITICAL)
    form = '%(levelname)s: %(message)s'
    logging.basicConfig(level=log_numeric, format=form,
                        handlers=[logging.StreamHandler()])

TEST_PATIENTS = get_test_patients()

# --------------------------------------------------------------------------------------------------------

def evaluate(net, dl_eval):
    net.eval()
    result = Result()
    with torch.no_grad():
        for step, batch in enumerate(dl_eval):
            input_b, label_b, _ = batch.tensors

            input_b = input_b.to(DEVICE)
            label_b = label_b.to(DEVICE)

            ys_pred = net(input_b)
            result.update(label_b,ys_pred)

            print(f"SV evaluation step [{step}/{len(dl_eval)}]")

    return result


def load_dataset():

    logging.info("Creating testset...")

    ds_test  = jds.DSTrimmed(path=testargs.dataset_healthy,patients=TEST_PATIENTS,trim_dict_path=testargs.trim_dict,ignore_files_path=testargs.implants)
    
    aug = au.Augmentor(func=[au.Ellipsoid(level=AUGMENTATION),au.Noise(level=AUGMENTATION)])
    #trans = [tr.Rotate(),tr.Flip()]

    pgtest=PatchGenerator(ds_test,attach_joint=ARGSFILE["attach_joint"])
    sv = {}
    sv["test"] = SVDataset(pg=pgtest,mask_by=MASK_BY,patch_prevalence=PATCH_POS,augmentor=aug, transforms=None) 
    
    dl_eval =SVDataloader(dataset=sv["test"],batch_size=BATCH_SIZE)

    return dl_eval

def load_model():
    import importlib.util

    model_name = "MyNetwork" 

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
    statsdir = os.path.join(pardir,f"evalstats_p{PATCH_POS}_{AUGMENTATION}")

    if os.path.exists(statsdir):
        raise Exception("Stats directory already exists. Remove it first.")

    if not os.path.exists(modelpath):
        raise Exception('You entered an invalid or nonexistent path to a saved model.')

    if not os.path.exists(datapath):
        raise Exception('You entered an invalid or nonexistent path to directory with healthy samples.')
    else:
        logging.info(f'Data directory entered: {datapath}')

    # load saved model
    net = load_model()
    
    # create an evaluation dataset
    dl_eval = load_dataset()

    logging.info(f'Model device: {net.device}')
    logging.info(f"Num patients for eval: {len(dl_eval.ds.pg.ds.patients)}")
    
    logging.info(f"Num patches: {len(dl_eval.ds)}")
    result = evaluate(net,dl_eval)


    if testargs.saveresult:
        if not os.path.exists(statsdir):
            os.mkdir(statsdir)

            np.array(result._y_pred).dump(os.path.join(statsdir, "_y_pred"))
            np.array(result._y_true).dump(os.path.join(statsdir, "_y_true"))

            roc_curve = np.array(result.roc_curve())
            roc_curve.dump(os.path.join(statsdir, "roc_curve"))

            roc_auc = np.array(result.roc_auc())
            roc_auc.dump(os.path.join(statsdir, "roc_auc"))

            pr_curve = np.array(result.pr_curve())
            pr_curve.dump(os.path.join(statsdir, "pr_curve"))
