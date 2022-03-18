import shutil, argparse,os,re
import numpy as np
import sys
sys.path.append(".")

from juputils import jds

def abspath(value):
    return os.path.abspath(value)

# Arg parser
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# mandatory argument
parser.add_argument("datadir", type=abspath, help="Absolute or relative path to directory with CT samples of healthy patients.")
#optional arguments
parser.add_argument("ratio", type=str, help="Ratio of train:test data.")
args = parser.parse_args()
# -------------------------------------------------------------------------

def prepare_dirs(dst, traindir, testdir):
    if os.path.exists(traindir):
        files = os.listdir(traindir)
        for f in files:
            shutil.move(os.path.join(traindir,f),os.path.join(dst,f))
    else:
        os.mkdir(traindir)

    if os.path.exists(testdir):
        files = os.listdir(testdir)
        for f in files:
            shutil.move(os.path.join(testdir,f),os.path.join(dst,f))
    else:
        os.mkdir(testdir)

def ispatient(p):
    re_patient = re.compile(r"S[0-9]{2,}")
    if re_patient.match(p):
        return True
    return False


def move_files(p_train,p_test,traindir,testdir,ds):

    files = ds.data_contents
    train_files,test_files = [],[]

    for f in files:
        for p in p_train:
            if ds.patient_of(f) == p:
                train_files.append(f)


    for f in files:
        for p in p_test:
            if ds.patient_of(f) == p:
                test_files.append(f)

    # assert no intersection between train and test set files
    assert len(set(train_files).intersection(set(test_files))) == 0

    for f in train_files: shutil.move(os.path.join(args.datadir, f),os.path.join(traindir,f))
    for f in test_files: shutil.move(os.path.join(args.datadir, f),os.path.join(testdir,f))



def split_patients(patients, ratio):

    def _parse_train_ratio(ratio):
        if ratio is None or ":" not in ratio:
            raise Exception("Bad ratio argument! See --help.")
        r = np.array(ratio.split(":"),dtype=np.int)
        return r[0] / np.sum(r)

    train_ratio = _parse_train_ratio(ratio)
    train_ratio = np.clip(train_ratio, 0, 1)
    num_train = int(len(patients)*train_ratio)

    p_train = np.random.choice(patients,replace=False,size=num_train)
    p_test = patients if train_ratio==1 else list(set(patients) - set(p_train))

    return p_train, p_test

if __name__ == "__main__":
    traindir = os.path.abspath(os.path.join(args.datadir,"train"))
    testdir = os.path.abspath(os.path.join(args.datadir,"test"))
    prepare_dirs(args.datadir, traindir, testdir)

    ds = jds.DS(path=args.datadir)
    patients = ds.patients
    p_train, p_test = split_patients(patients, args.ratio)

    move_files(p_train,p_test,traindir,testdir,ds)

