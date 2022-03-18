import shutil,os

def abspath(value):
    return os.path.abspath(value)

def run(dst, traindir, testdir):
    if os.path.exists(traindir):
        files = os.listdir(traindir)
        for f in files:
            shutil.move(os.path.join(traindir,f),os.path.join(dst,f))
        os.removedirs(traindir)


    if os.path.exists(testdir):
        files = os.listdir(testdir)
        for f in files:
            shutil.move(os.path.join(testdir,f),os.path.join(dst,f))
        os.removedirs(testdir)

if __name__ == "__main__":
    import argparse
    # Arg parser
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # mandatory argument
    parser.add_argument("datadir", type=abspath, help="Absolute or relative path to directory with CT samples of healthy patients.")
    #optional arguments
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    traindir = os.path.abspath(os.path.join(args.datadir,"train"))
    testdir = os.path.abspath(os.path.join(args.datadir,"test"))
    run(args.datadir, traindir, testdir)
