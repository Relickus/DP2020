import torch,shutil,logging, os

def save_session(dst, model, metric, epoch):

    if not os.path.isdir(dst):
        os.makedirs(dst)

    # save model
    torch.save(model.state_dict(), f'{dst}/model_val{metric:.3f}_ep:{epoch}' )

    try:
        shutil.copy2("./src/model.py", dst+"/ModelClass.py")
    except Exception:
        logging.exception("Unable to save model class file.")

def save_split(dst, d):

    if not os.path.isdir(dst):
        os.makedirs(dst)
    import json 
    # typecast from ndarray to list
    for k,v in d.items():
        d[k] = list(v)

    with open(os.path.join(dst,"train_val_patients"),"w") as f:
        json.dump(d,f)
