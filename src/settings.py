import json,torch,logging,os
from src import constants
import numpy as np


ARGS = None
PATH = None

def noise_prob():
	global ARGS

	if ARGS is None:
		load()

	auglvl = ARGS["augmentation"]
	return constants.AUGM["noise"][auglvl]

def ellipsoid_prob():
	global ARGS

	if ARGS is None:
		load()

	auglvl = ARGS["augmentation"]
	return constants.AUGM["ellipsoid"][auglvl]

def init(args):
	_init_logging(args)
	_check_args(args)
	_init_args(args)
	save(args)
	load()

def _init_args(args):
	np.random.seed(args.rngseed)
	args.device = (torch.device(args.device) if args.device else (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))).type	

def save(args):
	global PATH
	PATH = os.path.join(args.out_filename,"argsfile")

	with open(PATH, "w") as af:
		json.dump(args.__dict__,af)

def load(path=None):
	global PATH,ARGS

	if path is not None:
		PATH = path

	if ARGS is None:
		with open(PATH, "r") as af:
			ARGS = json.load(af)

def _init_logging(args, fileoutput=True):
	# extract numeric value from the argument
	log_numeric = getattr(logging, args.log.upper(), logging.CRITICAL)

	form = '%(levelname)s: %(message)s'
	# init the root logging object

	h = [logging.StreamHandler()]

	if fileoutput:
		if not os.path.exists(args.out_filename):
			os.makedirs(args.out_filename)

		h.append(logging.FileHandler(os.path.join(args.out_filename,"debug.log")))

	logging.basicConfig(level=log_numeric, format=form, handlers=h)
	
def _check_datadir(args):
	if not os.path.exists(os.path.abspath(args.train_path)):
		logging.info("You entered an invalid or nonexistent path to directory with healthy samples.")
		raise Exception(f"Invalid data directory entered: {args.train_path}")
	else:
		logging.info(f'Data directory entered: {os.path.abspath(args.train_path)}')
	logging.info(constants.SEPARATOR)

def _check_args(args):
	_check_datadir(args)
# =======================================================
# run on every import
