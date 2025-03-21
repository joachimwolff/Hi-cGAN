import logging
import os
import warnings
import sys
logging.basicConfig(level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('cooler').setLevel(logging.ERROR)
logging.getLogger('hicmatrix').setLevel(logging.ERROR)

logging.getLogger('numexpr').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
