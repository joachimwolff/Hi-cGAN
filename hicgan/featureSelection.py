import os
import logging
import argparse
from functools import partial

import h5py
import tensorflow as tf
import numpy as np
from hicrep.utils import readMcool
from hicrep import hicrepSCC


import itertools
import random
import string
import traceback

import ray
from ray import tune
from ray.air import session

import pygenometracks.plotTracks

from hicgan.training import training, create_data, delete_model_files
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation
from hicgan.lib import records
# from .lib import records
from hicgan._version import __version__
import threading

import time
import cooler
from tensorflow.keras.models import load_model
log = logging.getLogger(__name__)

import shap  # SHAP (SHapley Additive exPlanations) for explainability
import matplotlib.pyplot as plt



def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Hi-cGAN Feature Selection")
    parser.add_argument("--trainingMatrices", "-tm", required=False,
                        type=str, nargs='+',
                        help="mcooler matrices for training.")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=False,
                        type=str, nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromatinFolder", "-tcp", required=False,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--validationMatrices", "-vm", required=False,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=False,
                        type=str, nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromatinFolder", "-vcp", required=False,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    parser.add_argument("--originalDataMatrix", "-odm", required=False,
                        type=str,
                        help="Original data matrix for comparison.")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int, choices=[64, 128, 256, 512],
                        default=256,
                        help="window size for submatrices.")
    parser.add_argument("--outputFolder", "-o", required=False,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=False,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int, default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")
    parser.add_argument("--predictionChromatinFolder", "-pcp", required=False,
                        type=str, nargs='+',
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=False,
                        type=str, nargs='+',
                        help="Chromosomes the Hi-C matrix should be predicted. Must be available in all bigwig files")
    parser.add_argument("--trainingCellType", "-tct", required=False,
                        type=str,
                        default="GM12878",
                        help="Cell type for training.")
    parser.add_argument("--validationCellType", "-vct", required=False,
                        type=str,
                        default="GM12878",
                        help="Cell type for validation.")
    parser.add_argument("--testCellType", "-tect", required=False,
                        default="GM12878",
                        type=str,
                        help="Cell type for testing.")
    parser.add_argument("--correlationDepth", "-cd", required=False,
                        type=int,
                        default=1000000,
                        help="Bin size for the Hi-C matrix to compute the correlation.")
    parser.add_argument("--generatorName", "-gn", required=False,
                        type=str,
                        default="generator_00099.keras",
                        help="Name of the generator model file.")
    parser.add_argument("--numberSamples", "-ns", required=False,
                        type=int, default=1,
                        help="Number of samples for the optimizer.")
    parser.add_argument("--iterations", "-it", required=False,
                        type=int, default=10,
                        help="Number of iterations for the optimizer.")
    parser.add_argument("--threads", '-t', required=False,
                        type=int, default=16,
                        help="Number of CPU threads to use.")
    parser.add_argument("--gpu", '-g', required=False,
                        type=int, default=2,
                        help="Number of GPUs to use.")
    parser.add_argument("--continue_experiment", "-ce", required=False,
                        type=str,
                        help="Path to a previous experiment to continue.")
    parser.add_argument("--genomicRegion", "-gr", required=False,
                        type=str,
                        help="Genomic region to plot (e.g., chr1:1000000-2000000).")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int, default=32,
                        help="Batch size for training and prediction.")
    parser.add_argument("--lossWeightPixel", "-lwp", required=False,
                        type=float, default=100.0,
                        help="Loss weight for L1/L2 error of generator.")
    parser.add_argument("--lossWeightDiscriminator", "-lwd", required=False,
                        type=float, default=0.5,
                        help="Loss weight (multiplicator) for the discriminator loss.")
    parser.add_argument("--lossTypePixel", "-ltp", required=False,
                        type=str, choices=["L1", "L2"],
                        default="L1",
                        help="Type of per-pixel loss to use for the generator.")
    parser.add_argument("--lossWeightTV", "-lwt", required=False,
                        type=float, default=1e-10,
                        help="Loss weight for Total-Variation-loss of generator.")
    parser.add_argument("--lossWeightAdversarial", "-lwa", required=False,
                        type=float, default=1.0,
                        help="Loss weight for adversarial loss in generator.")
    parser.add_argument("--learningRateGenerator", "-lrg", required=False,
                        type=float, default=2e-5,
                        help="Learning rate for Adam optimizer of generator.")
    parser.add_argument("--learningRateDiscriminator", "-lrd", required=False,
                        type=float, default=1e-6,
                        help="Learning rate for Adam optimizer of discriminator.")
    parser.add_argument("--beta1", "-b1", required=False,
                        type=float, default=0.5,
                        help="Beta1 parameter for Adam optimizers (gen. and disc.)")
    parser.add_argument("--flipSamples", "-fs", required=False,
                        action='store_true',
                        help="Flip training matrices and chromatin features (data augmentation).")
    parser.add_argument("--figureFileFormat", "-ft", required=False,
                        type=str, choices=["png", "pdf", "svg"],
                        default="png",
                        help="Figure type for all plots.")
    parser.add_argument("--plotFrequency", "-pf", required=False,
                        type=int, default=50,
                        help="Frequency of plotting during training.")
    parser.add_argument("--matrixOutputName", "-mon", required=False,
                        type=str, default="predicted_matrix.cool",
                        help="Name of the output matrix file.")
    parser.add_argument("--parameterOutputFile", "-pof", required=False,
                        type=str, default="parameters.json",
                        help="Name of the output parameter file.")
    parser.add_argument("--multiplier", "-m", required=False,
                        type=float, default=1.0,
                        help="Multiplier for scaling predictions.")
    parser.add_argument("--testChromosomes", "-tc", required=False,
                        type=str, nargs='+',
                        help="Chromosomes to use for testing.")
    parser.add_argument("--numberOfRandomSamples", "-nsr", required=False,
                        type=int, default=0,
                        help="Number of random samples to use.")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

experiment_results = []

def runTrainingPredictionAndValidation(config, pArgs):
    try:
        assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"Ray assigned GPU devices: {assigned_gpus}")

        # From TF's perspective, only the GPUs listed in CUDA_VISIBLE_DEVICES exist.
        physical_gpus = tf.config.list_physical_devices('GPU')
        # print(f"Physical GPUs: {physical_gpus}")
        # exit()
        if physical_gpus:
            # Enable dynamic memory growth for each visible GPU
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # If exactly one GPU is visible, create a OneDeviceStrategy
            if len(physical_gpus) == 1:
                # device_name = physical_gpus[0].name  # e.g. '/physical_device:GPU:0'
                # print(f"Using OneDeviceStrategy on {device_name}")
                device_index = physical_gpus[0].name.split(":")[-1]  # e.g. '0'
                valid_tf_device = f"/device:GPU:{device_index}"

                print(f"Converting {physical_gpus[0].name} -> {valid_tf_device}")

                strategy = tf.distribute.OneDeviceStrategy(device=valid_tf_device)
            else:
                strategy = tf.distribute.MirroredStrategy()


        trial_id = session.get_trial_id()

        trainingFiles = config['input_file'][0]
        validationFiles = config['input_file'][1]
        predictionFiles = config['input_file'][2]
        print("Training and prediction with combination {}: {}".format(trial_id, len(trainingFiles)))
        filenames = [os.path.basename(path) for path in trainingFiles]
        # print("Training Chromatin Folder:", pArgs.trainingChromatinFolder)
        print("Filenames from trainingChromatinFolder: {}".format(filenames))
        # Use cooler to get the bin size from the training matrices
        if not pArgs.trainingMatrices or len(pArgs.trainingMatrices) == 0:
            raise ValueError("No training matrices provided.")
        clr = cooler.Cooler(pArgs.trainingMatrices[0])
        binSize = clr.binsize

        tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
                pTrainingMatrices=pArgs.trainingMatrices, 
                pTrainingChromosomes=pArgs.trainingChromosomes, 
                pTrainingChromatinFolders=trainingFiles, 
                pValidationMatrices=pArgs.validationMatrices, 
                pValidationChromosomes=pArgs.validationChromosomes, 
                pValidationChromatinFolders=validationFiles,
                pWindowSize=pArgs.windowSize,
                pOutputFolder=os.path.join(pArgs.outputFolder,trial_id),
                pBatchSize=pArgs.batchSize,
                pFlipSamples=False,
                pFigureFileFormat="png",
                pRecordSize=pArgs.recordSize
            )

        log.debug("Start training")
        with strategy.scope() as scope:
            training(
                pTfRecordFilenames=tfRecordFilenames,
                pLengthTrainDataContainerList=traindataContainerListLength,
                pWindowSize=pArgs.windowSize,
                pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
                pEpochs=pArgs.epochs,
                pBatchSize=pArgs.batchSize,
                pLossWeightPixel=pArgs.lossWeightPixel,
                pLossWeightDiscriminator=pArgs.lossWeightDiscriminator,
                pLossWeightAdversarial=pArgs.lossWeightAdversarial,
                pLossTypePixel=pArgs.lossTypePixel,
                pLossWeightTV=pArgs.lossWeightTV,
                pLearningRateGenerator=pArgs.learningRateGenerator,
                pLearningRateDiscriminator=pArgs.learningRateDiscriminator,
                pBeta1=pArgs.beta1,
                pFigureFileFormat=pArgs.figureFileFormat,
                pPlotFrequency=pArgs.plotFrequency,
                pScope=scope,
                pStoredFeaturesDict=storedFeatures,
                pNumberSamplesList=nr_samples_list,
                pNumberOfFactors=nr_factors,
                pFlipSamples=pArgs.flipSamples,
                pRecordSize=pArgs.recordSize
            )

        log.debug("Start prediction")
        if not os.path.exists(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName)):
            with h5py.File(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), "w") as f:
                # Optionally, initialize any groups or datasets if necessary.
                # For example: f.create_group("bins")
                pass  # For now, we're just creating an empty file.
        prediction(
            pTrainedModel=os.path.join(
                pArgs.outputFolder, trial_id, pArgs.generatorName),
            pPredictionChromatinFolders=predictionFiles,
            pPredictionChromosomes=pArgs.predictionChromosomes,
            pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
            pMultiplier=pArgs.multiplier,
            pBinSize=binSize,
            pBatchSize=pArgs.batchSize,
            pWindowSize=pArgs.windowSize,
            pMatrixOutputName=pArgs.matrixOutputName,
            pParameterOutputFile=pArgs.parameterOutputFile
        )

        try:
            log.debug("Compute hicrep")
            # activate_lock_or_wait(lock_file_hicrep_path, method="hicrep")
            
            cool1, binSize1 = readMcool(os.path.join(
            pArgs.outputFolder, trial_id, pArgs.matrixOutputName), -1)
            cool2, binSize2 = readMcool(pArgs.originalDataMatrix, -1)

            # smoothing window half-size
            h = 5

            # maximal genomic distance to include in the calculation
            dBPMax = 1000000

            # whether to perform down-sampling or not
            # if set True, it will bootstrap the data set # with larger contact counts to
            # the same number of contacts as in the other data set; otherwise, the contact
            # matrices will be normalized by the respective total number of contacts
            bDownSample = False

            # Optionally you can get SCC score from a subset of chromosomes
            sccSub = hicrepSCC(cool1, cool2, h, dBPMax,
                            bDownSample, pArgs.testChromosomes)
            # removeLock(lock_file_hicrep_path)
            score =  np.mean(sccSub)
            log.debug("SCC score: {:.4f}".format(score))
            # Save the score to a file
            with open(os.path.join(pArgs.outputFolder, trial_id, "scc_score.txt"), "w") as f:
                f.write(f"SCC score: {score:.4f}\n")
            # Save the score to a JSON file
        except Exception as e:
            traceback.print_exc()
            print(e)
            score = -1
        
        matrixOutputNameWithoutExt = os.path.splitext(pArgs.matrixOutputName)[0]
        if pArgs.genomicRegion:
            log.debug("Plot tracks")
                
            score_text = str(score)
            os.makedirs(os.path.join(pArgs.outputFolder, "scores_txt"), exist_ok=True)
            score_file_path = os.path.join(pArgs.outputFolder, "scores_txt", trial_id + '_' + matrixOutputNameWithoutExt + "_score_summary.txt")

            with open(score_file_path, 'w') as score_file:
                score_file.write(score_text)
            
            score_text = score_text.replace("\n", "; ")
            browser_tracks_with_hic = """
[hic matrix]
file = {0}
title = {2}
depth = {4}
transform = log1p
file_type = hic_matrix
show_masked_bins = false

[spacer]
height = 1

[hic matrix]
file = {1}
title = original matrix {3}
depth = {4}
transform = log1p
file_type = hic_matrix
show_masked_bins = false
orientation = inverted
            """.format(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pArgs.originalDataMatrix, score_text, pArgs.trainingCellType, 2000000)
                

            tracks_path = os.path.join(
                pArgs.outputFolder, "browser_tracks_hic.ini")
            with open(tracks_path, 'w') as fh:
                fh.write(browser_tracks_with_hic)

            outfile = os.path.join(
                pArgs.outputFolder, "pygenometracks", trial_id + '_' + matrixOutputNameWithoutExt + ".pdf")

            arguments = f"--tracks {tracks_path} --region {pArgs.genomicRegion} "\
                        f"--outFileName {outfile} --trackLabelFraction 0.1 --width 25 --height 15".split()
            try:
                pygenometracks.plotTracks.main(arguments)
            except Exception as e:
                traceback.print_exc()
                print(e)
        delete_model_files(pTFRecordFiles=tfRecordFilenames)

    except tf.errors.OpError as e:
        log.error("TensorFlow OpError caught")
        # tf.errors.OpError is a common superclass for many TF errors
        traceback_str = traceback.format_exc()
        # Re-raise as a generic Python exception with the original traceback
        raise RuntimeError(
            f"TensorFlow OpError caught. Original traceback:\n{traceback_str}"
        ) from e
    experiment_results.append({
        "input_file": config["input_file"],
        "score": score,
        "trial_id": trial_id
    })
    return score
     

def main(args=None):
    args = parse_arguments()
    os.makedirs(args.outputFolder, exist_ok=True)
    os.makedirs(os.path.join(args.outputFolder, "pygenometracks"), exist_ok=True)
    # print(args)
    # Read in the folder content of args.trainingChromatinFolder
    training_files = []
    validation_files = []
    prediction_files = []

    for folder in args.trainingChromatinFolder:
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a valid directory.")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                training_files.append(filepath)
    for folder in args.validationChromatinFolder:
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a valid directory.")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                validation_files.append(filepath)
    for folder in args.predictionChromatinFolder:
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a valid directory.")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                prediction_files.append(filepath)

    if not (len(training_files) == len(validation_files) == len(prediction_files)):
        raise ValueError("The number of training, validation, and prediction files must be the same.")
    
    combined_files_training = []
    combined_files_validation = []
    combined_files_prediction = []
    for i in range(len(training_files)):
        combined_files_training += [[training_files[i]]]
    
    for i in range(len(validation_files)):
        combined_files_validation += [[validation_files[i]]]

    for i in range(len(prediction_files)):
        combined_files_prediction += [[prediction_files[i]]]

    combined_files_all_training = []
    combined_files_all_validation = []
    combined_files_all_prediction = []
    for r in range(2, len(training_files) + 1):
        # combined_files += list(itertools.combinations(training_files, r))
        combined_files_all_training += list(itertools.combinations(training_files, r))
        combined_files_all_validation += list(itertools.combinations(validation_files, r))
        combined_files_all_prediction += list(itertools.combinations(prediction_files, r))

    if args.numberOfRandomSamples is not None and args.numberOfRandomSamples > 0:

        random.seed(args.seed)
        random_indices = random.sample(
            range(len(combined_files_all_training)),
            min(args.numberOfRandomSamples, len(combined_files_all_training))
        )

        print("Randomly selected indices for training, validation, and prediction files: {}".format(random_indices))
        for i in random_indices:
            print("Training files: {}".format(len(combined_files_all_training[i])))
            print("Validation files: {}".format(len(combined_files_all_validation[i])))
            print("Prediction files: {}".format(len(combined_files_all_prediction[i])))
        combined_files_all_training = [combined_files_all_training[i] for i in random_indices]
        combined_files_all_validation = [combined_files_all_validation[i] for i in random_indices]
        combined_files_all_prediction = [combined_files_all_prediction[i] for i in random_indices]
    
        combined_files_training.extend(combined_files_all_training)
        combined_files_validation.extend(combined_files_all_validation)
        combined_files_prediction.extend(combined_files_all_prediction)
    
    
    if not any(len(element) == len(training_files) for element in combined_files_training):
        combined_files_training.extend(training_files)
        combined_files_validation.extend(validation_files)
        combined_files_prediction.extend(prediction_files)
    
    print("Number of file combinations: {}".format(len(combined_files_training)))

    file_triplets = list(zip(combined_files_training, combined_files_validation, combined_files_prediction))
    metric = 'accuracy'
    mode = 'max'
    run_with_fixed_params = partial(runTrainingPredictionAndValidation, pArgs=args)

    search_space = {
    'input_file': tune.grid_search(file_triplets)  # Use grid search to select input files
    }
    objective_with_resources = tune.with_resources(run_with_fixed_params, resources={"cpu": args.threads, "gpu": args.gpu})

        # Initialize Ray
    ray.init(ignore_reinit_error=True)

    #    Run the experiment
    analysis = tune.run(
        objective_with_resources,  # Run the function with fixed parameters and allocated resources
        config=search_space,
    )

    with open(os.path.join(args.outputFolder, "results.txt"), "w") as result_file:
        for result in experiment_results:
            result_file.write(f"Input File: {result['input_file']} | Score: {result['score']} | Trailid: {result['trail_id']}\n")

    # Shutdown Ray when finished
    ray.shutdown()
