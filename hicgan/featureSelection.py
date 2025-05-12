import os
import logging
import argparse

import h5py
import tensorflow as tf

from hicrep.utils import readMcool
from hicrep import hicrepSCC


import itertools
import random
import string
import traceback

from hicgan.training import training, create_data, delete_model_files
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation
from hicgan._version import __version__
import threading


log = logging.getLogger(__name__)



def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Hi-cGAN Feature Selection")
    parser.add_argument("--trainingMatrices", "-tm", required=False,
                        type=str, nargs='+',
                        help="mcooler matrices for training.")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=False,
                        type=str, nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromatinFactorFolder", "-tcp", required=False,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--validationMatrices", "-vm", required=False,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=False,
                        type=str, nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromatinFactorFolder", "-vcp", required=False,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    
    parser.add_argument("--originalDataMatrix", "-odm", required=False,
                        type=str,
                        help="Original data matrix for comparison.")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int, choices=[64, 128, 256, 512],
                        default=128,
                        help="window size for submatrices.")
    parser.add_argument("--outputFolder", "-o", required=False,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=False,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int,
                        default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")

    parser.add_argument("--predictionChromatinFactorFolder", "-pcp", required=False,
                        type=str,
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
    parser.add_argument("--binSize", "-b", required=False,
                        type=int,
                        help="Bin size for binning the chromatin features")
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
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

def runTrainingPredictionAndValidation(pArgs, strategy):
    
    trial_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
            pTrainingMatrices=pArgs.trainingMatrices, 
            pTrainingChromosomes=pArgs.trainingChromosomes, 
            pTrainingChromosomesFolders=pArgs.trainingChromosomesFolders, 
            pValidationMatrices=pArgs.validationMatrices, 
            pValidationChromosomes=pArgs.validationChromosomes, 
            pValidationChromosomesFolders=pArgs.validationChromosomesFolders,
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
        pPredictionChromosomesFolders=pArgs.predictionChromosomesFolders,
        pPredictionChromosomes=pArgs.predictionChromosomes,
        pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
        pMultiplier=config["multiplier"],
        pBinSize=pArgs.binSize,
        pBatchSize=config["batch_size"],
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
    except Exception as e:
        traceback.print_exc()
        print(e)
        score = error_return_score
    return score
     

def main(args=None):
    args = parse_arguments()
    # print(args)
    # Read in the folder content of args.trainingChromatinFactorFolder
    training_files = []
    for folder in args.trainingChromatinFactorFolder:
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a valid directory.")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                training_files.append(filepath)

    combined_files = []
    for r in range(2, len(training_files) + 1):
        combined_files += list(itertools.combinations(training_files, r))

    # print("All file combinations:", combined_files)
    # print("Training files found:", training_files)
   
    # log.info("Using single GPU training")
    # log.info("Available GPUs: {}".format(gpu))
    # log.info("Using GPU: {}".format(args.whichGPU-1))
    # log.info("Using GPU: {}".format(gpu[args.whichGPU].name))
    # if args.whichGPU:
    #     if args.whichGPU >= len(gpu):
    #         raise ValueError("Invalid GPU index: {}".format(args.whichGPU - 1))
    #     # strategy = tf.distribute.OneDeviceStrategy(device=gpu[args.whichGPU].name)
    #     strategy = tf.distribute.OneDeviceStrategy(device=f"/GPU:{args.whichGPU-1}")
    
    gpu = tf.config.list_physical_devices('GPU')
    print("combined_files {} ".format(len((combined_files))))
    exit()
    results = {}
    for i, training_files in enumerate(combined_files):
        print(f"Training and prediction with combination {i+1}/{len(combined_files)}: {training_files}\n\n")
        # exit(0)
       

        args.trainingChromatinFactorFolder = new_folder_training
        args.whichGPU = (i % len(gpu))
        def run_and_store_score(files):
            strategy = tf.distribute.OneDeviceStrategy(device=f"/GPU:{args.whichGPU}")
            # score = runTrainingPredictionAndValidation(args, strategy)
            score = args.whichGPU
            results[tuple(files)] = score

        thread = threading.Thread(target=run_and_store_score, args=(training_files,))
        thread.start()
        # results[tuple(training_files)] = "Non-blocking call started"
        # Call the function to run training, prediction, and validation
        # score = runTrainingPredictionAndValidation(args, strategy)
        # results[tuple(training_files)] = score
    
    with open(os.path.join(args.outputFolder, "results.txt"), "w") as outfile:
        for combo, score_val in results.items():
            outfile.write(f"{combo}: {score_val}\n")