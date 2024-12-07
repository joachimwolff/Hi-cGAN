import argparse
import numpy as np
import os
import tensorflow as tf
from .lib import dataContainer
from .lib import records
from .lib import hicGAN
from .lib import utils

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

from hicrep.utils import readMcool
from hicrep import hicrepSCC

import pygenometracks.plotTracks


import logging
from hicgan._version import __version__
from hicgan.training import training, create_data, delete_model_files
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation

log = logging.getLogger(__name__)

import icecream as ic

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--optimizer", "-op", required=True,
                        type=str, choices=['optuna', 'hyperopt'],
                        help="Optimizer to use for training (options: 'optuna', 'hyperopt')")
    parser.add_argument("--trainingMatrices", "-tm", required=True,
                        type=str, nargs='+',
                        help="mcooler matrices for training.")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=True,
                        type=str, nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromosomesFolders", "-tcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--validationMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=True,
                        type=str, nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromosomesFolders", "-vcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    parser.add_argument("--testMatrices", "-tem", required=True,
                        type=str, nargs='+',
                        help="mcooler matrices for test.")
    parser.add_argument("--testChromosomes", "-techroms", required=True,
                        type=str, nargs='+',
                        help="Test chromosomes. Must be present in all train matrices.")
    parser.add_argument("--testChromosomesFolders", "-tecp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for test reside (bigwig files).")
    
    parser.add_argument("--originalDataMatrix", "-odm", required=True,
                        type=str,
                        help="Original data matrix for comparison.")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int, choices=[64, 128, 256, 512],
                        default=128,
                        help="window size for submatrices.")
    parser.add_argument("--outputFolder", "-o", required=True,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=False,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int,
                        default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")
    
    parser.add_argument("--predictionChromosomesFolders", "-pcp", required=True,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=True,
                        type=str, nargs='+',
                        help="Chromosomes the Hi-C matrix should be predicted. Must be available in all bigwig files")
    parser.add_argument("--matrixOutputName", "-mn", required=False,
                        type=str,
                        default="predMatrix.cool",
                        help="Name of the output cool-file")
    parser.add_argument("--parameterOutputFile", "-pf", required=False,
                        type=str,
                        default="predParams.csv",
                        help="Name of the parameter file")
    parser.add_argument("--correlationMethod", "-cm", required=False,
                        default='pearson',
                        type=str, choices=['pearson', 'spearman', 'hicrep'],
                        help="Type of error to compute (options: 'pearson', 'spearman', 'hicrep')")
    parser.add_argument("--errorType", "-et", required=False,
                        default="AUC",
                        type=str, choices=['R2', 'MSE', 'MAE', 'MSLE', 'AUC'],
                        help="Type of error to compute (options: 'R2', 'MSE', 'MAE', 'MSLE', 'AUC')")
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
    parser.add_argument("--binSize", "-b", required=True,
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


def objective(config, pArgs, pTfRecordFilenames=None, pTraindataContainerListLength=None, pNrSamplesList=None, pStoredFeatures=None, pNrFactors=None):

    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        try:
            for gpu_device in gpu:
                tf.config.experimental.set_memory_growth(gpu_device, True)
        except Exception as e:
            print("Error: {}".format(e))
    strategy = tf.distribute.MirroredStrategy()
    
    trial_id = session.get_trial_id()
    print("trail_id {}".format(trial_id))

    os.makedirs(os.path.join(pArgs.outputFolder, trial_id), exist_ok=True)

    
    
    with strategy.scope() as scope: 
        training(
            pTfRecordFilenames=pTfRecordFilenames,
            pLengthTrainDataContainerList=pTraindataContainerListLength,
            pWindowSize=pArgs.windowSize,
            pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
            pEpochs=pArgs.epochs,
            pBatchSize=pArgs.batchSize,
            pLossWeightPixel=config["loss_weight_pixel"],
            pLossWeightDiscriminator=config["loss_weight_discriminator"],
            pLossWeightAdversarial=config["loss_weight_adversarial"],
            pLossTypePixel=config["loss_type_pixel"],
            pLossWeightTV=config["loss_weight_tv"],
            pLearningRateGenerator=config["learning_rate_generator"],
            pLearningRateDiscriminator=config["learning_rate_discriminator"],
            pBeta1=config["beta1"],
            pFigureFileFormat="png",
            pPlotFrequency=20,
            pScope=scope,
            pStoredFeaturesDict=pStoredFeatures,
            pNumberSamplesList=pNrSamplesList,
            pNumberOfFactors=pNrFactors, 
            pFlipSamples=config["flip_samples"],
            pRecordSize=pArgs.recordSize
        )

    prediction(
        pTrainedModel=os.path.join(pArgs.outputFolder, trial_id, pArgs.generatorName),
        pPredictionChromosomesFolders=pArgs.predictionChromosomesFolders,
        pPredictionChromosomes=pArgs.predictionChromosomes,
        pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
        pMultiplier=config["multiplier"],
        pBinSize=pArgs.binSize,
        pBatchSize=pArgs.batchSize,
        pWindowSize=pArgs.windowSize,
        pMatrixOutputName=pArgs.matrixOutputName,
        pParameterOutputFile=pArgs.parameterOutputFile
    )
     
    score = 0

    if pArgs.correlationMethod == 'pearson':
        for chrom in pArgs.testChromosomes:
            score_dataframe = computePearsonCorrelation(pCoolerFile1=os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pCoolerFile2=pArgs.originalDataMatrix,
                                                        pWindowsize_bp=pArgs.correlationDepth, pModelChromList=pArgs.trainingChromosomes, pTargetChromStr=chrom,
                                                        pModelCellLineList=pArgs.trainingCellType, pTargetCellLineStr=pArgs.testCellType,
                                                        pPlotOutputFile=None, pCsvOutputFile=None)
            score += score_dataframe.loc[pArgs.correlationMethod, pArgs.errorType]
        score = score / len(pArgs.testChromosomes)

    elif pArgs.correlationMethod == 'hicrep':
        cool1, binSize1 = readMcool(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), -1)
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
        sccSub = hicrepSCC(cool1, cool2, h, dBPMax, bDownSample, pArgs.testChromosomes)

        score = np.mean(sccSub)

    if pArgs.genomicRegion:
        browser_tracks_with_hic = """
[hic matrix]
file = {0}
title = predicted score {2}
depth = 3000000
transform = log1p
file_type = hic_matrix
show_masked_bins = false

[hic matrix]
file = {1}
title = original matrix {3}
depth = 3000000
transform = log1p
file_type = hic_matrix
show_masked_bins = false
orientation = inverted
""".format(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pArgs.originalDataMatrix, score, pArgs.trainingCellType)
        
        tracks_path = os.path.join(pArgs.outputFolder, trial_id, "browser_tracks_hic.ini")
        with open(tracks_path, 'w') as fh:
            fh.write(browser_tracks_with_hic)

        outfile = os.path.join(pArgs.outputFolder, "pygenometracks",  trial_id + ".pdf")
        
        arguments = f"--tracks {tracks_path} --region {pArgs.genomicRegion} "\
                    f"--outFileName {outfile}".split()
        pygenometracks.plotTracks.main(arguments)
 
    return score

def objective_raytune(config, pArgs, pTfRecordFilenames=None, pTraindataContainerListLength=None, pNrSamplesList=None, pStoredFeatures=None, pNrFactors=None):

    score = objective(config, pArgs, pTfRecordFilenames, pTraindataContainerListLength, pNrSamplesList, pStoredFeatures, pNrFactors)
    print("accuracy: ", score)
    train.report({"accuracy": score})

def run_raytune(pArgs, pContinueExperiment=None):
    os.makedirs(os.path.join(pArgs.outputFolder, "pygenometracks"), exist_ok=True)
    # Create a ray tune experiment
    # Define the search space
    search_space = {
        # "steps": 5,
        "loss_weight_pixel": tune.uniform(50.0, 150.0),
        "loss_weight_discriminator": tune.uniform(0.1, 1.0),
        "loss_type_pixel": tune.choice(["L1", "L2"]),
        "loss_weight_tv": tune.uniform(1e-15, 1e-7),
        "loss_weight_adversarial": tune.uniform(0.5, 1.5),
        "learning_rate_generator": tune.uniform(2e-15, 2e-3),
        "learning_rate_discriminator": tune.uniform(1e-13, 1e-3),
        "beta1": tune.uniform(0.0, 1.0),
        "flip_samples": tune.choice(["--flipSamples", ""]),
        "multiplier": tune.randint(0, 1000),
    }

    points_to_evaluate = [
        {   
            "loss_weight_pixel": 59.37721008879611,
            "loss_weight_discriminator": 0.43972911238860063,
            "loss_type_pixel": "L1",
            "loss_weight_tv": 8.080735989100953e-08,
            "loss_weight_adversarial": 0.9248942024710739,
            "learning_rate_generator": 0.0006947782705665501,
            "learning_rate_discriminator": 0.0005652269944795734,
            "beta1": 0.40871128817217095,
            "flip_samples": "",
            "multiplier": 282
        }
    ]

    print("points_to_evaluate: ", points_to_evaluate)
    
    tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
        pTrainingMatrices=pArgs.trainingMatrices, 
        pTrainingChromosomes=pArgs.trainingChromosomes, 
        pTrainingChromosomesFolders=pArgs.trainingChromosomesFolders, 
        pValidationMatrices=pArgs.validationMatrices, 
        pValidationChromosomes=pArgs.validationChromosomes, 
        pValidationChromosomesFolders=pArgs.validationChromosomesFolders,
        pWindowSize=pArgs.windowSize,
        pOutputFolder=pArgs.outputFolder,
        pBatchSize=pArgs.batchSize,
        pFlipSamples=False,
        pFigureFileFormat="png",
        pRecordSize=pArgs.recordSize
    )
      
        # Define the objective function
        # objective = tune.function(objective_raytune)
    objective_with_param = tune.with_parameters(objective_raytune, pArgs=pArgs, 
                                                pTfRecordFilenames=tfRecordFilenames, 
                                                pTraindataContainerListLength=traindataContainerListLength, 
                                                pNrSamplesList=nr_samples_list, 
                                                pStoredFeatures=storedFeatures, 
                                                pNrFactors=nr_factors)
    # objective_with_param = tune.with_parameters(objective_raytune, pArgs=pArgs, pTfRecordFilenames=None, pTraindataContainerListLength=None, pNrSamplesList=None, pStoredFeatures=None, pNrFactors=None, pScope=scope)
    
    objective_with_resources = tune.with_resources(objective_with_param, resources={"cpu": pArgs.threads, "gpu": pArgs.gpu})

    if pArgs.optimizer == "hyperopt":
        search_algorithm = HyperOptSearch(metric="accuracy", 
                                        mode="max",
                                        points_to_evaluate=points_to_evaluate)
    elif pArgs.optimizer == "optuna":
        search_algorithm = OptunaSearch(metric="accuracy", 
                                    mode="max",
                                    points_to_evaluate=points_to_evaluate)

    # tuner = tune.Tuner(objective_with_resources, param_space=search_space)  #

    
    if pContinueExperiment is None or pContinueExperiment == "":
        tuner = tune.Tuner(
            objective_with_resources, 
            param_space=search_space, 
            tune_config=tune.TuneConfig(num_samples=pArgs.numberSamples,
                                        search_alg=search_algorithm),
        )
    else:
        tuner = tune.Tuner.restore(path=pContinueExperiment, trainable=objective_with_resources)
        
    results = tuner.fit()


    print(results.get_best_result(metric="accuracy", mode="max").config)
    # print(results.get_best_trial(metric="accuracy", mode="max"))

    delete_model_files(pTFRecordFiles=tfRecordFilenames)

    
    


def run_opttuner():
    pass

def main(args=None):
    args = parse_arguments()
    run_raytune(pArgs=args, pContinueExperiment=args.continue_experiment)
