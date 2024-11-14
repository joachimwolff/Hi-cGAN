import argparse
import numpy as np
import os
import csv
import tensorflow as tf
from .lib import dataContainer
from .lib import records
from .lib import hicGAN
from .lib import utils

from ray import train, tune


import logging
from hicgan._version import __version__
from hicgan.training import training
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation

log = logging.getLogger(__name__)

import icecream as ic

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--optimizer", "-op", required=True,
                        type=str, choices=['raytune', 'opttuner'],
                        help="Optimizer to use for training (options: 'raytune', 'opttuner')")
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
                        type=str, choices=['pearson', 'spearman'],
                        help="Type of error to compute (options: 'pearson', 'spearman')")
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
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()



def objective_raytune(config, pArgs):

    # trainingMatricesString = ' '.join(pArgs.trainingMatrices)
    # trainingChromosomesString = ' '.join(pArgs.trainingChromosomes)
    # trainingChromosomesFoldersString = ' '.join(pArgs.trainingChromosomesFolders)
    # validationMatricesString = ' '.join(pArgs.validationMatrices)
    # validationChromosomesString = ' '.join(pArgs.validationChromosomes)
    # validationChromosomesFoldersString = ' '.join(pArgs.validationChromosomesFolders)
    # # testMatricesString = ' '.join(args.testMatrices)
    # # testChromosomesString = ' '.join(args.testChromosomes)
    # # testChromosomesFoldersString = ' '.join(args.testChromosomesFolders)

    # args_string = "--trainingMatrices {} --trainingChromosomes {} \
    #            --trainingChromosomesFolders {} --validationMatrices {} \
    #            --validationChromosomes {} --validationChromosomesFolders {} \
    #            --windowSize {} --outputFolder {} --epochs {} --batchSize {} \
    #            --lossWeightPixel {} --lossWeightDiscriminator {} --lossTypePixel {} \
    #            --lossWeightTV {} --lossWeightAdversarial {} --learningRateGenerator {} \
    #            --learningRateDiscriminator {} --beta1 {} {} --figureFileFormat {} --recordSize {}".format(
    #                trainingMatricesString,  # --trainingMatrices
    #                trainingChromosomesString,  # --trainingChromosomes
    #                trainingChromosomesFoldersString,  # --trainingChromosomesFolders
    #                validationMatricesString,  # --validationMatrices
    #                validationChromosomesString,  # --validationChromosomes
    #                validationChromosomesFoldersString,  # --validationChromosomesFolders
    #                pArgs.windowSize,  # --windowSize
    #                pArgs.outputFolder,  # --outputFolder
    #                pArgs.epochs,  # --epochs
    #                pArgs.batchSize,  # --batchSize
    #                config["loss_weight_pixel"],  # --lossWeightPixel
    #                config["loss_weight_discriminator"],  # --lossWeightDiscriminator
    #                config["loss_type_pixel"],  # --lossTypePixel
    #                config["loss_weight_tv"],  # --lossWeightTV
    #                config["loss_weight_adversarial"],  # --lossWeightAdversarial
    #                config["learning_rate_generator"],  # --learningRateGenerator
    #                config["learning_rate_discriminator"],  # --learningRateDiscriminator
    #                config["beta1"],  # --beta1
    #                config["flip_samples"],  # --flipSamples
    #                "png",  # --figureFileFormat
    #                pArgs.recordSize  # --recordSize
    #            ).split()
    
    # print(args_string)
    training(trainingMatrices=pArgs.trainingMatrices, 
             trainingChromosomes=pArgs.trainingChromosomes, 
             trainingChromosomesFolders=pArgs.trainingChromosomesFolders, 
             validationMatrices=pArgs.validationMatrices, 
             validationChromosomes=pArgs.validationChromosomes, 
             validationChromosomesFolders=pArgs.validationChromosomesFolders,
             windowSize=pArgs.windowSize,
             outputFolder=pArgs.outputFolder,
             epochs=pArgs.epochs,
             batchSize=pArgs.batchSize,
             lossWeightPixel=config["loss_weight_pixel"],
             lossWeightDiscriminator=config["loss_weight_discriminator"],
             lossWeightAdversarial=config["loss_weight_adversarial"],
             lossTypePixel=config["loss_type_pixel"],
             lossWeightTV=config["loss_weight_tv"],
             learningRateGenerator=config["learning_rate_generator"],
             learningRateDiscriminator=config["learning_rate_discriminator"],
             beta1=config["beta1"],
             flipSamples=config["flip_samples"],
             figureFileFormat="png",
             recordSize=pArgs.recordSize,
             plotFrequency=20)

    # # predictionChromosomesFoldersString = ' '.join(args.predictionChromosomesFolders)
    # predictionChromosomesString = ' '.join(pArgs.predictionChromosomes)
    
    # args_string_for_prediction = "--trainedModel {} --predictionChromosomesFolders {} \
    #            --predictionChromosomes {} --matrixOutputName {} --parameterOutputFile {} \
    #            --outputFolder {} --multiplier {} --binSize {} --windowSize {}".format(
    #                os.path.join(pArgs.outputFolder, pArgs.generatorName),  # --trainedModel
    #                pArgs.predictionChromosomesFolders,  # --predictionChromosomesFolders
    #                predictionChromosomesString,  # --predictionChromosomes
    #                pArgs.matrixOutputName,  # --matrixOutputName
    #                pArgs.parameterOutputFile,  # --parameterOutputFile
    #                pArgs.outputFolder,  # --outputFolder
    #                config["multiplier"],  # --multiplier
    #                pArgs.binSize,  # --binSize
    #                pArgs.windowSize
    #            ).split()
    # print(args_string_for_prediction)
    # Run args_string_for_prediction using the computed model

    prediction(
        pTrainedModel=os.path.join(pArgs.outputFolder, pArgs.generatorName),
        pPredictionChromosomesFolders=pArgs.predictionChromosomesFolders,
        pPredictionChromosomes=pArgs.predictionChromosomes,
        pOutputFolder=pArgs.outputFolder,
        pMultiplier=config["multiplier"],
        pBinSize=pArgs.binSize,
        pBatchSize=pArgs.batchSize,
        pWindowSize=pArgs.windowSize,
        pMatrixOutputName=pArgs.matrixOutputName,
        pParameterOutputFile=pArgs.parameterOutputFile
    )
    # Compute the Pearson correlation error
    score = 0
    # test_chromosomes = args.testChromosomes.split(" ")
    for chrom in pArgs.testChromosomes:
        score_dataframe = computePearsonCorrelation(pCoolerFile1=os.path.join(pArgs.outputFolder, pArgs.matrixOutputName), pCoolerFile2=pArgs.originalDataMatrix,
                                                pWindowsize_bp=pArgs.correlationDepth, pModelChromList=pArgs.trainingChromosomes, pTargetChromStr=chrom,
                                                pModelCellLineList=pArgs.trainingCellType, pTargetCellLineStr=pArgs.testCellType,
                                                pPlotOutputFile=None, pCsvOutputFile=None)
        score += score_dataframe.loc[pArgs.correlationMethod, pArgs.errorType]
    score = score / len(pArgs.testChromosomes)
    print("Score: ", score)
    # Return the error as the objective value
    return {"score": score}

def run_raytune(pArgs):
    # Create a ray tune experiment
    # Define the search space
    search_space = {
        "loss_weight_pixel": tune.uniform(50.0, 150.0),
        "loss_weight_discriminator": tune.uniform(0.1, 1.0),
        "loss_type_pixel": tune.choice(["L1", "L2"]),
        "loss_weight_tv": tune.uniform(1e-15, 1e-1),
        "loss_weight_adversarial": tune.uniform(0.5, 1.5),
        "learning_rate_generator": tune.uniform(1e-15, 1e-1),
        "learning_rate_discriminator": tune.uniform(1e-15, 1e-1),
        "beta1": tune.uniform(0.0, 1.0),
        "flip_samples": tune.choice(["--flipSamples", ""]),
        "multiplier": tune.randint(0, 1000),
    }

    
    # Define the objective function
    # objective = tune.function(objective_raytune)
    objective_with_param = tune.with_parameters(objective_raytune, pArgs=pArgs)
    objective_with_resources = tune.with_resources(objective_with_param, resources={"cpu": 16, "gpu": 2})


    # tuner = tune.Tuner(objective_with_resources, param_space=search_space)  #

    

    tuner = tune.Tuner(
        objective_with_resources, param_space=search_space, tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()


    print(results.get_best_result(metric="score", mode="max").config)

    
    


def run_opttuner():
    pass

def main(args=None):
    args = parse_arguments()
    if args.optimizer == 'raytune':
        run_raytune(pArgs=args)
    elif args.optimizer == 'opttuner':
        run_opttuner()
