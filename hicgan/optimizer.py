import icecream as ic
import argparse
import numpy as np
import pandas as pd
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

import traceback

import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from hicrep.utils import readMcool
from hicrep import hicrepSCC

import pygenometracks.plotTracks

from hicexplorer import hicFindTADs


import logging
from hicgan._version import __version__
from hicgan.training import training, create_data, delete_model_files
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation
import time

log = logging.getLogger(__name__)


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
    parser.add_argument("--polynomialModel", "-pm", required=False,
                        default='pearson',
                        type=str, choices=['p_A-h',
                                            'p_A-T_f',
                                            'p_A-T_f_e_m',
                                            'h-T_f',
                                            'h-T_f_e_m',
                                            'T_f-T_f_e_m',
                                            'p_A-h-T_f',
                                            'p_A-h-T_f_e_m',
                                            'p_A-T_f-T_f_e_m',
                                            'h-T_f-T_f_e_m',
                                            'p_A-h-T_f-T_f_e_m'],

                        help="Type of error to compute (options: 'p_A_h', 'p_A_T_f', 'p_A_T_f_e_m', \
                                                                    'h_T_f', \
                                                                    'h_T_f_e_m', \
                                                                    'T_f_T_f_e_m', \
                                                                    'p_A_h_T_f', \
                                                                    'p_A_h_T_f_e_m', \
                                                                    'p_A_T_f_T_f_e_m', \
                                                                    'h_T_f_T_f_e_m', \
                                                                    'p_A_h_T_f_T_f_e_m'")
    parser.add_argument("--polynomialModelFolder", "-pmf", required=False,
                        type=str,
                        default=".",
                        help="The folder with the stored polynomial models")
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


def objective(config, pArgs):

    # if pArgs.gpu > 1:
    #     gpu = tf.config.list_physical_devices('GPU')
    #     if gpu:
    #         try:
    #             for gpu_device in gpu:
    #                 tf.config.experimental.set_memory_growth(gpu_device, True)
    #         except Exception as e:
    #             print("Error: {}".format(e))
    #     strategy = tf.distribute.MirroredStrategy()
    # else:
    #     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    try:

        assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"Ray assigned GPU devices: {assigned_gpus}")

        # From TF's perspective, only the GPUs listed in CUDA_VISIBLE_DEVICES exist.
        physical_gpus = tf.config.list_physical_devices('GPU')
        print(f"Physical GPUs: {physical_gpus}")
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
                # strategy = tf.distribute.OneDeviceStrategy(device=device_name)
            else:
                strategy = tf.distribute.MirroredStrategy()


        trial_id = session.get_trial_id()
        print("trail_id {}".format(trial_id))

        os.makedirs(os.path.join(pArgs.outputFolder, trial_id), exist_ok=True)
        matrixOutputNameWithoutExt = os.path.splitext(pArgs.matrixOutputName)[0]
        lock_file_data_generation_path = os.path.join(pArgs.outputFolder, "dataGeneration.lock")
        lock_file_prediction_path = os.path.join(pArgs.outputFolder, "prediction.lock")
        lock_file_pearson_path = os.path.join(pArgs.outputFolder, "pearson.lock")
        lock_file_hicrep_path = os.path.join(pArgs.outputFolder, "hicrep.lock")
        lock_file_tad_path = os.path.join(pArgs.outputFolder, "tad.lock")
        lock_file_polynomial_path = os.path.join(pArgs.outputFolder, "polynomial.lock")
        lock_file_pygenometracks_path = os.path.join(pArgs.outputFolder, "pygenometracks.lock")
        lock_file_delete_data_path = os.path.join(pArgs.outputFolder, "deleteData.lock")



        def activate_lock_or_wait(file_path, method="Data generation", timeout=100):
            start_time = time.time()
            while os.path.exists(file_path):
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Active lock: {file_path} found within {timeout} seconds.")
                time.sleep(1)
            
            # Create the lock file
            with open(file_path, 'w') as lock_file:
                lock_file.write("{} in progress".format(method))
        
        def removeLock(file_path):
            if os.path.exists(file_path):
                os.remove(file_path)
        activate_lock_or_wait(lock_file_data_generation_path)
        tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
            pTrainingMatrices=pArgs.trainingMatrices, 
            pTrainingChromosomes=pArgs.trainingChromosomes, 
            pTrainingChromosomesFolders=pArgs.trainingChromosomesFolders, 
            pValidationMatrices=pArgs.validationMatrices, 
            pValidationChromosomes=pArgs.validationChromosomes, 
            pValidationChromosomesFolders=pArgs.validationChromosomesFolders,
            pWindowSize=pArgs.windowSize,
            pOutputFolder=pArgs.outputFolder,
            pBatchSize=config['batch_size'],
            pFlipSamples=False,
            pFigureFileFormat="png",
            pRecordSize=pArgs.recordSize
        )

        # Remove the data generation lock file
        
        removeLock(lock_file_data_generation_path)
        with strategy.scope() as scope:
            training(
                pTfRecordFilenames=tfRecordFilenames,
                pLengthTrainDataContainerList=traindataContainerListLength,
                pWindowSize=pArgs.windowSize,
                pOutputFolder=os.path.join(pArgs.outputFolder, trial_id),
                pEpochs=pArgs.epochs,
                pBatchSize=config["batch_size"],
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
                pStoredFeaturesDict=storedFeatures,
                pNumberSamplesList=nr_samples_list,
                pNumberOfFactors=nr_factors,
                pFlipSamples=config["flip_samples"],
                pRecordSize=pArgs.recordSize
            )

        activate_lock_or_wait(lock_file_prediction_path, method="Prediction")
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
        removeLock(lock_file_prediction_path)
        score_dict = {}
        correlationMethodList = ['pearson_spearman', 'hicrep', 'TAD_score_MSE', "TAD_fraction"]
        errorType = ['R2', 'MSE', 'MAE', 'MSLE', 'AUC'] 
        for correlationMethod in correlationMethodList:
            
            if correlationMethod == 'pearson_spearman':
                for chrom in pArgs.testChromosomes:
                    activate_lock_or_wait(lock_file_pearson_path, method="Pearson correlation")
                    score_dataframe = computePearsonCorrelation(pCoolerFile1=os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pCoolerFile2=pArgs.originalDataMatrix,
                                                            pWindowsize_bp=pArgs.correlationDepth, pModelChromList=pArgs.trainingChromosomes, pTargetChromStr=chrom,
                                                            pModelCellLineList=pArgs.trainingCellType, pTargetCellLineStr=pArgs.testCellType,
                                                            pPlotOutputFile=None, pCsvOutputFile=None)
                    removeLock(lock_file_pearson_path)
                    for correlationMethod_ in ['pearson', 'spearman']:
                        for errorType_ in errorType:
                            if correlationMethod_ + '_' + errorType_ in score_dict:
                                score_dict[correlationMethod_ + '_' + errorType_][0] += score_dataframe.loc[correlationMethod_, errorType_]
                            else:
                                score_dict[correlationMethod_ + '_' + errorType_] = [score_dataframe.loc[correlationMethod_, errorType_]]

                for correlationMethod_ in ['pearson', 'spearman']:
                    for errorType_ in errorType:
                        score_dict[correlationMethod_ + '_' + errorType_][0] = score_dict[correlationMethod_ + '_' + errorType_][0] / len(pArgs.testChromosomes)

            elif correlationMethod == 'hicrep':
                activate_lock_or_wait(lock_file_hicrep_path, method="hicrep")
                
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
                removeLock(lock_file_hicrep_path)
                score_dict[correlationMethod] = [np.mean(sccSub)]
            elif correlationMethod == 'TAD_score_MSE' or correlationMethod == 'TAD_fraction':
                os.makedirs(os.path.join(pArgs.outputFolder, trial_id, "tads_predicted"), exist_ok=True)
                chromosomes = ' '.join(pArgs.testChromosomes)
                arguments_tad = "--matrix {} --minDepth {} --maxDepth {} --step {} --numberOfProcessors {}  \
                                --outPrefix {} --minBoundaryDistance {} \
                                --correctForMultipleTesting fdr --thresholdComparisons 0.5 --chromosomes {}".format(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pArgs.binSize * 3, pArgs.binSize * 10, pArgs.binSize, pArgs.threads,
                                os.path.join(pArgs.outputFolder, trial_id, "tads_predicted") + '/tads', 100000, chromosomes).split()
                activate_lock_or_wait(lock_file_tad_path, method="TADs")
                try:
                    hicFindTADs.main(arguments_tad)
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    return

                if correlationMethod == 'TAD_score_MSE':
                    tad_score_predicted = os.path.join(
                        pArgs.outputFolder, trial_id, "tads_predicted") + '/tads_score.bedgraph'
                    tad_score_orgininal = os.path.join(
                        pArgs.outputFolder, "tads_original") + '/tads_score.bedgraph'


                    tad_score_predicted_df = pd.read_csv(tad_score_predicted, names=[
                                                            'chromosome', 'start', 'end', 'score'], sep='\t')
                    tad_score_orgininal_df = pd.read_csv(tad_score_orgininal, names=[
                                                            'chromosome', 'start', 'end', 'score'], sep='\t')

                
                    mean_sum_of_squares = ((tad_score_predicted_df['score'] - tad_score_orgininal_df['score']) ** 2).mean()
                    score_dict[correlationMethod] = [mean_sum_of_squares]
                elif correlationMethod == 'TAD_fraction':
                    tad_boundaries_predicted = os.path.join(
                        pArgs.outputFolder, trial_id, "tads_predicted")  + '/tads_boundaries.bed'
                    tad_boundaries_orgininal = os.path.join(
                        pArgs.outputFolder, "tads_original")  + '/tads_boundaries.bed'

                    tad_boundaries_predicted_df = pd.read_csv(tad_boundaries_predicted, names=[
                                                            'chromosome', 'start', 'end', 'name', 'score', '.'], sep='\t')
                    tad_boundaries_orgininal_df = pd.read_csv(tad_boundaries_orgininal, names=[
                                                            'chromosome', 'start', 'end', 'name', 'score', '.'], sep='\t')
                    
                    tad_fraction = len(tad_boundaries_predicted_df) / len(tad_boundaries_orgininal_df)
                    
                    exact_matches = pd.merge(tad_boundaries_predicted_df[['chromosome', 'start', 'end']],
                                            tad_boundaries_orgininal_df[['chromosome', 'start', 'end']],
                                            on=['chromosome', 'start', 'end'], how='inner')
                    tad_fraction_exact_match = len(exact_matches) / len(tad_boundaries_orgininal_df)

                    score_dict[correlationMethod] = [tad_fraction]
                    score_dict[correlationMethod + '_exact_match'] = [tad_fraction_exact_match]
                removeLock(lock_file_tad_path)

        activate_lock_or_wait(lock_file_polynomial_path, method="Polynomial model")
        # List all files in the models_path directory
        model_files = [f for f in os.listdir(pArgs.polynomialModelFolder) if f.endswith('.pkl')]
        print(model_files)
        # Initialize an empty dictionary to store the loaded models
        loaded_models = {}
        model = None
        # Iterate over the saved model files and load them
        for model_name in model_files:
            if pArgs.polynomialModel + '.pkl' == model_name:
                
                model_file = os.path.join(pArgs.polynomialModelFolder, f"{model_name}")
                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    print(f"Loaded model: {model_name}")
                else:
                    print(f"Model file not found: {model_file}")
        if model is not None:
            scores_df = pd.DataFrame(score_dict)
            print(scores_df.columns)
            features = []
            features_short = {'p_A':'pearson_AUC', 'h':'hicrep', 'T_f':'TAD_fraction', 'T_f_e_m':'TAD_fraction_exact_match'}
            names_model = pArgs.polynomialModel.split("-")
            for name in names_model:
                features.append(features_short[name])

            score = model.predict(scores_df[features])[0]
        removeLock(lock_file_polynomial_path)
        activate_lock_or_wait(lock_file_pygenometracks_path, method="PyGenomeTracks")
        if pArgs.genomicRegion:
            score_text = pArgs.polynomialModel + str(score)
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
height = 0.5

[TAD seperation score]
file = {5}
height = 2
type = lines
individual_color = grey
pos_score_in_bin = center
summary_color = #1f77b4
show_data_range = true
file_type = bedgraph_matrix

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

[spacer]
height = 0.5

[TAD seperation score]
file = {6}
height = 2
type = lines
individual_color = grey
pos_score_in_bin = center
summary_color = #1f77b4
show_data_range = true
file_type = bedgraph_matrix
        """.format(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pArgs.originalDataMatrix, score_text, pArgs.trainingCellType, 2000000, \
                os.path.join(pArgs.outputFolder, trial_id, "tads_predicted", 'tads_tad_score.bm'),
                    os.path.join(pArgs.outputFolder, "tads_original", "tads_tad_score.bm"))
            

            tracks_path = os.path.join(
                pArgs.outputFolder, "browser_tracks_hic.ini")
            with open(tracks_path, 'w') as fh:
                fh.write(browser_tracks_with_hic)

            outfile = os.path.join(
                pArgs.outputFolder, "pygenometracks", trial_id + '_' + matrixOutputNameWithoutExt + ".pdf")

            arguments = f"--tracks {tracks_path} --region {pArgs.genomicRegion} "\
                        f"--outFileName {outfile} --trackLabelFraction 0.1 --width 38 --height 35".split()
            try:
                pygenometracks.plotTracks.main(arguments)
            except Exception as e:
                traceback.print_exc()
                print(e)
        removeLock(lock_file_pygenometracks_path)
        activate_lock_or_wait(lock_file_delete_data_path)
        delete_model_files(pTFRecordFiles=tfRecordFilenames)
        removeLock(lock_file_delete_data_path)
    except tf.errors.OpError as e:
        # tf.errors.OpError is a common superclass for many TF errors
        traceback_str = traceback.format_exc()
        # Re-raise as a generic Python exception with the original traceback
        raise RuntimeError(
            f"TensorFlow OpError caught. Original traceback:\n{traceback_str}"
        ) from e
    return score

def objective_raytune(config, pArgs, pMetric):

    score = objective(config, pArgs)
    train.report({pMetric: score})


def run_raytune(pArgs, pContinueExperiment=None):
    os.makedirs(os.path.join(pArgs.outputFolder,
                "pygenometracks"), exist_ok=True)
    os.makedirs(os.path.join(pArgs.outputFolder, "tads_original"), exist_ok=True)
    chromosomes = ' '.join(pArgs.testChromosomes)
    arguments_tad = "--matrix {} --minDepth {} --maxDepth {} --step {} --numberOfProcessors {}  \
                        --outPrefix {} --minBoundaryDistance {} \
                        --correctForMultipleTesting fdr --thresholdComparisons 0.5 --chromosomes {}".format(pArgs.originalDataMatrix, pArgs.binSize * 3, pArgs.binSize * 10, pArgs.binSize, pArgs.threads,
                    os.path.join(pArgs.outputFolder, "tads_original") + '/tads', 100000, chromosomes).split()
    print(arguments_tad)
    hicFindTADs.main(arguments_tad)
    tad_score_orgininal = os.path.join(
        pArgs.outputFolder, "tads_original") + '/tads_score.bedgraph'
    tad_score_orgininal_df = pd.read_csv(tad_score_orgininal, names=[
                                        'chromosome', 'start', 'end', 'score'])

    # Create a ray tune experiment
    # Define the search space
    search_space = {
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
        "batch_size": tune.randint(1,256)
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
            "multiplier": 282,
            "batch_size": 10
        }
    ]

      
        # Define the objective function
        # objective = tune.function(objective_raytune)
    metric = 'accuracy'
    mode = 'max'
        
    objective_with_param = tune.with_parameters(objective_raytune, pArgs=pArgs,
                                                pMetric=metric)
    
    objective_with_resources = tune.with_resources(objective_with_param, resources={"cpu": pArgs.threads, "gpu": pArgs.gpu})

    if pArgs.optimizer == "hyperopt":
        search_algorithm = HyperOptSearch(metric=metric,
                                        mode=mode,
                                        points_to_evaluate=points_to_evaluate)
    elif pArgs.optimizer == "optuna":
        search_algorithm = OptunaSearch(metric=metric, 
                                    mode=mode,
                                    points_to_evaluate=points_to_evaluate)

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


    print(results.get_best_result(metric=metric, mode=mode).config)

def run_opttuner():
    pass

def main(args=None):
    args = parse_arguments()
    run_raytune(pArgs=args, pContinueExperiment=args.continue_experiment)
