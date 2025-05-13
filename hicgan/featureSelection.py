import os
import logging
import argparse

import h5py
import tensorflow as tf
import numpy as np
from hicrep.utils import readMcool
from hicrep import hicrepSCC


import itertools
import random
import string
import traceback

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
                        type=int, default=1,
                        help="Frequency of plotting during training.")
    parser.add_argument("--matrixOutputName", "-mon", required=False,
                        type=str, default="predicted_matrix.mcool",
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
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

def runTrainingPredictionAndValidation(pArgs, strategy):
    
    trial_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    print("GPU: {}; Training and prediction with combination {}: {}".format(pArgs.whichGPU, trial_id, len(pArgs.trainingChromatinFolder)))
    filenames = [os.path.basename(path) for path in pArgs.trainingChromatinFolder]
    # print("Training Chromatin Folder:", pArgs.trainingChromatinFolder)
    print("GPU: {}; Filenames from trainingChromatinFolder: {}".format(pArgs.whichGPU, filenames))
    # Use cooler to get the bin size from the training matrices
    if not pArgs.trainingMatrices or len(pArgs.trainingMatrices) == 0:
        raise ValueError("No training matrices provided.")
    clr = cooler.Cooler(pArgs.trainingMatrices[0])
    binSize = clr.binsize

    tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
            pTrainingMatrices=pArgs.trainingMatrices, 
            pTrainingChromosomes=pArgs.trainingChromosomes, 
            pTrainingChromatinFolders=pArgs.trainingChromatinFolder, 
            pValidationMatrices=pArgs.validationMatrices, 
            pValidationChromosomes=pArgs.validationChromosomes, 
            pValidationChromatinFolders=pArgs.validationChromatinFolder,
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
        pPredictionChromatinFolders=pArgs.predictionChromatinFolder,
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
        print("SCC score: ", score)
        # Save the score to a file
        with open(os.path.join(pArgs.outputFolder, trial_id, "scc_score.txt"), "w") as f:
            f.write(f"SCC score: {score:.4f}\n")
        # Save the score to a JSON file
    except Exception as e:
        traceback.print_exc()
        print(e)
        score = -1

    log.debug("Compute feature importance using Explainable AI")
    # try:

    #     log.info("Compute feature importance using Gradient x Input")

    #     # Load the trained generator model
    #     model_path = os.path.join(pArgs.outputFolder, trial_id, pArgs.generatorName)
    #     model = load_model(model_path)

    #     # Prepare input feature matrix
    #     traindataRecords = [item for sublist in tfRecordFilenames[0:traindataContainerListLength] for item in sublist]
    #     trainDs = tf.data.TFRecordDataset(
    #         traindataRecords,
    #         num_parallel_reads=tf.data.experimental.AUTOTUNE,
    #         compression_type="GZIP"
    #     )
    #     sample_data = trainDs.map(
    #         lambda x: records.parse_function(x, storedFeatures),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     ).take(100)

    #     # Convert dataset to NumPy array
    #     # sample_data_np = np.array([x.numpy() for x in sample_data], dtype=np.float32)
    #     sample_data_np = np.array([
    #                                 np.stack([v.numpy().squeeze() for v in x[0].values()], axis=-1)  # shape (768, 12, num_features)
    #                                 for x in sample_data
    #                             ], dtype=np.float32)
    #     # Convert to tensor
    #     X_tensor = tf.convert_to_tensor(sample_data_np, dtype=tf.float32)
    #     with tf.GradientTape() as tape:
    #         tape.watch(X_tensor)
    #         preds = model(X_tensor)

    #     grads = tape.gradient(preds, X_tensor)
    #     grad_times_input = grads * X_tensor
    #     feature_importance = tf.reduce_mean(tf.abs(grad_times_input), axis=0).numpy()

    #     # Save feature importance as .npy
    #     importance_path = os.path.join(pArgs.outputFolder, trial_id, "feature_importance.npy")
    #     np.save(importance_path, feature_importance)

    #     # Optionally, visualize
    #     import matplotlib.pyplot as plt

    #     feature_names = list(storedFeatures.keys())
    #     feature_importance_1d = feature_importance.mean(axis=(0, 1))  # shape: (num_features,)

    #     # Plot
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(range(len(feature_names)), feature_importance_1d)
    #     plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    #     plt.tight_layout()
    #     plot_path = os.path.join(pArgs.outputFolder, trial_id, "feature_importance_plot.png")
    #     plt.savefig(plot_path)
    #     plt.close()

    #     log.info(f"Feature importance computed and saved to {importance_path} and {plot_path}")

    # except Exception as e:
    #     traceback.print_exc()
    #     print(f"Error computing feature importance: {e}")

    return score, trial_id
     

def main(args=None):
    args = parse_arguments()
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
    
        

    print("Number of file combinations: {}".format(len(combined_files_training)))
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
    if gpu:
        try:
            for gpu_device in gpu:
                tf.config.experimental.set_memory_growth(gpu_device, True)
        except Exception as e:
            print("Error: {}".format(e))

    results = {}
    active_threads = []
    gpu_status = [False] * len(gpu)  # False means GPU is free

    for i, (training_files, validation_files, prediction_files) in enumerate(zip(combined_files_training, combined_files_validation, combined_files_prediction)):

        args.trainingChromatinFolder = training_files
        args.validationChromatinFolder = validation_files
        args.predictionChromatinFolder = prediction_files

        while True:
            # Check for a free GPU
            free_gpu_index = next((index for index, status in enumerate(gpu_status) if not status), None)
            if free_gpu_index is not None:
                args.whichGPU = free_gpu_index
                gpu_status[free_gpu_index] = True  # Mark GPU as busy

                def run_and_store_score(files, gpu_index):
                    strategy = tf.distribute.OneDeviceStrategy(device=f"/GPU:{gpu_index}")
                    score, trial_id = runTrainingPredictionAndValidation(args, strategy)
                    # score = gpu_index  # Placeholder for actual score
                    results[tuple(files)] = (score, trial_id)
                    gpu_status[gpu_index] = False  # Mark GPU as free after completion

                # print("Training and prediction with combination {}: {}".format(i+1, training_files))
                print(f"\n\nTraining and prediction with combination {i+1}/{len(combined_files_training)} on GPU {free_gpu_index}, files {len(training_files)}: {training_files}\n\n")

                thread = threading.Thread(target=run_and_store_score, args=(training_files, free_gpu_index))
                thread.start()
                active_threads.append(thread)
                break  # Exit the loop to process the next combination
            else:
                # Wait for any thread to finish if no GPU is free
                for t in active_threads:
                    t.join(timeout=1)
                active_threads = [t for t in active_threads if t.is_alive()]

    # Ensure all threads are completed before exiting
    for t in active_threads:
        t.join()
    
    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)
    with open(os.path.join(args.outputFolder, "results.txt"), "w") as outfile:
        for combo, score_val in results.items():
            outfile.write(f"{combo}: {score_val}\n")