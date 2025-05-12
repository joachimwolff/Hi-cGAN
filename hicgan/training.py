import csv
import os
import numpy as np
import tensorflow as tf
import concurrent.futures
import argparse
from datetime import datetime
import cooler
import h5py

from .lib import hicGAN
from .lib import dataContainer
from .lib import records

from hicgan._version import __version__

import logging
log = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingMatrices", "-tm", required=True,
                        type=str, nargs='+',
                        help="mcooler matrices for training.")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=True,
                        type=str, nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromatinFolders", "-tcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files). Alternatively, a list of single bigwig paths can be provided.")
    parser.add_argument("--validationMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=True,
                        type=str, nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromatinFolders", "-vcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    parser.add_argument("--windowSize", "-ws", required=True,
                        type=int, choices=[64, 128, 256, 512],
                        help="window size for submatrices.")
    parser.add_argument("--outputFolder", "-o", required=True,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=True,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--lossWeightPixel", "-lwp", required=False,
                        type=float,
                        default=100.0,
                        help="Loss weight for L1/L2 error of generator.")
    parser.add_argument("--lossWeightDiscriminator", "-lwd", required=False,
                        type=float,
                        default=0.5,
                        help="Loss weight (multiplicator) for the discriminator loss.")
    parser.add_argument("--lossTypePixel", "-ltp", required=False,
                        type=str, choices=["L1", "L2"],
                        default="L1",
                        help="Type of per-pixel loss to use for the generator.")
    parser.add_argument("--lossWeightTV", "-lwt", required=False,
                        type=float,
                        default=1e-10,
                        help="Loss weight for Total-Variation-loss of generator.")
    parser.add_argument("--lossWeightAdversarial", "-lwa", required=False,
                        type=float,
                        default=1.0,
                        help="Loss weight for adversarial loss in generator.")
    parser.add_argument("--learningRateGenerator", "-lrg", required=False,
                        type=float,
                        default=2e-5,
                        help="Learning rate for Adam optimizer of generator.")
    parser.add_argument("--learningRateDiscriminator", "-lrd", required=False,
                        type=float,
                        default=1e-6,
                        help="Learning rate for Adam optimizer of discriminator.")
    parser.add_argument("--beta1", "-b1", required=False,
                        type=float,
                        default=0.5,
                        help="Beta1 parameter for Adam optimizers (gen. and disc.)")
    parser.add_argument("--flipSamples", "-fs", required=False,
                        action='store_true',
                        help="Flip training matrices and chromatin features (data augmentation).")
    parser.add_argument("--figureFileFormat", "-ft", required=False,
                        type=str, choices=["png", "pdf", "svg"],
                        default="png",
                        help="Figure type for all plots.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int,
                        default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")
    parser.add_argument("--plotFrequency", "-pfreq", required=False,
                        type=int,
                        default=10,
                        help="Update loss over epoch plots after this number of epochs.")
    parser.add_argument("--multiGPUTraining", "-mgpu", required=False,
                        type=bool,
                        default=False,
                        help="Enable multi-GPU training.")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default="",
                        help="Specify which GPU to use for training in the single GPU case. E.g. 1, 2, etc.")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser
def create_container(chrom, matrix, chromatinpath):
        container = dataContainer.DataContainer(chromosome=chrom,
                                                matrixFilePath=matrix,
                                                chromatinData=chromatinpath)
        return container

def create_data(pTrainingMatrices, 
                pTrainingChromosomes, 
                pTrainingChromatinFolders, 
                pValidationMatrices, 
                pValidationChromosomes, 
                pValidationChromosomesFolders,
                pWindowSize,
                pOutputFolder,
                pBatchSize,
                pFlipSamples,
                pFigureFileFormat,
                pRecordSize):
    os.makedirs(pOutputFolder, exist_ok=True)
    #few constants
    # windowSize = int(windowSize)
    debugstate = None
    paramDict = locals().copy()

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    # trainChromNameList = trainingChromosomes.replace(",","")
    # trainChromNameList = trainChromNameList.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in pTrainingChromosomes]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    # valChromNameList = validationChromosomes.replace(",","")
    # valChromNameList = valChromNameList.rstrip().split(" ")
    valChromNameList = [x.lstrip("chr") for x in pValidationChromosomes]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList

    #ensure there are as many matrices as chromatin paths
    trainingChromatinIsFolder = False
    for folder in pTrainingChromatinFolders:
        if os.path.isdir(folder):
            trainingChromatinIsFolder = True
            break
    validationChromatinIsFolder = False
    for folder in pValidationChromosomesFolders:
        if os.path.isdir(folder):
            validationChromatinIsFolder = True
            break
    if trainingChromatinIsFolder:
        if len(pTrainingMatrices) != len(pTrainingChromatinFolders):
            msg = "Number of train matrices and chromatin paths must match\n"
            msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
            msg = msg.format(len(pTrainingMatrices), len(pTrainingChromatinFolders))
            raise SystemExit(msg)
    if validationChromatinIsFolder:   
        if len(pValidationMatrices) != len(pValidationChromosomesFolders):
            msg = "Number of validation matrices and chromatin paths must match\n"
            msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
            msg = msg.format(len(pValidationMatrices), len(pValidationChromosomesFolders))
            raise SystemExit(msg)

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in trainChromNameList:
            if trainingChromatinIsFolder:
                for matrix, chromatinpath in zip(pTrainingMatrices, pTrainingChromatinFolders):
                    future = executor.submit(create_container, chrom, matrix, chromatinpath)
                    traindataContainerList.append(future.result())
            else:
                for matrix in pTrainingMatrices:
                    future = executor.submit(create_container, chrom, matrix, pTrainingChromatinFolders)
                    traindataContainerList.append(future.result())

    #prepare the validation data containers. No data is loaded yet.
    valdataContainerList = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in valChromNameList:
            if validationChromatinIsFolder:
                for matrix, chromatinpath in zip(pValidationMatrices, pValidationChromosomesFolders):
                    future = executor.submit(create_container, chrom, matrix, chromatinpath)
                    valdataContainerList.append(future.result())
            else:
                for matrix in pValidationMatrices:
                    future = executor.submit(create_container, chrom, matrix, pValidationChromosomesFolders)
                    valdataContainerList.append(future.result())

    #define the load params for the containers
    loadParams = {"scaleFeatures": True,
                "clampFeatures": False,
                "scaleTargets": True,
                "windowSize": pWindowSize,
                "flankingSize": pWindowSize,
                "maximumDistance": None}
    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = traindataContainerList[0]
    tfRecordFilenames = []
    nr_samples_list = []
    for container in traindataContainerList + valdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=pOutputFolder,
                                                        pRecordSize=pRecordSize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                        outpath=pOutputFolder,
                                        figureFileFormat=pFigureFileFormat)
            container.saveMatrix(outputpath=pOutputFolder, index=idx)
        nr_samples_list.append(container.getNumberSamples())
    print('ALL TF RECORDS CREATED!')

    #data is no longer needed
    for container in traindataContainerList + valdataContainerList:
        container.unloadData()
    
    print(tfRecordFilenames)
    print(len(tfRecordFilenames))
    print(len(traindataContainerList))

    #different binSizes are ok
    #not clear which binSize to use for prediction when they differ during training.
    #For now, store the max. 
    binSize = max([container.binSize for container in traindataContainerList])

    return tfRecordFilenames, len(traindataContainerList), nr_samples_list, container0.storedFeatures, container0.nr_factors

def training(pTfRecordFilenames,
             pLengthTrainDataContainerList,
             pWindowSize,
             pOutputFolder,
             pEpochs,
             pLossWeightPixel,
             pLossWeightDiscriminator,
             pLossWeightAdversarial,
             pLossTypePixel,
             pLossWeightTV,
             pLearningRateGenerator,
             pLearningRateDiscriminator,
             pBeta1,
             pFigureFileFormat,
             pPlotFrequency,
             pFlipSamples,
             pScope,
             pBatchSize,
             pRecordSize,
             pStoredFeaturesDict,
             pNumberSamplesList,
             pNumberOfFactors):

        traindataRecords = [item for sublist in pTfRecordFilenames[0:pLengthTrainDataContainerList] for item in sublist]
        valdataRecords = [item for sublist in pTfRecordFilenames[pLengthTrainDataContainerList:] for item in sublist]

        
        # paramDict["binSize"] = pBinSize
        #because of compatibility checks above, 
        #the following properties are the same with all containers,
        #so just use data from first container
        # nr_factors = container0.nr_factors
        # paramDict["nr_factors"] = nr_factors
        # for i in range(nr_factors):
        #     paramDict["chromFactor_" + str(i)] = container0.factorNames[i]
        nr_trainingSamples = sum(pNumberSamplesList[0:pLengthTrainDataContainerList])
        # storedFeaturesDict = container0.storedFeatures

        #save the training parameters to a file before starting to train
        #(allows recovering the parameters even if training is aborted
        # and only intermediate models are available)
        # parameterFile = os.path.join(pOutputFolder, "trainParams.csv")    
        # with open(parameterFile, "w") as csvfile:
        #     dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        #     dictWriter.writeheader()
        #     dictWriter.writerow(paramDict)

        #build the input streams for training
        shuffleBufferSize = 3*pRecordSize
        trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
        trainDs = trainDs.map(lambda x: records.parse_function(x, pStoredFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if pFlipSamples:
            flippedDs = trainDs.map(lambda a,b: records.mirror_function(a["factorData"], b["out_matrixData"]))
            trainDs = trainDs.concatenate(flippedDs)
        trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
        trainDs = trainDs.batch(pBatchSize, drop_remainder=True)
        trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
        #build the input streams for validation
        validationDs = tf.data.TFRecordDataset(valdataRecords, 
                                                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                compression_type="GZIP")
        validationDs = validationDs.map(lambda x: records.parse_function(x, pStoredFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationDs = validationDs.batch(pBatchSize)
        validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
        
        steps_per_epoch = int( np.floor(nr_trainingSamples / pBatchSize) )
        if pFlipSamples:
            steps_per_epoch *= 2

        hicGanModel = hicGAN.HiCGAN(log_dir=pOutputFolder, 
                                        number_factors=pNumberOfFactors,
                                        loss_weight_pixel=pLossWeightPixel,
                                        loss_weight_adversarial=pLossWeightAdversarial,
                                        loss_weight_discriminator=pLossWeightDiscriminator, 
                                        loss_type_pixel=pLossTypePixel, 
                                        loss_weight_tv=pLossWeightTV, 
                                        input_size=pWindowSize,
                                        learning_rate_generator=pLearningRateGenerator,
                                        learning_rate_discriminator=pLearningRateDiscriminator,
                                        adam_beta_1=pBeta1,
                                        plot_type=pFigureFileFormat,
                                        plot_frequency=pPlotFrequency,
                                        scope=pScope)
        
        hicGanModel.plotModels(pOutputPath=pOutputFolder, pFigureFileFormat=pFigureFileFormat)

        log.info("Starting training at %s" % datetime.now())
        hicGanModel.fit(train_ds=trainDs, epochs=pEpochs, test_ds=validationDs, steps_per_epoch=steps_per_epoch)
        log.info("Training finished at %s" % datetime.now())


def delete_model_files(pTFRecordFiles):
    log.info("Cleaning up temporary files...")
    print(pTFRecordFiles)
    for tfRecordfile in pTFRecordFiles:
        for file in tfRecordfile:
        # print(tfRecordfile)
            if os.path.exists(file):
                os.remove(file)

def main(args=None):
    args = parse_arguments().parse_args(args)
    
    for matrix in args.trainingMatrices + args.validationMatrices:
        if not os.path.exists(matrix):
            msg = "Exiting. Matrix file not found: {:s}".format(matrix)
            print(msg)
            return
        if not matrix.endswith(".cool"):
            msg = "Exiting. Only .cool matrices are supported: {:s}".format(matrix)
            print(msg)
            return
    
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        try:
            for gpu_device in gpu:
                tf.config.experimental.set_memory_growth(gpu_device, True)
        except Exception as e:
            print("Error: {}".format(e))
    
    if args.multiGPUTraining:
        strategy = tf.distribute.MirroredStrategy()
    else:
        log.info("Using single GPU training")
        log.info("Available GPUs: {}".format(gpu))
        log.info("Using GPU: {}".format(args.whichGPU-1))
        # log.info("Using GPU: {}".format(gpu[args.whichGPU].name))
        if args.whichGPU:
            if args.whichGPU > len(gpu):
                raise ValueError("Invalid GPU index: {}".format(args.whichGPU - 1))
            # strategy = tf.distribute.OneDeviceStrategy(device=gpu[args.whichGPU].name)
            strategy = tf.distribute.OneDeviceStrategy(device=f"/GPU:{args.whichGPU-1}")


    with strategy.scope() as scope: 
        tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(args.trainingMatrices, 
                    args.trainingChromosomes, 
                    args.trainingChromatinFolders, 
                    args.validationMatrices, 
                    args.validationChromosomes, 
                    args.validationChromatinFolders,
                    args.windowSize,
                    args.outputFolder,
                    args.batchSize,
                    args.flipSamples,
                    args.figureFileFormat,
                    args.recordSize)
        training(
            pTfRecordFilenames=tfRecordFilenames,
            pLengthTrainDataContainerList=traindataContainerListLength,
            pWindowSize=args.windowSize,
            pOutputFolder=args.outputFolder,
            pEpochs=args.epochs,
            pLossWeightPixel=args.lossWeightPixel,
            pLossWeightDiscriminator=args.lossWeightDiscriminator,
            pLossWeightAdversarial=args.lossWeightAdversarial,
            pLossTypePixel=args.lossTypePixel,
            pLossWeightTV=args.lossWeightTV,
            pLearningRateGenerator=args.learningRateGenerator,
            pLearningRateDiscriminator=args.learningRateDiscriminator,
            pBeta1=args.beta1,
            pFigureFileFormat=args.figureFileFormat,
            pPlotFrequency=args.plotFrequency,
            pFlipSamples=args.flipSamples,
            pScope=scope,
            pBatchSize=args.batchSize,
            pRecordSize=args.recordSize,
            pStoredFeaturesDict=storedFeatures,
            pNumberSamplesList=nr_samples_list,
            pNumberOfFactors=nr_factors
        )
        delete_model_files(pTFRecordFiles=tfRecordFilenames)