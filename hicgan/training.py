import csv
import os
import click
import numpy as np
import tensorflow as tf
import hicGAN
import dataContainer
import records
import argparse
# import tensorflow as tf
from datetime import datetime

import logging
log = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingMatrices", "-tm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for training.")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=True,
                        type=str,
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromosomesFolders", "-tcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--validationMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=True,
                        type=str,
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromosomesFolders", "-vcp", required=True,
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
    parser.add_argument("--pretrainedIntroModel", "-ptm", required=False,
                        type=str,
                        help="Pretrained model for 1D-2D conversion of inputs.")
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

    return parser
def create_container(chrom, matrix, chromatinpath):
        container = dataContainer.DataContainer(chromosome=chrom,
                                                matrixFilePath=matrix,
                                                chromatinFolder=chromatinpath)
        return container

def training(trainingMatrices, 
             trainingChromosomes, 
             trainingChromosomesFolders, 
             validationMatrices, 
             validationChromosomes, 
             validationChromosomesFolders,
             windowSize,
             outputFolder,
             epochs,
             batchSize,
             lossWeightPixel,
             lossWeightDiscriminator,
             lossWeightAdversarial,
             lossTypePixel,
             lossWeightTV,
             learningRateGenerator,
             learningRateDiscriminator,
             beta1,
             flipSamples,
             pretrainedIntroModel,
             figureFileFormat,
             recordSize,
             plotFrequency, scope=None):

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    #few constants
    # windowSize = int(windowSize)
    debugstate = None
    paramDict = locals().copy()

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    trainChromNameList = trainingChromosomes.replace(",","")
    trainChromNameList = trainChromNameList.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    valChromNameList = validationChromosomes.replace(",","")
    valChromNameList = valChromNameList.rstrip().split(" ")
    valChromNameList = [x.lstrip("chr") for x in valChromNameList]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList

    #ensure there are as many matrices as chromatin paths
    if len(trainingMatrices) != len(trainingChromosomesFolders):
        msg = "Number of train matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(trainingMatrices), len(trainingChromosomesFolders))
        raise SystemExit(msg)
    if len(validationMatrices) != len(validationChromosomesFolders):
        msg = "Number of validation matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(validationMatrices), len(validationChromosomesFolders))
        raise SystemExit(msg)

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    import concurrent.futures

    

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in trainChromNameList:
            for matrix, chromatinpath in zip(trainingMatrices, trainingChromosomesFolders):
                future = executor.submit(create_container, chrom, matrix, chromatinpath)
                traindataContainerList.append(future.result())

    #prepare the validation data containers. No data is loaded yet.
    valdataContainerList = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in valChromNameList:
            for matrix, chromatinpath in zip(validationMatrices, validationChromosomesFolders):
                future = executor.submit(create_container, chrom, matrix, chromatinpath)
                valdataContainerList.append(future.result())

    #define the load params for the containers
    loadParams = {"scaleFeatures": True,
                  "clampFeatures": False,
                  "scaleTargets": True,
                  "windowSize": windowSize,
                  "flankingSize": windowSize,
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
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=outputFolder,
                                                        pRecordSize=recordSize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                         outpath=outputFolder,
                                         figureFileFormat=figureFileFormat)
            container.saveMatrix(outputpath=outputFolder, index=idx)
        nr_samples_list.append(container.getNumberSamples())
    #data is no longer needed
    for container in traindataContainerList + valdataContainerList:
        container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    valdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]

    #different binSizes are ok
    #not clear which binSize to use for prediction when they differ during training.
    #For now, store the max. 
    binSize = max([container.binSize for container in traindataContainerList])
    paramDict["binSize"] = binSize
    #because of compatibility checks above, 
    #the following properties are the same with all containers,
    #so just use data from first container
    nr_factors = container0.nr_factors
    paramDict["nr_factors"] = nr_factors
    for i in range(nr_factors):
        paramDict["chromFactor_" + str(i)] = container0.factorNames[i]
    nr_trainingSamples = sum(nr_samples_list[0:len(traindataContainerList)])
    storedFeaturesDict = container0.storedFeatures

    #save the training parameters to a file before starting to train
    #(allows recovering the parameters even if training is aborted
    # and only intermediate models are available)
    parameterFile = os.path.join(outputFolder, "trainParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)

    #build the input streams for training
    shuffleBufferSize = 3*recordSize
    trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if flipSamples:
        flippedDs = trainDs.map(lambda a,b: records.mirror_function(a["factorData"], b["out_matrixData"]))
        trainDs = trainDs.concatenate(flippedDs)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchSize, drop_remainder=True)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    #build the input streams for validation
    validationDs = tf.data.TFRecordDataset(valdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchSize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
    
    steps_per_epoch = int( np.floor(nr_trainingSamples / batchSize) )
    if flipSamples:
        steps_per_epoch *= 2
    if pretrainedIntroModel is None:
        pretrainedIntroModel = ""

    hicGanModel = hicGAN.HiCGAN(log_dir=outputFolder, 
                                    number_factors=nr_factors,
                                    loss_weight_pixel=lossWeightPixel,
                                    loss_weight_adversarial=lossWeightAdversarial,
                                    loss_weight_discriminator=lossWeightDiscriminator, 
                                    loss_type_pixel=lossTypePixel, 
                                    loss_weight_tv=lossWeightTV, 
                                    input_size=windowSize,
                                    learning_rate_generator=learningRateGenerator,
                                    learning_rate_discriminator=learningRateDiscriminator,
                                    adam_beta_1=beta1,
                                    plot_type=figureFileFormat,
                                    plot_frequency=plotFrequency,
                                    pretrained_model_path=pretrainedIntroModel,
                                    scope=scope)
    
    hicGanModel.plotModels(pOutputPath=outputFolder, pFigureFileFormat=figureFileFormat)

    hicGanModel.fit(train_ds=trainDs, epochs=epochs, test_ds=validationDs, steps_per_epoch=steps_per_epoch)

def main(args=None):
    args = parse_arguments().parse_args(args)
    print(args)
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        try:
            for gpu_device in gpu:
                tf.config.experimental.set_memory_growth(gpu_device, True)
        except Exception as e:
            print("Error: {}".format(e))
    
    strategy = tf.distribute.MirroredStrategy()


    with strategy.scope() as scope: 
        training(
            trainingMatrices=args.trainingMatrices,
            trainingChromosomes=args.trainingChromosomes,
            trainingChromosomesFolders=args.trainingChromosomesFolders,
            validationMatrices=args.validationMatrices,
            validationChromosomes=args.validationChromosomes,
            validationChromosomesFolders=args.validationChromosomesFolders,
            windowSize=args.windowSize,
            outputFolder=args.outputFolder,
            epochs=args.epochs,
            batchSize=args.batchSize,
            lossWeightPixel=args.lossWeightPixel,
            lossWeightDiscriminator=args.lossWeightDiscriminator,
            lossWeightAdversarial=args.lossWeightAdversarial,
            lossTypePixel=args.lossTypePixel,
            lossWeightTV=args.lossWeightTV,
            learningRateGenerator=args.learningRateGenerator,
            learningRateDiscriminator=args.learningRateDiscriminator,
            beta1=args.beta1,
            flipSamples=args.flipSamples,
            pretrainedIntroModel=args.pretrainedIntroModel,
            figureFileFormat=args.figureFileFormat,
            recordSize=args.recordSize,
            plotFrequency=args.plotFrequency,
            scope=scope
        )  # pylint: disable=no-value-for-parameter
