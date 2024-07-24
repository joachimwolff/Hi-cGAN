import csv
import os
import numpy as np
import tensorflow as tf
import concurrent.futures
import argparse
from datetime import datetime
import cooler
import tarfile
import gzip

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
                        type=str,
                        nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromosomesFolders", "-tcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--validationMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--validationChromosomes", "-vchroms", required=True,
                        type=str,
                        nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromosomesFolders", "-vcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    parser.add_argument("--windowSize", "-ws", required=True,
                        type=int, choices=[16, 32,64, 128, 256, 512],
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
    parser.add_argument("--interChromosomalTraining", "-ict", required=False,
                        action='store_true',
                        help="Train additional on inter-chromosomal data.")
    parser.add_argument("--figureFileFormat", "-ft", required=False,
                        type=str, choices=["png", "pdf", "svg"],
                        default="png",
                        help="Figure type for all plots.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int,
                        default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")
    parser.add_argument("--smallestResolution", "-sr", required=False,
                        type=int,
                        default=100000,
                        help="Smallest resolution of the matrices for training and validation. Smaller resolutions might cause issues with large window sizes.")
    parser.add_argument("--plotFrequency", "-pfreq", required=False,
                        type=int,
                        default=10,
                        help="Update loss over epoch plots after this number of epochs.")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

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
             figureFileFormat,
             recordSize,
             plotFrequency, 
             interChoromosomalTraining,
             scope=None):

    interChoromosomalTraining = False
    os.makedirs(outputFolder, exist_ok=True)
    #few constants
    # windowSize = int(windowSize)
    debugstate = None
    paramDict = locals().copy()

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    trainChromNameList = trainingChromosomes
    # trainChromNameList = trainChromNameList.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    valChromNameList = validationChromosomes
    # valChromNameList = valChromNameList.rstrip().split(" ")
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
    valdataContainerList = []

    
    if interChoromosomalTraining:
        #inter-chromosomal training
        #create a container for each chromosome pair
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for chrom1 in trainChromNameList:
                for chrom2 in trainChromNameList:
                    if chrom1 == chrom2:
                        continue
                    for matrix, chromatinpath in zip(trainingMatrices, trainingChromosomesFolders):
                        future = executor.submit(create_container, [chrom1, chrom2], matrix, chromatinpath)
                        traindataContainerList.append(future.result())
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for chrom1 in valChromNameList:
                for chrom2 in valChromNameList:
                    if chrom1 == chrom2:
                        continue
                    for matrix, chromatinpath in zip(validationMatrices, validationChromosomesFolders):
                        future = executor.submit(create_container, [chrom1, chrom2], matrix, chromatinpath)
                        valdataContainerList.append(future.result())
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for chrom in trainChromNameList:
                for matrix, chromatinpath in zip(trainingMatrices, trainingChromosomesFolders):
                    future = executor.submit(create_container, chrom, matrix, chromatinpath)
                    traindataContainerList.append(future.result())

        #prepare the validation data containers. No data is loaded yet.

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
                                    scope=scope)
    
    hicGanModel.plotModels(pOutputPath=outputFolder, pFigureFileFormat=figureFileFormat)

    log.info("Starting training at %s" % datetime.now())
    hicGanModel.fit(train_ds=trainDs, epochs=epochs, test_ds=validationDs, steps_per_epoch=steps_per_epoch)
    log.info("Training finished at %s" % datetime.now())
    log.info("Cleaning up temporary files...")
    for tfRecordfile in traindataRecords + valdataRecords:
        if os.path.exists(tfRecordfile):
            os.remove(tfRecordfile)

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

    
    for matrix in args.trainingMatrices + args.validationMatrices:
        if not (cooler.fileops.is_cooler(matrix) or cooler.fileops.is_multires_file(matrix) or os.path.exists(matrix) or matrix.endswith(".mcool")):
            msg = "Exiting. Invalid cooler file: {:s}".format(matrix)
            print(msg)
            return
        # if not cooler.fileops.is_multires_file(matrix):
        #     msg = "Exiting. Invalid cooler file: {:s}".format(matrix)
        #     print(msg)
        #     return
        # if not os.path.exists(matrix):
        #     msg = "Exiting. Matrix file not found: {:s}".format(matrix)
        #     print(msg)
        #     return
        # if not matrix.endswith(".mcool"):
        #     msg = "Exiting. Only .mcool matrices are supported: {:s}".format(matrix)
        #     print(msg)
        #     return
        
            
    if cooler.fileops.is_multires_file(args.trainingMatrices[0]):
        submatrices = cooler.fileops.list_coolers(args.trainingMatrices[0])
        submatrices_write_out = []
        matrix_resolutions_training = [[] for _ in range(len(submatrices))]
        matrix_resolutions_validation = [[] for _ in range(len(submatrices))]

        # chromatine_paths_training = [args.trainingChromosomesFolders] * len(submatrices)
        # chromatine_paths_validation = [args.validationChromosomesFolders] * len(submatrices)
        log.debug("args.trainingMatrices: %s" % args.trainingMatrices)
        for matrix in args.trainingMatrices:
            for i, submatrix in enumerate(submatrices):
                if not cooler.fileops.is_cooler(matrix + "::" + submatrix):
                    msg = "Exiting. Invalid cooler file: {:s}. All matrices need to have the same resolutions.".format(matrix)
                    print(msg)
                    return
                cooler_file = cooler.Cooler(matrix + "::" + submatrix)
                resolution = cooler_file.info['bin-size']
                if resolution <= args.smallestResolution:
                    matrix_resolutions_training[i].append(matrix + "::" + submatrix)
                    submatrices_write_out.append(submatrix)

        for matrix in args.validationMatrices:
            for i, submatrix in enumerate(submatrices):
                if not cooler.fileops.is_cooler(matrix + "::" + submatrix):
                    msg = "Exiting. Invalid cooler file: {:s}. All matrices need to have the same resolutions.".format(matrix)
                    print(msg)
                    return
                cooler_file = cooler.Cooler(matrix + "::" + submatrix)
                resolution = cooler_file.info['bin-size']
                if resolution <= args.smallestResolution:
                    matrix_resolutions_validation[i].append(matrix + "::" + submatrix)
                    # submatrices_write_out.append(submatrix)
        matrix_resolutions_training = [sublist for sublist in matrix_resolutions_training if sublist]
        matrix_resolutions_validation = [sublist for sublist in matrix_resolutions_validation if sublist]

        log.debug("Submatrices: %s" % submatrices)
        log.debug("Training matrices: %s" % matrix_resolutions_training)
        log.debug("Validation matrices: %s" % matrix_resolutions_validation)
    else:
        matrix_resolutions_training = [args.trainingMatrices]
        matrix_resolutions_validation = [args.validationMatrices]
        submatrices = ["single"]
    for submatrix, training_matrices, validation_matrices in zip(submatrices, matrix_resolutions_training, matrix_resolutions_validation):

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope() as scope: 
            training(
                trainingMatrices=training_matrices,
                trainingChromosomes=args.trainingChromosomes,
                trainingChromosomesFolders=args.trainingChromosomesFolders,
                validationMatrices=validation_matrices,
                validationChromosomes=args.validationChromosomes,
                validationChromosomesFolders=args.validationChromosomesFolders,
                windowSize=args.windowSize,
                outputFolder=os.path.join(args.outputFolder, submatrix.split('/')[-1]),
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
                figureFileFormat=args.figureFileFormat,
                recordSize=args.recordSize,
                plotFrequency=args.plotFrequency,
                interChoromosomalTraining=args.interChromosomalTraining,
                scope=scope
            )  # pylint: disable=no-value-for-parameter

    with tarfile.open(os.path.join(args.outputFolder, 'trainedModel.tar.gz'), "w:gz") as tar:
        for submatrix in submatrices_write_out:
            file_name = os.path.join(args.outputFolder, submatrix.split('/')[-1], "generator_final.keras")
            tar.add(file_name, arcname=submatrix.split('/')[-1] + '.keras')
    
    # with h5py.File(os.path.join(args.outputFolder, 'trainedModel.hdf'), 'w') as hdf5_file:
    #     for submatrix in submatrices:
    #         file_name = os.path.join(args.outputFolder, submatrix.split('/')[-1], "generator_final.keras")
    #         with open(file_name, 'rb') as f:
    #             file_data = f.read()
    #             # Store the file data in the HDF5 file
    #             hdf5_file.create_dataset(submatrix.split('/')[-1] + '.keras', data=file_data)
