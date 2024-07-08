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
# tf.get_logger().setLevel(logging.INFO)
# tf.debugging.set_log_device_placement(True)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainMatrices", "-tm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for training.")
    parser.add_argument("--trainChroms", "-tchroms", required=True,
                        type=str,
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainChromPaths", "-tcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files).")
    parser.add_argument("--valMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Cooler matrices for validation.")
    parser.add_argument("--valChroms", "-vchroms", required=True,
                        type=str,
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--valChromPaths", "-vcp", required=True,
                        type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files).")
    parser.add_argument("--windowsize", "-ws", required=True,
                        type=str, choices=["64", "128", "256", "512"],
                        help="Windowsize for submatrices.")
    parser.add_argument("--outfolder", "-o", required=True,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=True,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batchsize", "-bs", required=False,
                        type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--lossWeightPixel", "-lwp", required=False,
                        type=float,
                        default=100.0,
                        help="Loss weight for L1/L2 error of generator.")
    parser.add_argument("--lossWeightDisc", "-lwd", required=False,
                        type=float,
                        default=0.5,
                        help="Loss weight (multiplicator) for the discriminator loss.")
    parser.add_argument("--lossTypePixel", "-ltp", required=False,
                        type=str, choices=["L1", "L2"],
                        default="L1",
                        help="Type of per-pixel loss to use for the generator.")
    parser.add_argument("--lossWeightTv", "-lwt", required=False,
                        type=float,
                        default=1e-10,
                        help="Loss weight for Total-Variation-loss of generator.")
    parser.add_argument("--lossWeightAdv", "-lwa", required=False,
                        type=float,
                        default=1.0,
                        help="Loss weight for adversarial loss in generator.")
    parser.add_argument("--learningRateGen", "-lrg", required=False,
                        type=float,
                        default=2e-5,
                        help="Learning rate for Adam optimizer of generator.")
    parser.add_argument("--learningRateDisc", "-lrd", required=False,
                        type=float,
                        default=1e-6,
                        help="Learning rate for Adam optimizer of discriminator.")
    parser.add_argument("--beta1", "-b1", required=False,
                        type=float,
                        default=0.5,
                        help="Beta1 parameter for Adam optimizers (gen. and disc.)")
    parser.add_argument("--flipsamples", "-fs", required=False,
                        action='store_true',
                        help="Flip training matrices and chromatin features (data augmentation).")
    parser.add_argument("--pretrainedIntroModel", "-ptm", required=False,
                        type=str,
                        help="Pretrained model for 1D-2D conversion of inputs.")
    parser.add_argument("--figuretype", "-ft", required=False,
                        type=str, choices=["png", "pdf", "svg"],
                        default="png",
                        help="Figure type for all plots.")
    parser.add_argument("--recordsize", "-rs", required=False,
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
                                                matrixfilepath=matrix,
                                                chromatinFolder=chromatinpath)
        return container

def training(trainmatrices, 
             trainchroms, 
             trainchrompaths, 
             valmatrices, 
             valchroms, 
             valchrompaths,
             windowsize,
             outfolder,
             epochs,
             batchsize,
             lossweightpixel,
             lossweightdisc,
             lossweightadv,
             losstypepixel,
             lossweighttv,
             learningrategen,
             learningratedisc,
             beta1,
             flipsamples,
             pretrainedintromodel,
             figuretype,
             recordsize,
             plotfrequency, scope=None):

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    #few constants
    windowsize = int(windowsize)
    debugstate = None
    paramDict = locals().copy()

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    trainChromNameList = trainchroms.replace(",","")
    trainChromNameList = trainChromNameList.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    valChromNameList = valchroms.replace(",","")
    valChromNameList = valChromNameList.rstrip().split(" ")
    valChromNameList = [x.lstrip("chr") for x in valChromNameList]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList

    #ensure there are as many matrices as chromatin paths
    if len(trainmatrices) != len(trainchrompaths):
        msg = "Number of train matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(trainmatrices), len(trainchrompaths))
        raise SystemExit(msg)
    if len(valmatrices) != len(valchrompaths):
        msg = "Number of validation matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(valmatrices), len(valchrompaths))
        raise SystemExit(msg)

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    import concurrent.futures

    

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in trainChromNameList:
            for matrix, chromatinpath in zip(trainmatrices, trainchrompaths):
                future = executor.submit(create_container, chrom, matrix, chromatinpath)
                traindataContainerList.append(future.result())

    #prepare the validation data containers. No data is loaded yet.
    valdataContainerList = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chrom in valChromNameList:
            for matrix, chromatinpath in zip(valmatrices, valchrompaths):
                future = executor.submit(create_container, chrom, matrix, chromatinpath)
                valdataContainerList.append(future.result())

    #define the load params for the containers
    loadParams = {"scaleFeatures": True,
                  "clampFeatures": False,
                  "scaleTargets": True,
                  "windowsize": windowsize,
                  "flankingsize": windowsize,
                  "maxdist": None}
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
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outfolder,
                                                        pRecordSize=recordsize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                         outpath=outfolder,
                                         figuretype=figuretype)
            container.saveMatrix(outputpath=outfolder, index=idx)
        nr_samples_list.append(container.getNumberSamples())
    #data is no longer needed
    for container in traindataContainerList + valdataContainerList:
        container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    valdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]

    #different binsizes are ok
    #not clear which binsize to use for prediction when they differ during training.
    #For now, store the max. 
    binsize = max([container.binsize for container in traindataContainerList])
    paramDict["binsize"] = binsize
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
    parameterFile = os.path.join(outfolder, "trainParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)

    #build the input streams for training
    shuffleBufferSize = 3*recordsize
    trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if flipsamples:
        flippedDs = trainDs.map(lambda a,b: records.mirror_function(a["factorData"], b["out_matrixData"]))
        trainDs = trainDs.concatenate(flippedDs)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchsize, drop_remainder=True)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    #build the input streams for validation
    validationDs = tf.data.TFRecordDataset(valdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
    
    steps_per_epoch = int( np.floor(nr_trainingSamples / batchsize) )
    if flipsamples:
        steps_per_epoch *= 2
    if pretrainedintromodel is None:
        pretrainedintromodel = ""

    hicGanModel = hicGAN.HiCGAN(log_dir=outfolder, 
                                    number_factors=nr_factors,
                                    loss_weight_pixel=lossweightpixel,
                                    loss_weight_adversarial=lossweightadv,
                                    loss_weight_discriminator=lossweightdisc, 
                                    loss_type_pixel=losstypepixel, 
                                    loss_weight_tv=lossweighttv, 
                                    input_size=windowsize,
                                    learning_rate_generator=learningrategen,
                                    learning_rate_discriminator=learningratedisc,
                                    adam_beta_1=beta1,
                                    plot_type=figuretype,
                                    plot_frequency=plotfrequency,
                                    pretrained_model_path=pretrainedintromodel,
                                    scope=scope)
    
    hicGanModel.plotModels(outputpath=outfolder, figuretype=figuretype)

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
            trainmatrices=args.trainMatrices,
            trainchroms=args.trainChroms,
            trainchrompaths=args.trainChromPaths,
            valmatrices=args.valMatrices,
            valchroms=args.valChroms,
            valchrompaths=args.valChromPaths,
            windowsize=args.windowsize,
            outfolder=args.outfolder,
            epochs=args.epochs,
            batchsize=args.batchsize,
            lossweightpixel=args.lossWeightPixel,
            lossweightdisc=args.lossWeightDisc,
            lossweightadv=args.lossWeightAdv,
            losstypepixel=args.lossTypePixel,
            lossweighttv=args.lossWeightTv,
            learningrategen=args.learningRateGen,
            learningratedisc=args.learningRateDisc,
            beta1=args.beta1,
            flipsamples=args.flipsamples,
            pretrainedintromodel=args.pretrainedIntroModel,
            figuretype=args.figuretype,
            recordsize=args.recordsize,
            plotfrequency=args.plotFrequency,
            scope=scope
        )  # pylint: disable=no-value-for-parameter

if __name__ == "__main__":
    main()