import argparse
import numpy as np
import os
import csv
import tensorflow as tf
import dataContainer
import records
import hicGAN
import utils
import logging
from hicgan._version import __version__

log = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-trm", required=True,
                        type=str,
                        help="Trained generator model to predict from")
    parser.add_argument("--testChromPath", "-tcp", required=True,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--testChroms", "-tchroms", required=True,
                        type=str,
                        help="Chromosomes for testing. Must be available in all bigwig files")
    parser.add_argument("--outfolder", "-o", required=False,
                        type=str,
                        default="./", 
                        help="Output path for predicted coolers")
    parser.add_argument("--multiplier", "-mul", required=False,
                        type=int, 
                        default=10, 
                        help="Multiplier for scaling the predicted coolers")
    parser.add_argument("--binsize", "-b", required=True,
                        type=int, 
                        help="Bin size for binning the chromatin features")
    parser.add_argument("--batchsize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowsize", "-ws", required=True,
                        type=str,
                        choices=["64", "128", "256", "512"],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    return parser.parse_args()

def prediction(args):
    trainedmodel = args.trainedModel
    testchrompath = args.testChromPath
    testchroms = args.testChroms
    outfolder = args.outfolder
    multiplier = args.multiplier
    binsize = args.binsize
    batchsize = args.batchsize
    windowsize = args.windowsize

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowsize = int(windowsize)
    flankingsize = windowsize

    paramDict = locals().copy()
        
    #extract chromosome names from the input
    chromNameList = testchroms.replace(",", " ").rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixfilepath=None,
                                                  chromatinFolder=testchrompath,
                                                  binsize=binsize)) 
    #define the load params for the containers
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowsize": windowsize,
                  "flankingsize": flankingsize,
                  "maxdist": maxdist}
    #now load the data and write TFRecords, one container at a time.
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = testdataContainerList[0]
    nr_factors = container0.nr_factors
    tfRecordFilenames = []
    sampleSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outfolder,
                                                        pRecordSize=None)[0]) #list with 1 entry
        sampleSizeList.append( int( np.ceil(container.getNumberSamples() / batchsize) ) )
    
    nr_factors = container0.nr_factors
    #data is no longer needed, unload it
    for container in testdataContainerList:
        container.unloadData() 

    trained_GAN = hicGAN.HiCGAN(log_dir=outfolder, number_factors=nr_factors)
    trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
    predList = []
    for record, container, nr_samples in zip(tfRecordFilenames, testdataContainerList, sampleSizeList):
        storedFeaturesDict = container.storedFeatures
        testDs = tf.data.TFRecordDataset(record, 
                                            num_parallel_reads=None,
                                            compression_type="GZIP")
        testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testDs = testDs.batch(batchsize, drop_remainder=False) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
        #if validationmatrix is not None:
        #    testDs = testDs.map(lambda x, y: x) #drop the target matrices (they are for evaluation)
        testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)
        predArray = trained_GAN.predict(test_ds=testDs, steps_per_record=nr_samples)
        triu_indices = np.triu_indices(windowsize)
        predArray = np.array( [np.array(x[triu_indices]) for x in predArray] )
        predList.append(predArray)
    predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowsize, pFlankingSize=windowsize) for x in predList]
    predList = [utils.scaleArray(x) * multiplier for x in predList]

    matrixname = os.path.join(outfolder, "predMatrix.cool")
    utils.writeCooler(pMatrixList=predList, 
                      pBinSizeInt=binsize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

    parameterFile = os.path.join(outfolder, "predParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)
    
    for tfrecordfile in tfRecordFilenames:
        if os.path.exists(tfrecordfile):
            os.remove(tfrecordfile)


def main():
    args = parse_arguments()
    prediction(args)
