import argparse
import numpy as np
import os
import csv
import tensorflow as tf
from .lib import dataContainer
from .lib import records
from .lib import hicGAN
from .lib import utils
import logging
from hicgan._version import __version__

log = logging.getLogger(__name__)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-trm", required=False,
                        type=str,
                        help="Trained generator model to predict from")
    parser.add_argument("--predictionChromosomesFolders", "-tcp", required=False,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=False,
                        type=str,
                        nargs='+',
                        help="Chromosomes the Hi-C matrix should be predicted. Must be available in all bigwig files")
    parser.add_argument("--matrixOutputName", "-mn", required=False,
                        type=str,
                        default="predMatrix.cool",
                        help="Name of the output cool-file")
    parser.add_argument("--parameterOutputFile", "-pf", required=False,
                        type=str,
                        default="predParams.csv",
                        help="Name of the parameter file")
    parser.add_argument("--outputFolder", "-o", required=False,
                        type=str,
                        default="./", 
                        help="Output path for predicted cool-file")
    parser.add_argument("--multiplier", "-mul", required=False,
                        type=int, 
                        default=10, 
                        help="Multiplier for scaling the predicted coolers")
    parser.add_argument("--binSize", "-b", required=False,
                        type=int, 
                        help="Bin size for binning the chromatin features")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int,
                        choices=[64, 128, 256, 512],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile):
    trainedmodel = pTrainedModel
    predictionChromosomesFolders = pPredictionChromosomesFolders
    predictionChromosomes = pPredictionChromosomes
    outputFolder = pOutputFolder
    multiplier = pMultiplier
    binSize = pBinSize
    batchSize = pBatchSize
    windowSize = pWindowSize

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowSize = int(windowSize)
    flankingsize = windowSize

    paramDict = locals().copy()
        
    chromNameList = sorted([x.lstrip("chr") for x in predictionChromosomes])
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixFilePath=None,
                                                  chromatinFolder=predictionChromosomesFolders,
                                                  binSize=binSize)) 
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowSize": windowSize,
                  "flankingSize": flankingsize,
                  "maximumDistance": maxdist}
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return
    container0 = testdataContainerList[0]
    nr_factors = container0.nr_factors
    tfRecordFilenames = []
    sampleSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=outputFolder,
                                                        pRecordSize=None)[0])
        sampleSizeList.append(int(np.ceil(container.getNumberSamples() / batchSize)))
    
    nr_factors = container0.nr_factors
    for container in testdataContainerList:
        container.unloadData() 

    trained_GAN = hicGAN.HiCGAN(log_dir=outputFolder, number_factors=nr_factors)
    trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
    predList = []
    for record, container, nr_samples in zip(tfRecordFilenames, testdataContainerList, sampleSizeList):
        storedFeaturesDict = container.storedFeatures
        testDs = tf.data.TFRecordDataset(record, 
                                            num_parallel_reads=None,
                                            compression_type="GZIP")
        testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testDs = testDs.batch(batchSize, drop_remainder=False)
        testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)
        predArray = trained_GAN.predict(test_ds=testDs, steps_per_record=nr_samples)
        triu_indices = np.triu_indices(windowSize)
        predArray = np.array([np.array(x[triu_indices]) for x in predArray])
        predList.append(predArray)
    predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowSize, pFlankingSize=windowSize) for x in predList]
    predList = [utils.scaleArray(x) * multiplier for x in predList]

    matrixname = os.path.join(outputFolder, pMatrixOutputName)
    log.info("Writing predicted matrix to disk on %s..." % matrixname)   

    utils.writeCooler(pMatrixList=predList, 
                      pBinSizeInt=binSize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

    parameterFile = os.path.join(outputFolder, pParameterOutputFile) 
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)
    
    log.info("Cleaning up temporary files...")
    for tfrecordfile in tfRecordFilenames:
        if os.path.exists(tfrecordfile):
            os.remove(tfrecordfile)


def main(args=None):
    args = parse_arguments().parse_args(args)
    # print(args)
    prediction(args)
