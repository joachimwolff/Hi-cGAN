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
import tarfile
import gzip
import tempfile
import io
import cooler
log = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-tm", required=True,
                        type=str,
                        help="Trained generator model to predict from")
    parser.add_argument("--predictionChromosomesFolders", "-pcf", required=True,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=True,
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
                        default=1000, 
                        help="Multiplier for scaling the predicted coolers")
    # parser.add_argument("--binSize", "-b", required=True,
    #                     type=int,
    #                     help="Bin size for binning the chromatin features")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowSize", "-ws", required=True,
                        type=int,
                        choices=[64, 128, 256, 512],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

def prediction(trainedmodel, predictionChromosomesFolders, predictionChromosomes, outputFolder, multiplier, binSize, batchSize, windowSize, matrixOutputName, parameterOutputFile):
    

    os.makedirs(outputFolder, exist_ok=True)

    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowSize = int(windowSize)
    flankingsize = windowSize

    paramDict = locals().copy()
        
    #extract chromosome names from the input
    chromNameList = predictionChromosomes#.replace(",", " ").rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixFilePath=None,
                                                  chromatinFolder=predictionChromosomesFolders,
                                                  binSize=binSize)) 
    #define the load params for the containers
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowSize": windowSize,
                  "flankingSize": flankingsize,
                  "maximumDistance": maxdist}
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
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=outputFolder,
                                                        pRecordSize=None)[0]) #list with 1 entry
        sampleSizeList.append( int( np.ceil(container.getNumberSamples() / batchSize) ) )
    
    nr_factors = container0.nr_factors
    #data is no longer needed, unload it
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
        testDs = testDs.batch(batchSize, drop_remainder=False) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
        #if validationmatrix is not None:
        #    testDs = testDs.map(lambda x, y: x) #drop the target matrices (they are for evaluation)
        testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)
        predArray = trained_GAN.predict(test_ds=testDs, steps_per_record=nr_samples)
        triu_indices = np.triu_indices(windowSize)
        predArray = np.array( [np.array(x[triu_indices]) for x in predArray] )
        predList.append(predArray)
    predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowSize, pFlankingSize=windowSize) for x in predList]
    predList = [utils.scaleArray(x) * multiplier for x in predList]

    matrixname = os.path.join(outputFolder, matrixOutputName)
    log.info("Writing predicted matrix to disk on %s..." % matrixname)   

    predicted_matrix_cooler_raw = utils.writeCooler(pMatrixList=predList, 
                      pBinSizeInt=binSize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

    parameterFile = os.path.join(outputFolder, parameterOutputFile) 
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)
    
    log.info("Cleaning up temporary files...")
    for tfrecordfile in tfRecordFilenames:
        if os.path.exists(tfrecordfile):
            os.remove(tfrecordfile)

    return predicted_matrix_cooler_raw

def extract_specific_file_to_temp_dir(tar_gz_path, file_to_extract):
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # Check if the file exists in the archive
        if file_to_extract in tar.getnames():
            # Extract the specific file to the temporary directory
            tar.extract(file_to_extract, path=temp_dir.name)
            extracted_file_path = os.path.join(temp_dir.name, file_to_extract)
            return extracted_file_path, temp_dir
        else:
            print(f"{file_to_extract} not found in the archive")
            temp_dir.cleanup()
            return None, None

def list_files_in_tar_gz(tar_gz_path):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        file_names = tar.getnames()
    return file_names

def main():
    args = parse_arguments()

    predictionChromosomesFolders = args.predictionChromosomesFolders
    predictionChromosomes = args.predictionChromosomes
    outputFolder = args.outputFolder
    multiplier = args.multiplier
    # binSize = args.binSize
    batchSize = args.batchSize
    windowSize = args.windowSize
    matrixOutputName = args.matrixOutputName
    parameterOutputFile = args.parameterOutputFile
   
    
    generator_files = list_files_in_tar_gz(args.trainedModel)
    log.debug(f"Generator files: {generator_files}")
    
    resolutions_predicted = {}
        # Iterate through each model directory and load the model
    for generator in generator_files:
        
        extracted_file_path, temp_dir = extract_specific_file_to_temp_dir(args.trainedModel, generator)

        # extracted_file_path = extract_specific_file(args.trainedModel, generator, extract_path)

        # args.outputFolder = os.path.join(args.outputFolder, os.path.basename(model_dir))
        log.debug(f"Predicting with model in {extracted_file_path}")
        log.debug(f"Output folder: {args.outputFolder}")
        binSize = int(os.path.basename(generator).split(".")[0])
        log.debug(f"Bin size: {binSize}")
        predicted_matrix_cooler_raw = prediction(
            extracted_file_path, predictionChromosomesFolders, predictionChromosomes, outputFolder, 
            multiplier, binSize, batchSize, windowSize, str(binSize) + "_" + matrixOutputName, parameterOutputFile
        )
        resolutions_predicted[binSize] = predicted_matrix_cooler_raw
        temp_dir.cleanup()

    for resolution, matrix in resolutions_predicted.items():
        log.info(f"Predicted matrix at resolution {resolution} written to {matrix}")
        cooler.create_cooler(matrixOutputName + "::/resolutions/" + str(resolution), bins=matrix["bins"], pixels=matrix["pixels"], dtypes=matrix["dtypes"], ordered=matrix["ordered"], metadata=matrix["metadata"], mode='a')
