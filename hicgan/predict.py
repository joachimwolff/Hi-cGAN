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
import pickle

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
                        choices=[64, 128, 256, 512, 768, 1024],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    parser.add_argument("--saveMemory", "-sm", action="store_true",
                        help="Enable memory-saving mode for prediction")
    parser.add_argument("--numberOfBatches", "-nb", required=False,
                        type=int,
                        default=20,
                        help="Number of batches to split predictions when --saveMemory is enabled")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default="",
                        help="Specify which GPU to use for training in the single GPU case. E.g. 1, 2, etc.")
    parser.add_argument("--mode", "-m", required=False,
                        type=str,
                        choices=["create-data", "predict", "make-matrix", 'all'],
                        default="all",
                        help="Operation mode: 'create-data' (only create TFRecord data; CPU only), "
                                "'predict' (run prediction using the trained model; requires GPU), "
                                "'make-matrix' (build final matrix from existing predictions; CPU only), "
                                "'all' (run all steps sequentially)")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser

def createDataContainer(pChromNameList, pOutputFolder, pChromatinFolder, pBinSize, 
                        scalefactors, clampfactors, scalematrix, windowSize, flankingsize, maxdist, batchSize):
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in pChromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixFilePath=None,
                                                  chromatinFolder=pChromatinFolder,
                                                  binSize=pBinSize)) 
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
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=pOutputFolder,
                                                        pRecordSize=None)[0])
        sampleSizeList.append(int(np.ceil(container.getNumberSamples() / batchSize)))
    
    nr_factors = container0.nr_factors
    for container in testdataContainerList:
        container.unloadData() 
    return testdataContainerList, tfRecordFilenames, sampleSizeList, nr_factors

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile, pSaveMemory=False, pNumberOfBatches=20, pScope=None, pMode="all"):
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

    if pMode in ["create-data", "all"]:
        testdataContainerList, tfRecordFilenames, sampleSizeList, nr_factors = createDataContainer(
            pChromNameList=chromNameList, 
            pOutputFolder=outputFolder, 
            pChromatinFolder=predictionChromosomesFolders, 
            pBinSize=binSize, 
            scalefactors=scalefactors, 
            clampfactors=clampfactors, 
            scalematrix=scalematrix, 
            windowSize=windowSize, 
            flankingsize=flankingsize, 
            maxdist=maxdist, 
            batchSize=batchSize)
        if pMode == "create-data":
            save_vars = {
                "testdataContainerList": testdataContainerList,
                "tfRecordFilenames": tfRecordFilenames,
                "sampleSizeList": sampleSizeList,
                "nr_factors": nr_factors,
            }

            pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
            with open(pickle_path, "wb") as fh:
                pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("Wrote prediction variables to %s", pickle_path)
            return
    elif pMode == "predict":
        pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with prediction variables not found at %s. Please run with --mode create-data first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        testdataContainerList = loaded_vars["testdataContainerList"]
        tfRecordFilenames = loaded_vars["tfRecordFilenames"]
        sampleSizeList = loaded_vars["sampleSizeList"]
        nr_factors = loaded_vars["nr_factors"]

    if pMode in ["predict", "all"]:
        trained_GAN = hicGAN.HiCGAN(log_dir=outputFolder, number_factors=nr_factors, scope=pScope)
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

            # Predict in batches to manage memory usage
            predBatches = []
            if pSaveMemory:
                total_batches = pNumberOfBatches  # You can adjust this to 5 or more
                batch_size = nr_samples // total_batches
                log.debug("Predicting on dataset in %d batches to save memory..." % total_batches)
                for batch_idx in range(0, nr_samples, batch_size):
                    batch_pred = trained_GAN.predict(test_ds=testDs.skip(batch_idx).take(batch_size), steps_per_record=1)
                    triu_indices = np.triu_indices(windowSize)
                    batch_pred_array = np.array([np.array(x[triu_indices], dtype=np.float16) for x in batch_pred], dtype=np.float16)
                    predBatches.append(batch_pred_array)
                predArray = np.concatenate(predBatches, axis=0).astype(np.float16)
                del predBatches
                predList.append(predArray)
                del predArray
            else:
                log.debug("Predicting on full dataset...")
                predDs = trained_GAN.predict(test_ds=testDs, steps_per_record=1)
                log.debug("Converting predictions to numpy arrays...")
                triu_indices = np.triu_indices(windowSize)
                predArray = np.array([np.array(x[triu_indices]) for x in predDs])
                predList.append(predArray)
                del predDs
                del predArray
        log.debug("Prediction on all chromosomes completed.")

        log.info("Cleaning up temporary files...")
        for tfrecordfile in tfRecordFilenames:
            if os.path.exists(tfrecordfile):
                os.remove(tfrecordfile)
        if pMode == "predict":
            save_vars = {
                "predList": predList,
                "chromNameList": chromNameList,
            }

            pickle_path = os.path.join(outputFolder, "predictions.pkl")
            with open(pickle_path, "wb") as fh:
                pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("Wrote predictions to %s", pickle_path)
            return
    if pMode == "make-matrix":
        pickle_path = os.path.join(outputFolder, "predictions.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with predictions not found at %s. Please run with --mode predict first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        predList = loaded_vars["predList"]
        chromNameList = loaded_vars["chromNameList"]

    if pMode in ["make-matrix", "all"]:
        log.debug("Rebuilding full matrices from predicted triangles...")
        predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowSize, pFlankingSize=windowSize, pSaveMemory=pSaveMemory) for x in predList]
        log.debug("Scaling predicted matrices...")
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
        
       


def main(args=None):
    args = parse_arguments().parse_args(args)
    # print(args)
    # 1. Get all physical GPUs

    if args.mode == "all" or args.mode == "predict":
        physical_gpus = tf.config.list_physical_devices('GPU')

        if physical_gpus:
            try:
                # Calculate index (assuming args.whichGPU is 1-based data like 1, 2, 3...)
                index = args.whichGPU - 1
                if index < 0 or index >= len(physical_gpus):
                    raise ValueError(f"Invalid GPU index: {index}. Available: {len(physical_gpus)}")

                # 2. Key Step: Make ONLY the selected GPU visible to TensorFlow
                # This prevents TF from touching the memory of the other GPU.
                tf.config.set_visible_devices(physical_gpus[index], 'GPU')

                # 3. Set memory growth only on the visible device
                # Note: After setting visibility, we interact with the specific physical object
                tf.config.experimental.set_memory_growth(physical_gpus[index], True)

                log.info(f"Physically selected GPU: {physical_gpus[index].name}")
                log.info("Other GPUs are now invisible to TensorFlow.")

            except Exception as e:
                print("Error setting up GPU: {}".format(e))
        else:
            log.info("No GPUs found. Running on CPU.")
        
    prediction(pTrainedModel=args.trainedModel,
        pPredictionChromosomesFolders=args.predictionChromosomesFolders,
        pPredictionChromosomes=args.predictionChromosomes,
        pOutputFolder=args.outputFolder,
        pMultiplier=args.multiplier,
        pBinSize=args.binSize,
        pBatchSize=args.batchSize,
        pWindowSize=args.windowSize,
        pMatrixOutputName=args.matrixOutputName,
        pParameterOutputFile=args.parameterOutputFile,
        pSaveMemory=args.saveMemory,
        pNumberOfBatches=args.numberOfBatches,
        pScope=None, 
        pMode=args.mode)
