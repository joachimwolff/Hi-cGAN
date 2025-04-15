import icecream as ic
import argparse
import numpy as np
import pandas as pd
import os
from hicrep.utils import readMcool
from hicrep import hicrepSCC
from hicexplorer import hicFindTADs
import logging
from hicgan._version import __version__
from hicgan.training import training, create_data, delete_model_files
from hicgan.predict import prediction
from hicgan.lib.utils import computePearsonCorrelation
import joblib
import traceback
import shutil
import cooler

log = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--matrix1", "-m1", required=True,
                        type=str,
                        help="Path to the first Hi-C matrix file")
    parser.add_argument("--matrix2", "-m2", required=True,
                        type=str,
                        help="Path to the second Hi-C matrix file")
    parser.add_argument("--output", "-o", required=True,
                        type=str,
                        help="Output directory")
    parser.add_argument("--binSize", "-bs", required=True,
                        type=int,
                        help="Bin size of the Hi-C matrix")
    parser.add_argument("--chromosomes", "-chr", required=False,
                        type=str, nargs='+',
                        help="Chromosomes compute the similarity on. If empty, all chromosomes will be used")
    parser.add_argument("--cellType", "-ct", required=True,
                        type=str,
                        help="Cell type to predict")
    parser.add_argument("--threads", "-t", required=True,
                        type=int,
                        help="Number of threads")
    parser.add_argument("--correlationDepth", "-cd", required=True,
                        type=int,
                        help="Correlation depth")
    parser.add_argument("--originalDataMatrix", "-odm", required=True,
                        type=str,
                        help="Original data matrix")
    parser.add_argument("--outputFolder", "-of", required=True,
                        type=str,
                        help="Output folder")
    parser.add_argument("--polynomialModelPath", "-pmp", required=True,
                        type=str,
                        help="Polynomial model path")
    return parser.parse_args()

def scoring(pArgs, pChromosome):
    correlationMethodList = ['pearson_spearman', 'hicrep', 'TAD_score_MSE', "TAD_fraction"]
    errorType = ['R2', 'MSE', 'MAE', 'MSLE', 'AUC']
    score_dict = {}
    trial_id = "trial"  # Assuming trial_id is a constant or can be passed as an argument
    os.makedirs(os.path.join(pArgs.outputFolder, "tads_predicted"), exist_ok=True)
    # chromosomes = ' '.join(pArgs.chromosomes)
    arguments_tad = "--matrix {} --minDepth {} --maxDepth {} --step {} --numberOfProcessors {}  \
                    --outPrefix {} --minBoundaryDistance {} \
                    --correctForMultipleTesting fdr --thresholdComparisons 0.5 --chromosomes {}".format(pArgs.matrix2, pArgs.binSize * 3, pArgs.binSize * 10, pArgs.binSize, pArgs.threads,
                    os.path.join(pArgs.outputFolder, "tads_predicted") + '/tads', pArgs.binSize, pChromosome).split()
    try:
        hicFindTADs.main(arguments_tad)
    except Exception as e:
        # traceback.print_exc()
        print("No TADs found for chromosome: ", pChromosome)
        return None
    
    for correlationMethod in correlationMethodList:
        if correlationMethod == 'pearson_spearman':
            log.info("Scoring using pearson_spearman")

            # for chrom in pArgs.chromosomes:
            score_dataframe = computePearsonCorrelation(pCoolerFile1=pArgs.matrix1, pCoolerFile2=pArgs.matrix2,
                                                        pWindowsize_bp=pArgs.correlationDepth, pModelChromList=[pChromosome], pTargetChromStr=pChromosome,
                                                        pModelCellLineList=[], pTargetCellLineStr="",
                                                        pPlotOutputFile=None, pCsvOutputFile=None)
            for correlationMethod_ in ['pearson', 'spearman']:
                for errorType_ in errorType:
                    if correlationMethod_ + '_' + errorType_ in score_dict:
                        score_dict[correlationMethod_ + '_' + errorType_][0] += score_dataframe.loc[correlationMethod_, errorType_]
                    else:
                        score_dict[correlationMethod_ + '_' + errorType_] = [score_dataframe.loc[correlationMethod_, errorType_]]

            for correlationMethod_ in ['pearson', 'spearman']:
                for errorType_ in errorType:
                    score_dict[correlationMethod_ + '_' + errorType_][0] = score_dict[correlationMethod_ + '_' + errorType_][0] / len([pChromosome])

        elif correlationMethod == 'hicrep':
            log.info("Scoring using hicrep")
            if pArgs.matrix1.split(':')[0].endswith('.mcool'):
                cool1, binSize1 = readMcool(pArgs.matrix1.split(':')[0], pArgs.binSize)
            else:
                cool1, binSize1 = readMcool(pArgs.matrix1, -1)
            if pArgs.matrix2.split(':')[0].endswith('.mcool'):
                cool2, binSize2 = readMcool(pArgs.matrix2.split(':')[0], pArgs.binSize)
            else:
                cool2, binSize2 = readMcool(pArgs.matrix2, -1)
            # cool1, binSize1 = readMcool(pArgs.matrix1, -1)
            # cool2, binSize2 = readMcool(pArgs.matrix2, -1)
            h = 5
            dBPMax = 1000000
            bDownSample = False
            sccSub = hicrepSCC(cool1, cool2, h, dBPMax, bDownSample, [pChromosome])
            score_dict[correlationMethod] = [np.mean(sccSub)]

        elif correlationMethod == 'TAD_score_MSE' or correlationMethod == 'TAD_fraction':
            log.info("Scoring using TADs")
            

            if correlationMethod == 'TAD_score_MSE':
                tad_score_predicted = os.path.join(pArgs.outputFolder, "tads_predicted") + '/tads_score.bedgraph'
                tad_score_orgininal = os.path.join(pArgs.outputFolder, "tads_original") + '/tads_score.bedgraph'
                tad_score_predicted_df = pd.read_csv(tad_score_predicted, names=['chromosome', 'start', 'end', 'score'], sep='\t')
                tad_score_orgininal_df = pd.read_csv(tad_score_orgininal, names=['chromosome', 'start', 'end', 'score'], sep='\t')
                mean_sum_of_squares = ((tad_score_predicted_df['score'] - tad_score_orgininal_df['score']) ** 2).mean()
                score_dict[correlationMethod] = [mean_sum_of_squares]
            elif correlationMethod == 'TAD_fraction':
                tad_boundaries_predicted = os.path.join(pArgs.outputFolder, "tads_predicted") + '/tads_boundaries.bed'
                tad_boundaries_orgininal = os.path.join(pArgs.outputFolder, "tads_original") + '/tads_boundaries.bed'
                tad_boundaries_predicted_df = pd.read_csv(tad_boundaries_predicted, names=['chromosome', 'start', 'end', 'name', 'score', '.'], sep='\t')
                tad_boundaries_orgininal_df = pd.read_csv(tad_boundaries_orgininal, names=['chromosome', 'start', 'end', 'name', 'score', '.'], sep='\t')
                tad_fraction = len(tad_boundaries_predicted_df) / len(tad_boundaries_orgininal_df)
                exact_matches = pd.merge(tad_boundaries_predicted_df[['chromosome', 'start', 'end']],
                                        tad_boundaries_orgininal_df[['chromosome', 'start', 'end']],
                                        on=['chromosome', 'start', 'end'], how='inner')
                tad_fraction_exact_match = len(exact_matches) / len(tad_boundaries_orgininal_df)
                score_dict[correlationMethod] = [tad_fraction]
                score_dict[correlationMethod + '_exact_match'] = [tad_fraction_exact_match]

    model_files = None
    if os.path.exists(pArgs.polynomialModelPath):
        if os.path.isdir(pArgs.polynomialModelPath):
            model_files = [os.path.join(pArgs.polynomialModelPath, f) for f in os.listdir(pArgs.polynomialModelPath) if f.endswith('.pkl') and not f.startswith('.')]
        else:
            model_files = [pArgs.polynomialModelPath]
    else:
        print(f"Model file folder not found: {pArgs.polynomialModelPath}")
    
    for model_path in model_files:
        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
        if model is not None:
            scores_df = pd.DataFrame(score_dict)
            features = []
            names_model = os.path.basename(model_path).split('.')[1].split('-')
            for name in names_model:
                features.append(name)
            score = model.predict(scores_df[features])[0]
            score_dict[os.path.basename(model_path)] = [score]

    model_name = "polynomial_models"
    if not os.path.isdir(pArgs.polynomialModelPath):
        model_name = os.path.basename(pArgs.polynomialModelPath).split('.')[0]

    if pArgs.matrix1.split(':')[0].endswith('.mcool'):
        matrix1_name = os.path.basename(pArgs.matrix1.split(':')[0]) + ':' + pArgs.matrix1.split('::')[1].split('/')[-1]
        matrix2_name = os.path.basename(pArgs.matrix2.split(':')[0]) + ':' + pArgs.matrix2.split('::')[1].split('/')[-1]
    else:
        matrix1_name = os.path.basename(pArgs.matrix1).split('.')[0]
        matrix2_name = os.path.basename(pArgs.matrix2).split('.')[0]
    output_file = os.path.join(pArgs.outputFolder, f"scores_{model_name}_{matrix1_name}_{matrix2_name}_{pChromosome}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Matrix1: {pArgs.matrix1}\n")
        f.write(f"Matrix2: {pArgs.matrix2}\n")
        for score_name, score_value in score_dict.items():
            f.write(f"{score_name}: {score_value[0]}\n")
        # f.write(f"{model_name} Score: {score}\n")
    print(f"Scores written to {output_file}")
    return score_dict

def main(args=None):
    pArgs = parse_arguments()
    tad_original_folder = os.path.join(pArgs.outputFolder, "tads_original")
    tad_predicted_folder = os.path.join(pArgs.outputFolder, "tads_predicted")
    if pArgs.chromosomes is None:
        c = cooler.Cooler(pArgs.matrix1)
        pArgs.chromosomes = c.chromnames

    score_dicts = []
    for chromosome in pArgs.chromosomes:
        try:
            if os.path.exists(tad_original_folder):
                shutil.rmtree(tad_original_folder)
                log.info(f"Deleted folder: {tad_original_folder}")

            if os.path.exists(tad_predicted_folder):
                shutil.rmtree(tad_predicted_folder)
                log.info(f"Deleted folder: {tad_predicted_folder}")
            os.makedirs(os.path.join(pArgs.outputFolder, "tads_original"), exist_ok=True)
            if not os.path.exists(pArgs.polynomialModelPath):
                raise FileNotFoundError(f"Polynomial model file not found: {pArgs.polynomialModelPath}")
            # chromosomes = ' '.join(p)
            arguments_tad = "--matrix {} --minDepth {} --maxDepth {} --step {} --numberOfProcessors {}  \
                                --outPrefix {} --minBoundaryDistance {} \
                                --correctForMultipleTesting fdr --thresholdComparisons 0.5 --chromosomes {}".format(pArgs.originalDataMatrix, pArgs.binSize * 3, pArgs.binSize * 10, pArgs.binSize, pArgs.threads,
                            os.path.join(pArgs.outputFolder, "tads_original") + '/tads', pArgs.binSize, chromosome).split()
            print(arguments_tad)
            try:
                hicFindTADs.main(arguments_tad)
            except Exception as e:
                # traceback.print_exc()
                print("No TADs found for chromosome: ", chromosome)
                continue
            tad_score_orgininal = os.path.join(pArgs.outputFolder, "tads_original") + '/tads_score.bedgraph'
            tad_score_orgininal_df = pd.read_csv(tad_score_orgininal, names=['chromosome', 'start', 'end', 'score'], sep='\t')
            log.info(f"Original TADs score: {tad_score_orgininal_df}")
            score_dicts.append(scoring(pArgs,chromosome))

            # Delete the created folders for TAD original and predicted
            # if os.path.exists(tad_original_folder):
            #     shutil.rmtree(tad_original_folder)
            #     log.info(f"Deleted folder: {tad_original_folder}")

            # if os.path.exists(tad_predicted_folder):
            #     shutil.rmtree(tad_predicted_folder)
            #     log.info(f"Deleted folder: {tad_predicted_folder}")
        except Exception as e:
            traceback.print_exc()
            print(e)
            continue
    # Sum all score_dicts together and compute the average per field
    final_score_dict = {}
    for score_dict in score_dicts:
        for key, value in score_dict.items():
            if key in final_score_dict:
                final_score_dict[key] += value[0]
            else:
                final_score_dict[key] = value[0]

    for key in final_score_dict:
        final_score_dict[key] /= len(score_dicts)

    model_name = "polynomial_models"
    if not os.path.isdir(pArgs.polynomialModelPath):
        model_name = os.path.basename(pArgs.polynomialModelPath).split('.')[0]

    if pArgs.matrix1.split(':')[0].endswith('.mcool'):
        matrix1_name = os.path.basename(pArgs.matrix1.split(':')[0]) + ':' + pArgs.matrix1.split('::')[1].split('/')[-1]
        matrix2_name = os.path.basename(pArgs.matrix2.split(':')[0]) + ':' + pArgs.matrix2.split('::')[1].split('/')[-1]
    else:
        matrix1_name = os.path.basename(pArgs.matrix1).split('.')[0]
        matrix2_name = os.path.basename(pArgs.matrix2).split('.')[0]
    # Write the final scores into a file
    final_output_file = os.path.join(pArgs.outputFolder, f"final_scores_{model_name}_{matrix1_name}_{matrix2_name}_all.txt")
    with open(final_output_file, 'w') as f:
        f.write(f"Matrix1: {pArgs.matrix1}\n")
        f.write(f"Matrix2: {pArgs.matrix2}\n")
        for score_name, score_value in final_score_dict.items():
            f.write(f"{score_name}: {score_value}\n")
    print(f"Final scores written to {final_output_file}")