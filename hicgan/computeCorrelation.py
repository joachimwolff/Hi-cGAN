import argparse
from utils import computePearsonCorrelation

from hicgan._version import __version__

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='Compute Correlation')
    parser.add_argument('--coolfile1', type=str, help='Path to cool file 1')
    parser.add_argument('--coolfile2', type=str, help='Path to cool file 2')
    parser.add_argument('--windowSize', type=int, help='Window size')
    parser.add_argument('--outputname-csv', type=str, help='Output name csv')
    parser.add_argument('--outputname-plot', type=str, help='Output name plot')

    parser.add_argument('--model-chromosomes', nargs='+', type=str, help='List of chromosomes for the model')
    parser.add_argument('--target-chromosome', type=str, help='List of chromosomes for the target')
    parser.add_argument('--model-cell-lines',  type=str, help='List of cell lines for the model')
    parser.add_argument('--target-cell-lines',  type=str, help='List of cell lines for the target')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    return parser.parse_args()

def main():
   
    args = parse_arguments()

    # Access the arguments
    coolfile1 = args.coolfile1
    coolfile2 = args.coolfile2
    windowSize = args.windowSize
    outputname_csv = args.outputname_csv
    outputname_plot = args.outputname_plot

    model_chromosomes = args.model_chromosomes
    target_chromosome = args.target_chromosome
    model_cell_lines = args.model_cell_lines
    target_cell_lines = args.target_cell_lines

    computePearsonCorrelation(coolfile1, coolfile2, windowSize, 
                                model_chromosomes, target_chromosome, 
                                model_cell_lines, target_cell_lines,
                                outputname_plot, outputname_csv)
  