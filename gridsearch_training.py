#!/usr/bin/env python3
import argparse
import subprocess
import yaml
import glob
import os

def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  
    parser.add_argument('-y',
                        '--gridsearch_yaml',
                        help='yaml that defines gridsearch',
                        type=str,
                        required=True)
    return parser.parse_args()



def main():
    args = get_args()
    with open(args.gridsearch_yaml, 'r') as file:
        grid_yaml = yaml.safe_load(file)

    print(grid_yaml)

    training_jsons = glob.glob(os.path.join(grid_yaml['training_json_directory'], '*'))

    for i in training_jsons:
        for y in grid_yaml['epochs']:
            for z in grid_yaml['learning_rates']:
                subprocess.run(['python3', 'modelTraining.py', '-j', i, '-e', str(y), '-lr', str(z), '-b', str(2)])
    # subprocess.run(['ls'])


# --------------------------------------------------
if __name__ == '__main__':
    main()    
