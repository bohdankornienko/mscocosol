import argparse
import yaml

from mscocosol.dsapproach.experiment import Experiment
from mscocosol.approach.approach_factory import approach_factory
from mscocosol.utils.datagen_factory import datagen_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The subsystem for running Data Science Experiments.')

    parser.add_argument('--sets', help="The path to settings file for the experiment.")

    args = parser.parse_args()

    with open(args.sets, 'r') as fp:
        sets = yaml.safe_load(fp)

    exp = Experiment(approach_factory, datagen_factory, **sets)

    exp.perform()
