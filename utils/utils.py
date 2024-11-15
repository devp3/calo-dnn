import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='Path to the config file',
    )

    argparser.add_argument(
        '-log', '--loglevel', '--log-level', '--log_level', 
        '--log', '--l', '--L', # is this too many options?
        metavar='L', 
        default='INFO',
        help='The log level to use. Default is INFO. Options are: DEBUG, INFO, WARNING, ERROR, CRITICAL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )

    argparser.add_argument(
        '-n', '--name',
        metavar='N',
        default='None',
        help='Name of the training or plotting run. Default is None.',
    )

    argparser.add_argument(
        '-a', '--archive',
        metavar='A',
        default='False',
        help='Archive the run. Default is False.',
    )

    args = argparser.parse_args()
    return args