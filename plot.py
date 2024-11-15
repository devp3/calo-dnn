import logging, pickle

from utils.utils import get_args
from utils.config import process_config
from utils.handler import Handler
from data_loader.example_data_loader import DatasetGenerator
from plotters.kinematics import Kinematics
from plotters.history import History
from plotters.plot_all import PlotAll
from plotters.performance import Performance
from plotters.compare import Compare


def plot():
    try:
        args = get_args()
        config = process_config(args.config)

        logging.basicConfig(
            # filename='logs.log',
            level=args.loglevel.upper(),
            format='%(asctime)s:%(levelname)s:%(message)s',
            encoding='utf-8',
            handlers=[
                logging.FileHandler('logs.log'),
                logging.StreamHandler(),
            ], 
            force=True, # this is important to overwrite the previously set default logging level
        )

        # matplotlib logger is being too verbose, need to turn it off
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        # numeric_level = getattr(logging, args.loglevel.upper(), None)

        # logger = logging.getLogger('no_spam')
        # logger.setLevel(numeric_level)

        logging.info('Logging level set to: {}'.format(args.loglevel.upper()))
        logging.debug('args: {}'.format(args))
        logging.debug('args.config: {}'.format(args.config))
        logging.debug('config: {}'.format(config))

    except: 
        logging.critical("Missing or invalid arguments")
        exit(0)
    
    # df, _ = DatasetGenerator(config, mode='pd', normed=False)()

    handler = Handler(config)
    dataloader = DatasetGenerator(config)

    raw_data = handler.load('raw_data')
    normalized_data = handler.load('normalized_data')
    model = handler.load('model')
    train_history = handler.load('history')
    test_data = handler.load('test_data') # loads in pandas dataframe
    train_data = handler.load('train_data')
    normalization_factors = handler.load('normalization_factors')

    logging.debug(f'type(model): {type(model)}')
    logging.debug(f'Normalization factors: {normalization_factors}')
    logging.debug(f'test_data[0].describe(): {test_data[0].describe()}')
    logging.debug(f'test_data[1].describe(): {test_data[1].describe()}')

    kinematics = Kinematics(config)
    history = History(config)
    plot_all = PlotAll(config)
    performance = Performance(config, normalization_factors)
    compare = Compare(config)

    # kinematics.plot(raw_data, title='Normalized', yscale='log')
    history.plot(train_history, title='Training History')
    plot_all.plot(raw_data, yscale='log')
    plot_all.plot(normalized_data, pretitle='Normalized', yscale='log')
    performance.plot(
        model, 
        test_data, 
        # avoid_plotting=['target_vs_predicted', 'total_residuals', 'binned_residuals', 'feature_importance', 'feature_vs_residuals'],
        comparison_ylim=(0,100)
    )

    with open(config.base_path + '/raw_data_Zee.pkl', 'rb') as f:
        raw_data_Zee = pickle.load(f)

    with open(config.base_path + '/raw_data_signal.pkl', 'rb') as f:
        raw_data_signal = pickle.load(f)

    raw_data_Zee = raw_data_Zee.drop(columns=[
        'el1_pt',
        'el2_pt',
        'el1_E',
        'el2_E',
        # 'el1_eta',
        # 'el2_eta',
        'el1_phi',
        'el2_phi',
        'ph1_pt',
        'ph2_pt',
        'ph1_E',
        'ph2_E',
        # 'ph1_eta',
        # 'ph2_eta',
        'ph1_phi',
        'ph2_phi',
        'PassTrig_g35_loose_g25_loose',
        'PassTrig_g35_medium_g25_medium',
        'TV_x',
        'TV_y',
    ], 
    errors='ignore',
    )
    raw_data_signal = raw_data_signal.drop(columns=[
        'el1_pt',
        'el2_pt',
        'el1_E',
        'el2_E',
        # 'el1_eta',
        # 'el2_eta',
        'el1_phi',
        'el2_phi',
        'ph1_pt',
        'ph2_pt',
        'ph1_E',
        'ph2_E',
        # 'ph1_eta',
        # 'ph2_eta',
        'ph1_phi',
        'ph2_phi',
        'PassTrig_g35_loose_g25_loose',
        'PassTrig_g35_medium_g25_medium',
        'TV_x',
        'TV_y',
    ], 
    errors='ignore',
    )



    # compare.plot(raw_data_Zee, raw_data_signal, names=['Zee', 'Signal'], yscale='log')

    

    

if __name__ == '__main__':
    # only if ran as a script
    plot()


