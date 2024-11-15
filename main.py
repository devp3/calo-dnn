import logging
import pickle, sys

from data_loader.example_data_loader import DatasetGenerator
from models.base_model import BaseModel
from models.parallel_model_2d import ParallelModel2D
from trainers.example_train import ExampleTrainer
from utils.utils import get_args
from utils.config import process_config
from utils.handler import Handler
from plotters.kinematics import Kinematics
from plotters.history import History
from plotters.plot_all import PlotAll
from plotters.performance import Performance



def main():
    try:
        args = get_args()
        config = process_config(args.config)

        numeric_level = getattr(logging, args.loglevel.upper(), None)

        logging.basicConfig(
            # filename='logs.log',
            level=numeric_level,
            format='%(asctime)s:%(levelname)s:%(message)s',
            encoding='utf-8',
            handlers=[
                logging.FileHandler('logs.log'),
                logging.StreamHandler(),
            ], 
            force = True, # this is important to overwrite the previously set default logging level
        )

        logging.info(f'Logging level set to: {args.loglevel.upper()}')

        logging.debug(f'args: {args}')
        logging.debug(f'args.config: {args.config}')
        logging.debug(f'config: {config}')

        # config = process_config('configs/config.yml')


    except:
        print("missing or invalid arguments")
        exit(0)

    dataloader = DatasetGenerator(config)

    train_data, test_data, val_data = dataloader.load()
    handler = Handler(config)

    # logging.info('DATA SIZES:')
    # logging.info(f"train_data size: {sys.getsizeof(train_data)}")
    # logging.info(f"test_data size: {sys.getsizeof(test_data)}")
    # logging.info(f"val_data size: {sys.getsizeof(val_data)}")

    # handler = Handler(config)
    # handler.save(train_data, 'train_data')
    # handler.save(test_data, 'test_data')

    

    
    # test_string = 'This is a test string.'

    # with open('val_data.pkl', 'wb') as f:
    #     pickle.dump(test_string, f)

    # Create an instance of the model you want
    model = BaseModel(config)
    # model = ParallelModel2D(config)

    trainer = ExampleTrainer(model, train_data, val_data, config)

    trainer.train()

    raw_data = handler.load('raw_data')
    normalized_data = handler.load('normalized_data')
    model = handler.load('model')
    train_history = handler.load('history')
    test_data = handler.load('test_data') # loads in pandas dataframe
    normalization_factors = handler.load('normalization_factors')

    logging.debug(f'type(model): {type(model)}')
    logging.debug(f'Normalization factors: {normalization_factors}')
    logging.debug(f'test_data[0].describe(): {test_data[0].describe()}')
    logging.debug(f'test_data[1].describe(): {test_data[1].describe()}')

    kinematics = Kinematics(config)
    history = History(config)
    plot_all = PlotAll(config)
    performance = Performance(config, normalization_factors)

    kinematics.plot(raw_data, yscale='linear')
    plot_all.plot(raw_data, yscale='linear')
    plot_all.plot(normalized_data, pretitle='Normalized', yscale='log')
    history.plot(train_history, title='Training History')
    performance.plot(model, test_data)

if __name__ == '__main__':
    # only if ran as a script
    main()