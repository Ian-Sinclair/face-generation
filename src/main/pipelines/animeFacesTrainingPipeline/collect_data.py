from src.main.classes.dataHandlers import AnimeFacesDataHandler
from src.main.classes.modelArchitectures.GAN import WGAN_GP
import os

if __name__ == "__main__": 
    DATA_SAVE_PATH = "data/animeFaces/"
    RESULTS_PATH_BASE = "models/AnimeFaces/WGAN_GP/"    
    SAVED_MODEL_PATH = RESULTS_PATH_BASE + "saved_models/"
    ARCHITECTURE_DIAGRAM_PATH = RESULTS_PATH_BASE + "visualizations/architecture/"
    METRICS_SAVE_PATH = RESULTS_PATH_BASE + "visualizations/results/"


    image_size = (64, 64)
    h,w = image_size
    data_hanlder = AnimeFacesDataHandler(
        image_size=image_size,
        collection_size=30_000
        ) \
        .collect_data()\
        .save_data(path=DATA_SAVE_PATH)