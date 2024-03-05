from src.main.classes.dataHandlers import CelebFacesDataHandler

if __name__ == "__main__": 
    DATA_SAVE_PATH = "data/CelebFaces/"
    RESULTS_PATH_BASE = "models/CelebFaces/WGAN_GP/"    
    SAVED_MODEL_PATH = RESULTS_PATH_BASE + "saved_models/"
    ARCHITECTURE_DIAGRAM_PATH = RESULTS_PATH_BASE + "visualizations/architecture/"
    METRICS_SAVE_PATH = RESULTS_PATH_BASE + "visualizations/results/"


    image_size = (128, 128)
    h,w = image_size
    data_hanlder = CelebFacesDataHandler(
        image_size=image_size,
        collection_size=10_000
        ) \
        .collect_data()\
        .save_data(path=DATA_SAVE_PATH)