from src.main.classes.dataHandlers import AnimeFacesDataHandler
from src.main.classes.modelArchitectures.GAN import WGAN_GP
import os

if __name__ == "__main__": 
    BATCH_SIZE = 32
    EPOCHS = 1 # Number of epochs to run before advancing pipeline

    DATA_SAVE_PATH = "data/animeFaces/"
    RESULTS_PATH_BASE = "models/AnimeFaces/WGAN_GP/"    
    SAVED_MODEL_PATH = RESULTS_PATH_BASE + "saved_models/"
    ARCHITECTURE_DIAGRAM_PATH = RESULTS_PATH_BASE + "visualizations/architecture/"
    METRICS_SAVE_PATH = RESULTS_PATH_BASE + "visualizations/results/"

    image_size = (64, 64)
    h,w = image_size

    data_hanlder = AnimeFacesDataHandler(
        image_size=image_size,
        collection_size=30_000)\
            .read_local_data(file_path=DATA_SAVE_PATH)\
            .process_data()
    (x_train, y_train), (x_test, y_test) = data_hanlder.get_processed_data()

    gan = WGAN_GP(
            image_size = (h,w,3),

            latent_space_dim=200,

            generator_filter_sizes=[128,64,64,3],
            generator_kernel_sizes=[5,5,5,5],
            generator_upsample_sizes = [2,2,1,1],

            critic_filter_sizes=[64,64,128,128],
            critic_kernel_sizes=[5,5,5,5],
            critic_strides=[2,2,2,1],
            critic_use_drop_out=False,
            generator_use_batch_norm=False,
            generator_use_drop_out=False,
            critic_activation_functions=["leaky_relu"]
        )
    
    if os.path.exists(SAVED_MODEL_PATH):
        gan = gan.load_weights(SAVED_MODEL_PATH)

    gan.save_architecture_diagram(ARCHITECTURE_DIAGRAM_PATH,"WGAN_GP_ARCHITECTURE")
    gan.save_model(SAVED_MODEL_PATH)

    gan.train(x_train=x_train, 
              batch_size=BATCH_SIZE, 
              epochs=EPOCHS, 
              save_folder=METRICS_SAVE_PATH,
              x_val=x_test)
    gan.save_model(SAVED_MODEL_PATH)