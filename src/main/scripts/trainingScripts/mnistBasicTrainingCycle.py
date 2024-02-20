from src.main.classes.dataHandlers import MnistDataHandler
from src.main.classes.modelArchitectures.GAN import WGAN_GP

if __name__ == "__main__": 

    RESULTS_PATH_BASE = "models/MNIST/WGAN_GP/"    
    SAVED_MODEL_PATH = RESULTS_PATH_BASE + "saved_models/"
    ARCHITECTURE_DIAGRAM_PATH = RESULTS_PATH_BASE + "visualizations/architecture/"
    METRICS_SAVE_PATH = RESULTS_PATH_BASE + "visualizations/results/"

    data_hanlder = MnistDataHandler().collect_data().process_data()
    (x_train, y_train), (x_test, y_test) = data_hanlder.get_processed_data()

    gan = WGAN_GP(
        image_size = (28,28,1),

        latent_space_dim=10,

        generator_filter_sizes=[128,64,64,1],
        generator_kernel_sizes=[5,5,5,5],
        generator_upsample_sizes = [2,2,1,1],

        critic_filter_sizes=[64,64,128,128],
        critic_kernel_sizes=[5,5,5,5],
        critic_strides=[2,2,2,1],
    )

    BATCH_SIZE = 32
    EPOCHS = 8
    
    gan.train(x_train=x_train, batch_size=BATCH_SIZE, epochs=EPOCHS, save_folder=METRICS_SAVE_PATH)

    gan.save_architecture_diagram(ARCHITECTURE_DIAGRAM_PATH,"WGAN_GP_ARCHITECTURE")
    gan.save_model(SAVED_MODEL_PATH)
    gan.load_model(SAVED_MODEL_PATH)