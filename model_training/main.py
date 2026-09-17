from model_training.train_val_loop import train

from model_training.objects import get_data_loaders

from model_training.util.visualize_real_world import visualize
# from datasets.synthetic_dataset.generate_synthetic_data import generate_synthetic_dataset


def main():

    # ==================================================
    # Choose dataset
    # ==================================================

    DATASET = "model_training/datasets/video_dataset" 
    data_loader, data_loader_test = get_data_loaders(DATASET)


    # ==================================================
    # Train model
    # ==================================================

    # train(
    #     loaded_model="",

    #     data_loader=data_loader,
    #     data_loader_test=data_loader_test,

    #     num_epochs=130,

    #     lr=1e-2,

    #     finetuning=False,

    #     checkpoint=True,

    #     model_suffix="_new_synthetic",

    #     batch_size=32
    # )


    # visualize trained model
    
    visualize(
        "model_training/results/initial_training_model_new_synthetic_lr0.01_epochs100_bs32_cw10_ow2.0_cew0.5_20260916_012223/best_model.pth",
        data_loader_test=data_loader_test
    )


    # Example: generate synthetic dataset
    #
    # generate_synthetic_dataset(5000)


if __name__ == "__main__":
    main()