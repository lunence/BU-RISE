from model_training.train_val_loop import train

from model_training.objects import get_data_loaders

# from model_training.util.visualize_real_world import visualize
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

    train(
        loaded_model="",

        data_loader=data_loader,
        data_loader_test=data_loader_test,

        num_epochs=40,

        lr=1e-2,

        finetuning=False,

        checkpoint=False,

        model_suffix="_new_synthetic",

        batch_size=32
    )


    # ==================================================
    # Other experiments
    # ==================================================

    # Example: train on simple synthetic dataset
    #
    # train(
    #     loaded_model="",
    #     data_loader=simple.data_loader,
    #     data_loader_test=simple.data_loader_test,
    #     num_epochs=40,
    #     lr=1e-3,
    #     finetuning=False,
    #     checkpoint=False,
    #     model_suffix="_new_synthetic",
    #     batch_size=32
    # )


    # Example: finetuning
    #
    # train(
    #     loaded_model="model_training/best_initial_training_model.pth",
    #     data_loader=synthetic.data_loader,
    #     data_loader_test=synthetic.data_loader_test,
    #     num_epochs=40,
    #     lr=1e-2,
    #     finetuning=True,
    #     checkpoint=False,
    #     model_suffix="_new_synthetic2",
    #     batch_size=32
    # )


    # Example: visualize trained model
    #
    # visualize(
    #     "model_training/best_finetuning_model_new_synthetic2.pth"
    # )


    # Example: generate synthetic dataset
    #
    # generate_synthetic_dataset(5000)


if __name__ == "__main__":
    main()