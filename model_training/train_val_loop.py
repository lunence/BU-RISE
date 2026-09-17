import os
import json
import time
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from model_training.model.model import GridNet
from model_training.model.loss_function import (
    CenterLossFunction,
    OrientationLossFunction
)
import torch.nn as nn
from model_training.util.config import (
    ORIENTATION_LOSS_WEIGHT,
    CENTER_LOSS_WEIGHT,
    CE_LOSS_WEIGHT,
    CENTER_CORRECT_RANGE
)
import numpy as np
from model_training.model.training import train_one_epoch
from model_training.model.eval import eval
from model_training.model.val_accuracy import calculate_val_accuracy


device = torch.device("cpu")


# ==================================================
# Helper function for JSON serialization
# ==================================================

def make_json_serializable(obj):

    if isinstance(obj, torch.Tensor):

        if obj.numel() == 1:
            return obj.item()

        return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):

        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):

        return [
            make_json_serializable(value)
            for value in obj
        ]

    return obj


# ==================================================
# Training function
# ==================================================

def train(
    loaded_model,
    data_loader,
    data_loader_test,
    num_epochs,
    lr=1e-3,
    finetuning=False,
    checkpoint=False,
    model_suffix="",
    batch_size=32,
    experiment_name="experiment"
):

    print("training model with real world data")


    # ==================================================
    # Determine model name
    # ==================================================

    if finetuning:

        save_file_name = "finetuning_model" + model_suffix

    else:

        save_file_name = "initial_training_model" + model_suffix


    # ==================================================
    # Create unique experiment folder
    # ==================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    experiment_folder_name = (
        f"{experiment_name}_{timestamp}"
    )

    results_dir = os.path.join(
        "model_training",
        "results",
        experiment_folder_name
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    print("Saving training results to:")
    print(results_dir)


    # ==================================================
    # Create model
    # ==================================================

    model = GridNet().to(device)


    # ==================================================
    # Load pretrained model if finetuning
    # ==================================================

    if finetuning:

        state_dict = torch.load(
            loaded_model,
            map_location=device
        )

        model.load_state_dict(
            state_dict
        )

        model.to(device)


    # ==================================================
    # Loss functions
    # ==================================================

    center_criterion = CenterLossFunction().to(device)

    orientation_criterion = OrientationLossFunction().to(device)

    class_criterion = nn.CrossEntropyLoss()


    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    # ==================================================
    # Learning-rate scheduler
    # ==================================================

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )


    # ==================================================
    # Training variables
    # ==================================================

    start_epoch = 0

    best_val_loss = float("inf")


    # ==================================================
    # Loss/history storage
    # ==================================================

    train_losses = []
    val_losses = []

    ce_losses = []
    center_losses = []
    orientation_losses = []
    class_losses = []

    all_center_error = []
    all_orientation_error = []

    ce_train_losses = []
    center_train_losses = []
    orientation_train_losses = []
    class_train_losses = []


    # ==================================================
    # Timing storage
    # ==================================================

    epoch_times = []

    training_start_time = time.perf_counter()


    # ==================================================
    # Load checkpoint
    # ==================================================

    if checkpoint:

        checkpoint_path = os.path.join(
            "model_training",
            save_file_name + "_checkpoint.pth"
        )

        # ----------------------------------------------
        # Check whether checkpoint exists
        # ----------------------------------------------

        if os.path.exists(checkpoint_path):

            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=device
            )

            model.load_state_dict(
                checkpoint_data["model_state_dict"]
            )

            optimizer.load_state_dict(
                checkpoint_data["optimizer_state_dict"]
            )

            lr_scheduler.load_state_dict(
                checkpoint_data["scheduler_state_dict"]
            )

            best_val_loss = checkpoint_data.get(
                "best_val_loss",
                float("inf")
            )

            train_losses = checkpoint_data.get(
                "train_losses",
                []
            )

            val_losses = checkpoint_data.get(
                "val_losses",
                []
            )

            start_epoch = (
                checkpoint_data["epoch"] + 1
            )

            model.to(device)

            print(
                f"Loaded checkpoint: "
                f"{checkpoint_path}"
            )

        else:

            print(
                f"No checkpoint found at "
                f"{checkpoint_path}"
            )

            print(
                "Starting training from scratch."
            )


    # ==================================================
    # Training loop
    # ==================================================

    for epoch in range(
        start_epoch,
        num_epochs
    ):

        # ----------------------------------------------
        # Start epoch timer
        # ----------------------------------------------

        epoch_start_time = time.perf_counter()


        # ==================================================
        # Train
        # ==================================================

        (
            train_loss,
            train_accuracy,
            ce_loss,
            center_loss,
            orientation_loss,
            class_loss
        ) = train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            class_criterion,
            center_criterion,
            orientation_criterion
        )


        # ==================================================
        # Update learning rate
        # ==================================================

        lr_scheduler.step()


        # ==================================================
        # Store training losses
        # ==================================================

        train_losses.append(
            train_loss
        )

        ce_train_losses.append(
            ce_loss
        )

        center_train_losses.append(
            center_loss
        )

        orientation_train_losses.append(
            orientation_loss
        )

        class_train_losses.append(
            class_loss
        )


        # ==================================================
        # Evaluate model
        # ==================================================

        (
            accuracy,
            pose_accuracy,
            orientation_accuracy,
            class_accuracy,
            center_error,
            orientation_error
        ) = eval(
            model,
            data_loader_test,
            device
        )


        # ----------------------------------------------
        # Store errors
        # ----------------------------------------------

        all_center_error += center_error

        # Convert orientation bins to degrees
        all_orientation_error += (
            orientation_error * 5
        )


        # ==================================================
        # Calculate validation losses
        # ==================================================

        (
            val_loss,
            ce_loss,
            center_loss,
            orientation_loss,
            class_loss
        ) = calculate_val_accuracy(
            model,
            device,
            data_loader_test,
            class_criterion,
            center_criterion,
            orientation_criterion
        )


        val_losses.append(
            val_loss
        )

        ce_losses.append(
            ce_loss
        )

        center_losses.append(
            center_loss
        )

        orientation_losses.append(
            orientation_loss
        )

        class_losses.append(
            class_loss
        )


        # ==================================================
        # End epoch timer
        # ==================================================

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        epoch_times.append(
            epoch_time
        )


        # ==================================================
        # Print results
        # ==================================================

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: "
            f"train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.3f}\n"
            f"val loss {val_loss:.4f}, "
            f"val accuracy {accuracy:.3f}, "
            f"Pose accuracy: {pose_accuracy:.3f}, "
            f"Orientation accuracy: "
            f"{orientation_accuracy:.3f}, "
            f"Class accuracy: "
            f"{class_accuracy:.3f}"
        )

        print(
            f"CE Loss = "
            f"{ce_losses[-1]:.4f}, "
            f"Center Loss = "
            f"{center_losses[-1]:.4f}, "
            f"Orientation Loss = "
            f"{orientation_losses[-1]:.4f}, "
            f"Class Loss = "
            f"{class_losses[-1]:.4f}"
        )

        print(
            f"Epoch Time = "
            f"{epoch_time:.2f} seconds "
            f"({epoch_time / 60:.2f} minutes)"
        )


        # ==================================================
        # Save best model
        # ==================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss


            # ----------------------------------------------
            # Save global best model
            # ----------------------------------------------

            global_best_path = os.path.join(
                "model_training",
                "best_" + save_file_name + ".pth"
            )

            torch.save(
                model.state_dict(),
                global_best_path
            )


            # ----------------------------------------------
            # Save best model inside experiment folder
            # ----------------------------------------------

            experiment_best_path = os.path.join(
                results_dir,
                "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                experiment_best_path
            )

            print(
                f"Saved best model "
                f"(val loss = {val_loss:.4f})"
            )


        # ==================================================
        # Save checkpoint
        # ==================================================

        checkpoint_data = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                lr_scheduler.state_dict(),

            "best_val_loss":
                best_val_loss,

            "train_losses":
                train_losses,

            "val_losses":
                val_losses
        }


        # ----------------------------------------------
        # Save normal checkpoint
        # ----------------------------------------------

        checkpoint_path = os.path.join(
            "model_training",
            save_file_name + "_checkpoint.pth"
        )

        torch.save(
            checkpoint_data,
            checkpoint_path
        )


        # ----------------------------------------------
        # Save checkpoint inside experiment folder
        # ----------------------------------------------

        experiment_checkpoint_path = os.path.join(
            results_dir,
            "checkpoint.pth"
        )

        torch.save(
            checkpoint_data,
            experiment_checkpoint_path
        )


    # ==================================================
    # End total training timer
    # ==================================================

    total_training_time = (
        time.perf_counter()
        - training_start_time
    )

    average_epoch_time = (
        np.mean(epoch_times)
        if epoch_times
        else 0
    )


    # ==================================================
    # Measure inference time
    # ==================================================

    print(
        "\nMeasuring inference time..."
    )

    model.eval()

    inference_times = []

    with torch.no_grad():

        for images, targets in data_loader_test:

            images = images.to(device)

            start_time = time.perf_counter()

            logits = model(images)

            end_time = time.perf_counter()

            inference_times.append(
                end_time - start_time
            )


    # ==================================================
    # Calculate inference statistics
    # ==================================================

    if inference_times:

        total_inference_time = np.sum(
            inference_times
        )

        average_batch_inference_time = np.mean(
            inference_times
        )

        median_batch_inference_time = np.median(
            inference_times
        )

        p95_batch_inference_time = np.percentile(
            inference_times,
            95
        )


        # ----------------------------------------------
        # Average time per image
        # ----------------------------------------------

        total_images = (
            len(data_loader_test.dataset)
            if hasattr(
                data_loader_test,
                "dataset"
            )
            else len(inference_times) * batch_size
        )

        average_inference_time_per_image = (
            total_inference_time
            / total_images
        )

        inference_fps = (
            1
            / average_inference_time_per_image
        )

    else:

        total_inference_time = 0

        average_batch_inference_time = 0

        median_batch_inference_time = 0

        p95_batch_inference_time = 0

        average_inference_time_per_image = 0

        inference_fps = 0


    print(
        f"Average inference time per image: "
        f"{average_inference_time_per_image * 1000:.3f} ms"
    )

    print(
        f"Inference FPS: "
        f"{inference_fps:.2f}"
    )


    # ==================================================
    # Create epochs array
    # ==================================================

    epochs = range(
        1,
        len(train_losses) + 1
    )


    # ==================================================
    # Graph 1:
    # Training vs Validation Loss
    # ==================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )


    # ----------------------------------------------
    # Training vs validation
    # ----------------------------------------------

    axes[0].plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    axes[0].plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].set_title(
        "Training vs Validation Loss"
    )

    axes[0].legend()

    axes[0].grid(True)


    # ----------------------------------------------
    # Center error histogram
    # ----------------------------------------------

    axes[1].hist(
        all_center_error,
        bins=20,
        edgecolor="black"
    )

    axes[1].set_xlabel(
        "Center Error"
    )

    axes[1].set_ylabel(
        "Frequency Count"
    )

    axes[1].set_title(
        "Center Error Distribution"
    )


    # ----------------------------------------------
    # Orientation error histogram
    # ----------------------------------------------

    axes[2].hist(
        all_orientation_error,
        bins=72,
        edgecolor="black"
    )

    axes[2].set_xlabel(
        "Orientation Error (degrees)"
    )

    axes[2].set_ylabel(
        "Frequency Count"
    )

    axes[2].set_title(
        "Orientation Error Distribution"
    )


    # ==================================================
    # Add training parameters to graph
    # ==================================================

    axes[2].text(

        1.05,
        0.5,

        f"Experiment: {experiment_name}\n"
        f"Learning Rate: {lr}\n"
        f"Epochs: {num_epochs}\n"
        f"Batch Size: {batch_size}\n"
        f"Center Weight: {CENTER_LOSS_WEIGHT}\n"
        f"Orientation Weight: "
        f"{ORIENTATION_LOSS_WEIGHT}\n"
        f"CE Weight: {CE_LOSS_WEIGHT}\n"
        f"Center Correct Range: "
        f"{CENTER_CORRECT_RANGE}",

        transform=axes[2].transAxes,

        va="center",

        ha="left"
    )


    plt.tight_layout()


    # ==================================================
    # Save Graph 1
    # ==================================================

    graph1_path = os.path.join(
        results_dir,
        "training_vs_validation_and_errors.png"
    )

    plt.savefig(
        graph1_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # ==================================================
    # Graph 2:
    # Individual training loss components
    # ==================================================

    epochs = range(
        1,
        len(ce_train_losses) + 1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,
        ce_train_losses,
        label="Cross Entropy"
    )

    plt.plot(
        epochs,
        center_train_losses,
        label="Center Loss"
    )

    plt.plot(
        epochs,
        orientation_train_losses,
        label="Orientation Loss"
    )

    plt.plot(
        epochs,
        class_train_losses,
        label="Class Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training Loss Components"
    )

    plt.grid(True)

    plt.legend()


    # ==================================================
    # Save Graph 2
    # ==================================================

    graph2_path = os.path.join(
        results_dir,
        "training_loss_components.png"
    )

    plt.savefig(
        graph2_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # ==================================================
    # Save all numerical training data
    # ==================================================

    training_data = {

        "train_losses":
            train_losses,

        "val_losses":
            val_losses,

        "ce_losses":
            ce_losses,

        "center_losses":
            center_losses,

        "orientation_losses":
            orientation_losses,

        "class_losses":
            class_losses,

        "ce_train_losses":
            ce_train_losses,

        "center_train_losses":
            center_train_losses,

        "orientation_train_losses":
            orientation_train_losses,

        "class_train_losses":
            class_train_losses,

        "center_errors":
            all_center_error,

        "orientation_errors":
            all_orientation_error,

        "epoch_times_seconds":
            epoch_times,

        "total_training_time_seconds":
            total_training_time,

        "average_epoch_time_seconds":
            average_epoch_time,

        "inference_times_seconds":
            inference_times,

        "average_inference_time_per_image_seconds":
            average_inference_time_per_image,

        "inference_fps":
            inference_fps
    }


    training_data = make_json_serializable(
        training_data
    )


    training_data_path = os.path.join(
        results_dir,
        "training_data.json"
    )


    with open(
        training_data_path,
        "w"
    ) as file:

        json.dump(
            training_data,
            file,
            indent=4
        )


    # ==================================================
    # Save experiment parameters and final metrics
    # ==================================================

    metrics = {

        # ----------------------------------------------
        # Experiment identification
        # ----------------------------------------------

        "experiment_name":
            experiment_name,

        "experiment_folder":
            experiment_folder_name,

        "model":
            "GridNet",

        "dataset":
            model_suffix,


        # ----------------------------------------------
        # Training parameters
        # ----------------------------------------------

        "num_epochs":
            num_epochs,

        "learning_rate":
            lr,

        "batch_size":
            batch_size,

        "finetuning":
            finetuning,

        "checkpoint":
            checkpoint,


        # ----------------------------------------------
        # Loss parameters
        # ----------------------------------------------

        "center_loss_weight":
            CENTER_LOSS_WEIGHT,

        "orientation_loss_weight":
            ORIENTATION_LOSS_WEIGHT,

        "ce_loss_weight":
            CE_LOSS_WEIGHT,

        "center_correct_range":
            CENTER_CORRECT_RANGE,


        # ----------------------------------------------
        # Final model metrics
        # ----------------------------------------------

        "best_validation_loss":
            best_val_loss,

        "final_training_loss":
            (
                train_losses[-1]
                if train_losses
                else None
            ),

        "final_validation_loss":
            (
                val_losses[-1]
                if val_losses
                else None
            ),


        # ----------------------------------------------
        # Training time
        # ----------------------------------------------

        "total_training_time_seconds":
            total_training_time,

        "total_training_time_minutes":
            total_training_time / 60,

        "average_epoch_time_seconds":
            average_epoch_time,

        "average_epoch_time_minutes":
            average_epoch_time / 60,


        # ----------------------------------------------
        # Inference time
        # ----------------------------------------------

        "total_inference_time_seconds":
            total_inference_time,

        "average_batch_inference_time_seconds":
            average_batch_inference_time,

        "median_batch_inference_time_seconds":
            median_batch_inference_time,

        "p95_batch_inference_time_seconds":
            p95_batch_inference_time,

        "average_inference_time_per_image_seconds":
            average_inference_time_per_image,

        "average_inference_time_per_image_ms":
            average_inference_time_per_image * 1000,

        "inference_fps":
            inference_fps,


        # ----------------------------------------------
        # Timestamp
        # ----------------------------------------------

        "timestamp":
            timestamp
    }


    metrics = make_json_serializable(
        metrics
    )


    metrics_path = os.path.join(
        results_dir,
        "metrics.json"
    )


    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


    # ==================================================
    # Print final results
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "Training complete!"
    )

    print(
        "========================================"
    )

    print(
        f"Experiment: "
        f"{experiment_name}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Total training time: "
        f"{total_training_time:.2f} seconds "
        f"({total_training_time / 60:.2f} minutes)"
    )

    print(
        f"Average epoch time: "
        f"{average_epoch_time:.2f} seconds "
        f"({average_epoch_time / 60:.2f} minutes)"
    )

    print(
        f"Average inference time: "
        f"{average_inference_time_per_image * 1000:.3f} ms/image"
    )

    print(
        f"Inference FPS: "
        f"{inference_fps:.2f}"
    )

    print(
        f"Results saved to:\n"
        f"{results_dir}"
    )

    print(
        "========================================\n"
    )