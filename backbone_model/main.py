from train_val_loop import train_real_world
import video_data_objects as vid
import real_world_objects as synthetic
import simple_model_objects_modified as simple
from visualize_real_world import visualize
from datasets.synthetic_dataset.generate_synthetic_data import generate_synthetic_dataset

def main():
    # train_real_world(vid.data_loader, vid.data_loader_test, 40, lr = 1e-3, finetuning = True, checkpoint = False, str = "_vid")
    # train_real_world(synthetic.data_loader, synthetic.data_loader_test, 40, lr = 1e-3, finetuning = True, checkpoint = False, str = "_synthetic")

    # "backbone_model/best_model_5000imgs.pth", # for synthetic
    # "backbone_model/best_finetuning_model_lr1e-3.pth", # for real world
    # "backbone_model/best_initial_training_model.pth"

    # train_real_world("", simple.data_loader, simple.data_loader_test, 40, lr = 1e-3, finetuning = False, checkpoint = False, str = "_new_synthetic")
    # train_real_world("backbone_model/best_initial_training_model_new_synthetic.pth", synthetic.data_loader, synthetic.data_loader_test, 40, lr = 1e-2, finetuning = True, checkpoint = False, str = "_new_synthetic1")
    # train_real_world("backbone_model/best_finetuning_model_new_synthetic1.pth", synthetic.data_loader, synthetic.data_loader_test, 40, lr = 1e-2, finetuning = True, checkpoint = False, str = "_new_synthetic2")

    visualize("/home/mooncyli/BU-RISE/backbone_model/best_finetuning_model_new_synthetic2.pth")
    # generate_synthetic_dataset(5000)

if __name__ == "__main__":
    main()