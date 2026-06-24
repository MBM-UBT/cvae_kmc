import os

from cvae.training import training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. Standard arguments for the isothermal model, adjust for the temperature dependent model if needed.

    default_train_dir = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "training_datasets",
        "single_256_5000_sims_200_steps_dt5e-06",
        "train",
    )
    default_val_data_path = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "validation_datasets",
        "256_50_samples_T300_200_steps_seed0_dt5e-06",
    )

    # Arguments for the isothermal model:
    # The hyperparameters were determined by a study
    # comments behind the values are for the temeprature dependent model
    my_args_dict_isothermal = {
        "model_name": "isothermal_model_new",  # temperature_model new
        "train_data_dir": default_train_dir,  # adjust for temperature dataset
        "val_data_path": default_val_data_path,  # adjust for temperature validation data
        "num_workers": 4,
        "persistent_workers": True,
        "batch_size": 128,
        "epochs": 100,
        "anneal_epochs": 20,
        "frac_sims": 1.0,
        "max_steps": 75,
        "base_channels": 64,
        "warmup_epochs": 5,
        "dt_step": 1,
        "lr": 0.0004,  # 0.002
        "max_negative_change": 7,
        "max_positive_change": 13,  # 16
        "optimizer": "adamw",  # adam
        "latent_channels": 8,  # 4
        "beta_kl_final": 1.84,  # 1.6
        "free_bits": 0.17,  # 0.44
        "n_layers": 4,
        "kernal_size": 3,  # 7
        "activation": "gelu",  # "leaky_relu",
        "skip_dropout": 0.025,  # 0.015
        "temp_channel": False,  # True
        "use_pooling": True,
        "pool_type": "mean",  # max
        "temp_range": [
            275,
            325,
        ],  # does not influence the training, only used if temp_channel is True
    }

    # 2. Parse the arguments
    parsed_args = training.parse_args([])
    vars(parsed_args).update(my_args_dict_isothermal)

    # 3. Start the training
    print(f"'Start training for model: {parsed_args.model_name}")
    training.run_training(parsed_args)
    print("Training abgeschlossen!")


if __name__ == "__main__":
    main()
