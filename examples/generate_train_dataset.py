import os

from data.data_generation import training_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # For isothermal model, comments afterwards are for the temperature dependent model
    # train_sims + validation_sims + test_sims should be == samples_per_temp,
    # usually everything is for training, since for validation other data is needed
    my_args_dict = {
        "output_dir": os.path.abspath(
            os.path.join(BASE_DIR, "../data/training_datasets/test_dataset")
        ),  # Adjust the path as needed
        "lattice_size": 256,
        "dt": 5e-6,
        "depth": 75,
        "leaves": 1,
        "num_workers": 12,
        "possible_temps": [300],  # for temp model set: None
        "temp_range": None,  # for temp model set: [275, 325]
        "samples_per_temp": 1250,  # for temp model set: 5000
        "train_sims": 1250,  # for temp model set: 5000
        "validation_sims": 0,
        "test_sims": 0,
        "log_file": "test_dataset.log",
        "warmup": 0,
        "random_start": 0,
        "empty_probability": 0,
        "max_height": 0,
    }

    parser = training_data._setup_argument_parser()
    parsed_args = parser.parse_args([])
    vars(parsed_args).update(my_args_dict)

    training_data.main(parsed_args)


if __name__ == "__main__":
    main()
