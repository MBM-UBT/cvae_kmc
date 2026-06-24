from data.data_generation import ml_trajectories


def main():
    my_args_dict = {
        "total_simulations": 100,
        "steps": 200,
        "temperature": [300.0],
        "ml_model_name": "isothermal_model",
        "lattice_size": 256,
        "pass_temp_to_ml": 0,  # set to 1 for temperature dependent model
        "ood_test": 0,
        "random_start": 0,
    }

    parser = ml_trajectories._setup_argument_parser()
    parsed_args = parser.parse_args([])
    vars(parsed_args).update(my_args_dict)

    ml_trajectories.main(parsed_args)


if __name__ == "__main__":
    main()
