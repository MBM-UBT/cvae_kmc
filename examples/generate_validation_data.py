from data.data_generation import kmc_trajectories

def main():
    my_args_dict = {
        "total_simulations": 12,
        "steps": 200,
        "temperature": [300.0],
        "dt": 5e-6,
        "num_workers": 4,
        "lattice_size": 256,
        "val_or_test": "val",
        "ood_test": 0,
        "random_start": 0,
    }

    parser = kmc_trajectories._setup_argument_parser()
    parsed_args = parser.parse_args([])
    vars(parsed_args).update(my_args_dict)

    kmc_trajectories.main(parsed_args)


if __name__ == "__main__":
    main()
