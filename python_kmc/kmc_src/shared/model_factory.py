"""KMC Model Factory for creating models with minimal code duplication.

This module provides a factory function to instantiate KMC models
without needing to explicitly import dimension-specific classes.
"""


def create_kmc_model(dimension: str = "1d1", **kwargs):
    """Factory function to create KMC models for different dimensions.

    Args:
        dimension (str): The dimension of the model. Options: "1d1", "2d".
                        Defaults to "1d1".
        **kwargs: All parameters required by the specific model class.

    Returns:
        KMCModel_1D1 or KMCModel_2D: An instance of the requested KMC model.

    Raises:
        ValueError: If an unknown dimension is specified.

    Examples:
        >>> model = create_kmc_model("1d1", lattice_size=100, ...)
        >>> model = create_kmc_model("2d", size_x=50, size_y=50, ...)
    """
    dimension_lower = dimension.lower().replace("+", "")

    if dimension_lower == "1d1" or dimension_lower == "1d1":
        from kmc_model_1d1 import KMCModel_1D1

        return KMCModel_1D1(**kwargs)
    elif dimension_lower == "2d":
        from kmc_model_2d import KMCModel_2D

        return KMCModel_2D(**kwargs)
    else:
        raise ValueError(
            f"Unknown dimension: {dimension}. " f"Supported dimensions are: '1d1', '2d'"
        )
