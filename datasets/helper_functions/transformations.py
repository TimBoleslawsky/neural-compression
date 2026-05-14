# Registry mapping transformation names to functions
TRANSFORMATIONS = {}


def get_transformation(name: str):
    """
    Get a transformation function by name.
    """
    if name not in TRANSFORMATIONS:
        available = ', '.join(TRANSFORMATIONS.keys())
        raise ValueError(
            f"Unknown transformation '{name}'. "
            f"Available transformations: {available}"
        )
    return TRANSFORMATIONS[name]
