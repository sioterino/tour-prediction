"""Model registry for regression experiments."""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def get_regression_models(random_state: int = 42) -> dict:
    """
    Get a dictionary of unfitted regression models with centralized hyperparameters.

    Args:
        random_state: Random state for reproducibility (default: 42).

    Returns:
        Dictionary with model names as keys and unfitted sklearn regressors as values.
    """
    return {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
    }
