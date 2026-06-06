"""Model registry for regression experiments."""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def get_regression_models(random_state: int = 42) -> dict:
    return {
        # "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(
            # Your original max_depth=10 was already strong.
            # 8 performed slightly better on v1, but 10 is safer overall.
            max_depth=10,

            # Keeps the tree from creating leaves based on only 1 row.
            # Very light regularization.
            min_samples_leaf=1,

            random_state=random_state,
        ),

        # "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            # More trees than before, but not too many.
            # This may improve stability without changing the model behavior too much.
            n_estimators=200,

            # Keep your original depth.
            # max_depth=20 was too flexible and max_features="sqrt" hurt badly.
            max_depth=10,

            # Do NOT use max_features="sqrt" here.
            # Your dataset has important one-hot encoded features,
            # and sqrt was hiding too many of them from each split.

            random_state=random_state,

            # Uses all CPU cores.
            n_jobs=-1,
        ),

        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
    }