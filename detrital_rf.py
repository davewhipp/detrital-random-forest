#!/home/dwhipp/mambaforge/bin/python
"""
detrital_rf.py

Uses the Random Forest classifier in Scikit Learn to extract important parameters
from the BHF detrital thermochron models.
"""

# Import libraries we need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def read_and_merge_data(fp, log_df_file, result_df_file):
    """Reads data files containing model log and results, merges, and cleans"""

    # Read data files
    log_df = pd.read_csv(fp + log_df_file)
    result_df = pd.read_csv(fp + result_df_file)

    # Concatenate DataFrames
    merge_df = pd.concat([log_df, result_df], axis=1)

    # Drop Model column
    merge_df = merge_df.drop(columns="Model")

    # Return DataFrame
    return merge_df


def inspect_catchment_data(
    catchment,
    df,
    fp="",
    show_plots=False,
    save_plots=False,
    pairplot=False,
    corrplot=False,
    echo_info=True,
):
    """Reviews data in DataFrame and prints summary statistics"""

    # Read in parameters of interest from model DataFrame
    rf_df = df[
        [
            catchment,
            "GHS",
            "LHS",
            "Paro",
            "Glaciers",
            "Moraines",
            "Non-glacial",
            "Slopes >30 degrees",
            "Slopes <10 degrees",
            "Erosion scaling",
        ]
    ]

    # Rename some columns
    rename_columns = {
        "GHS": "High grade rocks",
        "LHS": "Low grade rocks",
        "Glaciers": "Glacial",
    }
    rf_df = rf_df.rename(columns=rename_columns)

    # Drop columns with NA values
    rf_df = rf_df.dropna()

    if echo_info:
        print(f"- Number of models after dropping NA values: {len(rf_df)}.")
        # print("")

    # Make plots, if requested
    if show_plots:
        if pairplot or corrplot:
            # Make pair plot, save if requested
            if pairplot:
                sns.pairplot(rf_df.drop(catchment, axis=1))
                if save_plots:
                    plt.savefig(fp + f"pairplot_{pecube_model}.pdf", dpi=600)
                plt.show()
            # Make correlation plot, save if requested
            if corrplot:
                sns.heatmap(
                    rf_df.corr(), xticklabels=rf_df.columns, yticklabels=rf_df.columns
                )
                if save_plots:
                    plt.savefig(fp + f"corrplot_{pecube_model}.pdf", dpi=600)
                plt.show()
        else:
            print("Plots requested, but neither pairplot or corrplot was set to true.")
            print("Nothing to plot.")
            print("")

    return rf_df


def calculate_percentiles(
    catchment, df, low_percentile=75, high_percentile=90, echo_info=True
):
    """Calculates rating percentiles in passed DataFrame"""

    # Define rating percentiles
    rating_pctile = np.percentile(df[catchment], [low_percentile, high_percentile])

    if echo_info:
        print(f"- Rating percentiles: {rating_pctile}")
        print("")

    # Classify values based on percentiles
    df["n_rating"] = 0
    df["n_rating"] = np.where(df[catchment] < rating_pctile[0], 1, df["n_rating"])
    df["n_rating"] = np.where(
        (df[catchment] >= rating_pctile[0]) & (df[catchment] <= rating_pctile[1]),
        2,
        df["n_rating"],
    )
    df["n_rating"] = np.where(df[catchment] > rating_pctile[1], 3, df["n_rating"])

    return df


def run_model(random_search=True, grid_search=True, n_jobs=4):
    # Define file paths
    # fp = "/home/dwhipp/Model-output/BHF-model-summary-data/"
    fp = "/Users/whipp/Work/Modeling/Source/Python/detrital-random-forest-git/"

    # Create list of Pecube models to consider
    pecube_models = ["WB009", "WB012", "WB014"]

    # Create list of catchments to consider
    catchments = [
        "BH27",
        "BH389",
        "BH398",
        "BH402",
        "BH403",
        "BH404",
        "BH414",
        "BH435",
        "SK075",
    ]

    # Loop over different Pecube models
    for pecube_model in pecube_models:
        # Define log and result files
        log_file = f"BHF-model-log-{pecube_model}.csv"
        result_file = f"BHF-results-{pecube_model}.csv"

        # Read input data
        source_df = read_and_merge_data(fp, log_file, result_file)

        # Create empty DataFrame for feature importance summary
        summary_df = pd.DataFrame()

        # Create plot figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Loop over all catchments
        for catchment in catchments:
            print(80 * "*")
            print("")
            print(f"Working on catchment {catchment}...")
            # print("")

            # Process catchment data files, clean, and make plots (if requested)
            rf_df = inspect_catchment_data(
                catchment,
                source_df,
                fp=fp,
                show_plots=False,
                save_plots=False,
                pairplot=False,
                corrplot=False,
                echo_info=True,
            )

            # Calculate rating percentiles
            rf_df = calculate_percentiles(
                catchment, rf_df, low_percentile=75, high_percentile=90, echo_info=True
            )

            print(f"Creating test and training data for catchment {catchment}...")
            print("")
            # Create DataFrame copy without n_rating column
            X = rf_df.drop(
                [catchment, "n_rating"],
                axis=1,
            )
            # Create Series with only n_rating column
            y = rf_df["n_rating"]
            # Split arrays into random test/train subsets
            training, testing, training_labels, testing_labels = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            # Normalize the data
            sc = StandardScaler()
            normed_train_data = pd.DataFrame(
                sc.fit_transform(training), columns=X.columns
            )
            normed_test_data = pd.DataFrame(
                sc.fit_transform(testing), columns=X.columns
            )

            print("Performing random forest classification with default parameters...")
            print("")

            # Run the random forest classifier with the default parameters
            clf = RandomForestClassifier()
            clf.fit(training, training_labels)

            # Predict feature classes
            preds = clf.predict(testing)

            print("Training and testing scores:")
            print(f"Training: {clf.score(training, training_labels):.4f}")
            print(f"Testing: {clf.score(testing, testing_labels):.4f}")
            print("")

            print("Confusion matrix:")
            print("")
            print(metrics.confusion_matrix(testing_labels, preds, labels=[1, 2, 3]))
            print("")

            print("Feature importances")
            print("")
            print(
                pd.DataFrame(
                    clf.feature_importances_, index=training.columns
                ).sort_values(by=0, ascending=False)
            )
            print("")

            if random_search:
                print("Performing random grid search...")
                print("")

                # Number of trees in random forest
                n_estimators = np.linspace(
                    100, 3000, int((3000 - 100) / 200) + 1, dtype=int
                )
                # Number of features to consider at every split
                max_features = ["auto", "sqrt"]
                # Maximum number of levels in tree
                max_depth = [
                    1,
                    5,
                    10,
                    20,
                    50,
                    75,
                    100,
                    150,
                    200,
                ]
                # Minimum number of samples required to split a node
                min_samples_split = [1, 2, 5, 10, 15, 20, 30]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 3, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]
                # Criterion
                criterion = ["gini", "entropy"]

                random_grid = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": bootstrap,
                    "criterion": criterion,
                }

                # Perform random grid classification
                rf_base = RandomForestClassifier()
                rf_random = RandomizedSearchCV(
                    estimator=rf_base,
                    param_distributions=random_grid,
                    n_iter=30,
                    cv=5,
                    verbose=2,
                    random_state=42,
                    n_jobs=n_jobs,
                )
                rf_random.fit(training, training_labels)
                print("")

                print(f"Best parameters: {rf_random.best_params_}")
                print("")

                print("Training and testing scores:")
                print(f"Training: {rf_random.score(training, training_labels):.4f}")
                print(f"Testing: {rf_random.score(testing, testing_labels):.4f}")
                print("")

                print("Feature importances")
                print("")
                print(
                    pd.DataFrame(
                        rf_random.best_estimator_.feature_importances_,
                        index=training.columns,
                    ).sort_values(by=0, ascending=False)
                )
                print("")

                # Perform grid search
                if grid_search:
                    # Find parameter bounds
                    n_est_range = np.linspace(
                        rf_random.best_params_["n_estimators"] - 100,
                        rf_random.best_params_["n_estimators"] + 100,
                        5,
                        dtype=int,
                    )
                    max_depth_range = np.arange(
                        rf_random.best_params_["max_depth"] - 30,
                        rf_random.best_params_["max_depth"] + 30,
                        10,
                    )
                    min_samples_split_bf = rf_random.best_params_["min_samples_split"]
                    if min_samples_split_bf < 3:
                        min_samples_split_range = np.arange(
                            min_samples_split_bf, min_samples_split_bf + 3, 1
                        )
                    else:
                        min_samples_split_range = np.arange(
                            min_samples_split_bf - 1, min_samples_split_bf + 2, 1
                        )
                    min_samples_leaf_bf = rf_random.best_params_["min_samples_leaf"]
                    if min_samples_leaf_bf < 3:
                        min_samples_leaf_range = np.arange(
                            min_samples_leaf_bf, min_samples_leaf_bf + 4, 1
                        )
                    else:
                        min_samples_leaf_range = np.arange(
                            min_samples_leaf_bf - 1, min_samples_leaf_bf + 3, 1
                        )
                    criterion = rf_random.best_params_["criterion"]
                    bootstrap = rf_random.best_params_["bootstrap"]

                    print(f"n_est_range: {n_est_range}")
                    print(f"max_depth_range: {max_depth_range}")
                    print(f"min_samples_split_range: {min_samples_split_range}")
                    print(f"min_samples_leaf_range: {min_samples_leaf_range}")
                    print(f"criterion: {criterion}")
                    print(f"bootstrap: {bootstrap}")
                    print("")

                    # Create grid for grid-based random forest search
                    param_grid = {
                        "n_estimators": n_est_range,
                        "max_depth": max_depth_range,
                        "min_samples_split": min_samples_split_range,
                        "min_samples_leaf": min_samples_leaf_range,
                    }

                    # Base model
                    rf_grid = RandomForestClassifier(
                        criterion=criterion, bootstrap=bootstrap
                    )
                    # Instantiate the grid search model
                    grid_rf_search = GridSearchCV(
                        estimator=rf_grid,
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=n_jobs,
                        verbose=2,
                    )
                    grid_rf_search.fit(training, training_labels)

                    print("Training and testing scores:")
                    print(
                        f"Training: {grid_rf_search.score(training, training_labels):.4f}"
                    )
                    print(
                        f"Testing: {grid_rf_search.score(testing, testing_labels):.4f}"
                    )
                    print("")

                    best_rf_grid = grid_rf_search.best_estimator_
                    print(f"Best parameters: {grid_rf_search.best_params_}")
                    print("")

                    print(
                        pd.DataFrame(
                            grid_rf_search.best_estimator_.feature_importances_,
                            index=training.columns,
                        ).sort_values(by=0, ascending=False)
                    )
                    print("")

                    # Append results to a summary DataFrame that can be used to compare results
                    summary_df[catchment] = pd.DataFrame(
                        rf_random.best_estimator_.feature_importances_,
                        index=training.columns,
                    )

                    # Write summary DF output to file
                    out_file = f"rf_summary_{pecube_model}.csv"
                    summary_df.to_csv(out_file, float_format=".4f")

                    # Make heatmap plot and write to file
                    sns.heatmap(summary_df.round(2), annot=True, ax=ax)
                    plt.savefig(f"{fp}rf_summary_plot_{pecube_model}.pdf", dpi=600)

    print(80 * "*")


# Run model
run_model(random_search=True, grid_search=False)
