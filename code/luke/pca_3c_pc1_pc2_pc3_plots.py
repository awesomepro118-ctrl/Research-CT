from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


METADATA_COLUMNS = {
    "dataset",
    "subject",
    "subject_id",
    "recording",
    "recording_id",
    "label",
    "condition",
    "source_file",
    "absolute_load",
    "load_in_sequence",
    "epoch_id",
    "epoch_uid",
    "row_in_epoch",
    "time_seconds",
    "epoch_start_seconds",
    "event_count",
    "event_labels",
    "chunk_id",
}


def get_eeg_feature_columns(dataframe):
    """
    Returns only the 95 EEG band-feature columns.
    """
    numeric_columns = dataframe.select_dtypes(include=np.number).columns

    feature_columns = [
        column
        for column in numeric_columns
        if column not in METADATA_COLUMNS
    ]

    return feature_columns


def flatten_split_file(
    input_file,
    output_file,
    expected_rows_per_epoch=24,
    expected_feature_count=95,
):
    """
    Convert each 24-row EEG epoch into one flattened sample.

    Input epoch:
        24 rows x 95 EEG features

    Output sample:
        1 row x 2280 flattened EEG features

    Epochs are grouped by epoch_uid and sorted by row_in_epoch.
    Metadata that should be constant within an epoch is copied from
    the first row of that epoch.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    dataframe = pd.read_csv(input_file)

    required_columns = {
        "epoch_uid",
        "row_in_epoch",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    feature_columns = get_eeg_feature_columns(dataframe)

    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} EEG features, "
            f"but found {len(feature_columns)}."
        )

    flattened_rows = []
    skipped_epochs = 0

    grouped = dataframe.groupby(
        "epoch_uid",
        sort=False,
    )

    for epoch_uid, epoch in grouped:

        epoch = epoch.sort_values(
            "row_in_epoch"
        )

        if len(epoch) != expected_rows_per_epoch:
            skipped_epochs += 1
            continue

        expected_row_numbers = list(
            range(expected_rows_per_epoch)
        )

        actual_row_numbers = (
            epoch["row_in_epoch"]
            .astype(int)
            .tolist()
        )

        if actual_row_numbers != expected_row_numbers:
            skipped_epochs += 1
            continue

        eeg_matrix = epoch[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )

        if eeg_matrix.shape != (
            expected_rows_per_epoch,
            expected_feature_count,
        ):
            skipped_epochs += 1
            continue

        flattened = eeg_matrix.reshape(-1)

        first_row = epoch.iloc[0]

        output_row = {}

        # Keep one copy of epoch-level metadata.
        for column in [
            "dataset",
            "subject",
            "subject_id",
            "recording",
            "recording_id",
            "label",
            "condition",
            "source_file",
            "absolute_load",
            "load_in_sequence",
            "epoch_id",
            "epoch_uid",
            "epoch_start_seconds",
            "event_count",
            "event_labels",
            "chunk_id",
        ]:
            if column in epoch.columns:
                output_row[column] = first_row[column]

        # Flatten in time-major order:
        # row 0's 95 features, row 1's 95 features, ... row 23.
        flat_index = 0

        for row_number in range(expected_rows_per_epoch):
            for feature_name in feature_columns:
                output_row[
                    f"t{row_number:02d}__{feature_name}"
                ] = flattened[flat_index]
                flat_index += 1

        flattened_rows.append(output_row)

    flattened_dataframe = pd.DataFrame(
        flattened_rows
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flattened_dataframe.to_csv(
        output_file,
        index=False,
    )

    flattened_feature_columns = [
        column
        for column in flattened_dataframe.columns
        if column.startswith("t")
        and "__" in column
    ]

    print("Input file:", input_file)
    print("Original rows:", len(dataframe))
    print(
        "Expected samples from rows / 24:",
        len(dataframe) / expected_rows_per_epoch,
    )
    print(
        "Flattened samples:",
        len(flattened_dataframe),
    )
    print(
        "Original EEG features per row:",
        len(feature_columns),
    )
    print(
        "Flattened features per sample:",
        len(flattened_feature_columns),
    )
    print(
        "Expected flattened dimension:",
        expected_rows_per_epoch * expected_feature_count,
    )
    print(
        "Skipped incomplete/invalid epochs:",
        skipped_epochs,
    )
    print("Saved:", output_file)

    return output_file


def load_flattened_data(file_path):
    dataframe = pd.read_csv(file_path)

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith("t")
        and "__" in column
    ]

    if len(feature_columns) != 24 * 95:
        raise ValueError(
            "Expected 2280 flattened EEG features, "
            f"but found {len(feature_columns)}."
        )

    X = dataframe[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    return dataframe, X, feature_columns



def make_pca_plots(
    train_pca,
    variance_table,
    loadings,
    output_directory,
):
    """
    Create:
    1. Scree plot
    2. Cumulative explained variance
    3. Full loading plots for PC1, PC2, PC3
    4. 24 x 95 loading surfaces for PC1, PC2, PC3
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_directory = Path(output_directory)
    plot_folder = output_directory / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # 1. Scree plot
    plt.figure(figsize=(9, 5))
    plt.plot(
        variance_table["principal_component"],
        variance_table["explained_variance_percent"],
        marker="o",
    )
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    plt.title("PCA scree plot")
    plt.tight_layout()
    plt.savefig(
        plot_folder / "01_pca_scree_plot.png",
        dpi=200,
    )
    plt.show()

    # 2. Cumulative explained variance
    plt.figure(figsize=(9, 5))
    plt.plot(
        variance_table["principal_component"],
        variance_table["cumulative_variance_percent"],
        marker="o",
    )
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("PCA cumulative explained variance")
    plt.tight_layout()
    plt.savefig(
        plot_folder / "02_pca_cumulative_variance.png",
        dpi=200,
    )
    plt.show()

    # PC1, PC2, PC3 loading plots and surfaces
    for pc_number in [1, 2, 3,4,5]:
        pc_name = f"PC{pc_number}"

        if pc_name not in loadings.columns:
            continue

        values = loadings[pc_name].to_numpy()

        if len(values) != 24 * 95:
            raise ValueError(
                f"{pc_name} has {len(values)} loadings; expected 2280."
            )

        # Full loading line plot
        plt.figure(figsize=(16, 6))
        plt.plot(
            np.arange(len(values)),
            values,
        )
        plt.xlabel("Flattened feature index")
        plt.ylabel(f"{pc_name} loading")
        plt.title(f"Principal Component {pc_number} loadings")
        plt.tight_layout()
        plt.savefig(
            plot_folder / f"0{pc_number + 2}_{pc_name.lower()}_all_loadings.png",
            dpi=200,
        )
        plt.show()

        # Reshape loadings back into 24 x 95
        loading_array = values.reshape(24, 95)

        x = np.arange(95)
        y = np.arange(24)
        X, Y = np.meshgrid(x, y)

        figure = plt.figure(figsize=(13, 8))
        axis = figure.add_subplot(
            111,
            projection="3d",
        )

        axis.plot_surface(
            X,
            Y,
            loading_array,
            linewidth=0,
            antialiased=True,
        )

        axis.set_xlabel("EEG feature index (0-94)")
        axis.set_ylabel("Row in epoch (0-23)")
        axis.set_zlabel(f"{pc_name} loading")
        axis.set_title(
            f"{pc_name} loading surface — 24 x 95"
        )

        plt.tight_layout()
        plt.savefig(
            plot_folder / f"0{pc_number + 5}_{pc_name.lower()}_loading_surface.png",
            dpi=200,
        )
        plt.show()

        print(
            f"{pc_name} loading array shape:",
            loading_array.shape,
        )

    print("Plots saved to:")
    print(plot_folder)

    return plot_folder

def plot_epoch_surface(
    flattened_file,
    output_directory,
    sample_index=0,
    expected_rows_per_epoch=24,
    expected_feature_count=95,
):
    """
    Take one flattened 2280-feature sample, reshape it back to
    a 24 x 95 array, and plot that matrix as a 3D surface.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    flattened_file = Path(flattened_file)
    output_directory = Path(output_directory)

    plot_folder = output_directory / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(
        flattened_file,
        skiprows=range(1, sample_index + 1),
        nrows=1,
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith("t")
        and "__" in column
    ]

    expected_flattened = (
        expected_rows_per_epoch
        * expected_feature_count
    )

    if len(feature_columns) != expected_flattened:
        raise ValueError(
            f"Expected {expected_flattened} flattened features, "
            f"but found {len(feature_columns)}."
        )

    flattened_values = (
        dataframe[feature_columns]
        .iloc[0]
        .to_numpy(dtype=np.float32)
    )

    epoch_array = flattened_values.reshape(
        expected_rows_per_epoch,
        expected_feature_count,
    )

    x = np.arange(expected_feature_count)
    y = np.arange(expected_rows_per_epoch)
    X, Y = np.meshgrid(x, y)

    figure = plt.figure(figsize=(13, 8))
    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    axis.plot_surface(
        X,
        Y,
        epoch_array,
        linewidth=0,
        antialiased=True,
    )

    axis.set_xlabel("EEG feature index (0-94)")
    axis.set_ylabel("Row in epoch (0-23)")
    axis.set_zlabel("EEG feature value")
    axis.set_title(
        f"24 x 95 EEG epoch surface — sample {sample_index}"
    )

    plt.tight_layout()

    output_path = (
        plot_folder
        / f"04_epoch_surface_sample_{sample_index}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.show()

    print("Epoch array shape:", epoch_array.shape)
    print("Surface saved:", output_path)

    return epoch_array

def run_pca(
    train_flat_file,
    validation_flat_file,
    holdout_flat_file,
    output_directory,
    number_of_components=20,
):
    

    output_directory = Path(
        output_directory
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, X_train, feature_columns = load_flattened_data(
        train_flat_file
    )

    validation_df, X_validation, validation_features = load_flattened_data(
        validation_flat_file
    )

    holdout_df, X_holdout, holdout_features = load_flattened_data(
        holdout_flat_file
    )

    if validation_features != feature_columns:
        raise ValueError(
            "Validation feature columns do not match training."
        )

    if holdout_features != feature_columns:
        raise ValueError(
            "Holdout feature columns do not match training."
        )

    max_components = min(
        X_train.shape[0],
        X_train.shape[1],
    )

    number_of_components = min(
        int(number_of_components),
        max_components,
    )

    print("Total flattened training samples:", len(X_train))
    print("Validation samples:", len(X_validation))
    print("Hold-out samples saved for later:", len(X_holdout))
    print("Number of original flattened features:", X_train.shape[1])
    print("Number of principal components:", number_of_components)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_holdout_scaled = scaler.transform(
        X_holdout
    )

    pca = PCA(
        n_components=number_of_components
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_validation_pca = pca.transform(
        X_validation_scaled
    )

    X_holdout_pca = pca.transform(
        X_holdout_scaled
    )

    explained = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(explained)

    print("\nPrincipal Components")
    print("--------------------")

    for index in range(number_of_components):
        print(
            f"PC{index + 1}: "
            f"Eigenvalue = {eigenvalues[index]:.4f}, "
            f"Variance Explained = {explained[index] * 100:.2f}%"
        )

    print(
        "\nTotal variance retained:",
        f"{cumulative[-1] * 100:.2f}%"
    )

    variance_table = pd.DataFrame(
        {
            "principal_component": np.arange(
                1,
                number_of_components + 1,
            ),
            "eigenvalue": eigenvalues,
            "explained_variance_percent": explained * 100,
            "cumulative_variance_percent": cumulative * 100,
        }
    )

    variance_file = (
        output_directory
        / "pca_variance.csv"
    )

    variance_table.to_csv(
        variance_file,
        index=False,
    )

    pc_columns = [
        f"PC{index + 1}"
        for index in range(
            number_of_components
        )
    ]

    train_pca_df = pd.DataFrame(
        X_train_pca,
        columns=pc_columns,
    )

    validation_pca_df = pd.DataFrame(
        X_validation_pca,
        columns=pc_columns,
    )

    holdout_pca_df = pd.DataFrame(
        X_holdout_pca,
        columns=pc_columns,
    )

    metadata_columns = [
        column
        for column in train_df.columns
        if column not in feature_columns
    ]

    train_output = pd.concat(
        [
            train_df[metadata_columns].reset_index(drop=True),
            train_pca_df,
        ],
        axis=1,
    )

    validation_metadata = [
        column
        for column in metadata_columns
        if column in validation_df.columns
    ]

    holdout_metadata = [
        column
        for column in metadata_columns
        if column in holdout_df.columns
    ]

    validation_output = pd.concat(
        [
            validation_df[validation_metadata].reset_index(drop=True),
            validation_pca_df,
        ],
        axis=1,
    )

    holdout_output = pd.concat(
        [
            holdout_df[holdout_metadata].reset_index(drop=True),
            holdout_pca_df,
        ],
        axis=1,
    )

    train_output.to_csv(
        output_directory / "train_pca.csv",
        index=False,
    )

    validation_output.to_csv(
        output_directory / "validation_pca.csv",
        index=False,
    )

    holdout_output.to_csv(
        output_directory / "holdout_pca.csv",
        index=False,
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=pc_columns,
    )

    loadings.to_csv(
        output_directory
        / "pca_loadings.csv"
    )

    print("\nTransformed Training Samples")
    print("----------------------------")
    print(
        train_output.head().to_string(
            index=False
        )
    )

    print("\nPCA Loadings")
    print("------------")
    print(
        loadings.head(30).to_string()
    )

    print("\nSaved PCA files to:")
    print(output_directory)

    plot_folder = make_pca_plots(
        train_pca=train_output,
        variance_table=variance_table,
        loadings=loadings,
        output_directory=output_directory,
    )

    epoch_array = plot_epoch_surface(
        flattened_file=train_flat_file,
        output_directory=output_directory,
        sample_index=0,
    )

    return {
        "scaler": scaler,
        "pca": pca,
        "train_pca": train_output,
        "validation_pca": validation_output,
        "holdout_pca": holdout_output,
        "variance": variance_table,
        "loadings": loadings,
        "plot_folder": plot_folder,
        "example_epoch_array": epoch_array,
    }
