#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import os
from pathlib import Path


def read_sample_list(path):
    """
    GUI-function. Function that reads the sample names .csv / txt file and returns it as a list.
    :param path: Path to file
    :return: List with sample names
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'File "{path.resolve()}" could not be found ')

    suffix = Path(path).suffix

    if suffix == ".csv":
        sample_list = pd.read_csv(path, header=None, usecols=[0])[0].values.tolist()

    elif suffix == ".txt":
        sample_list = pd.read_csv(path, header=None, usecols=[0], sep='\t')[0].values.tolist()

    else:
        raise FileNotFoundError("Please provide a comma-separated values file (.csv) or "
                                "a tab-separated values file (.txt)")

    return sample_list


def gradual_rotate_grid_elements(grid: np.array, r: int, direction: str, for_integer: bool = True):
    """
    This is the core function of MultiplexMS moving rows and columns to produce either the rearranged grid (
    create_multiplex_ms_grid()) or solve/deconvolute the rearranged grids (revert_multiplex_ms_grid()).
    This function explicitly performs only one of the rotations. Two rotations are needed to create or
    solve MultiplexMS grids.
    :param for_integer: Adaptation of the function to be used for string/float-type grids if False (default True)
    :param grid: Square grid
    :param r: Grid-side length
    :param direction: Either create a rearranged grid ('rearrange') or solve a rearranged grid ('else')
    :return:
    """

    # Use float/object to enable np.nan inserting. However, np.is.nan() only works with 'float', not object,
    # so both are needed
    if for_integer:
        lower_grid = grid.copy().astype('float')
        upper_grid = grid.copy().astype('float')
    else:
        lower_grid = grid.copy().astype('object')
        upper_grid = grid.copy().astype('object')

    # Use this if-statement to create the rearranged grid
    if direction == 'rearrange':
        # Fill triangles with np.nan to enable stepwise MultiplexMS rotation logic
        # Example 1. row -> no rotation; 2. row -> 1 rotation; 3. row -> 2 rotations; ...
        utri_idx = np.triu_indices(r, 1)
        ltri_idx = np.tril_indices(r)

        lower_grid[utri_idx] = np.nan
        upper_grid[ltri_idx] = np.nan

        # Stack both grids to build rotated grid
        stack = np.vstack((lower_grid, upper_grid))

        # Flatten the grid (ie finalize rotation) and return the rotated r x r grid
        flattened_stack = stack.flatten('F')
        if for_integer:
            flattened_stack = flattened_stack[~np.isnan(flattened_stack)]
        else:
            flattened_stack = np.array(list(filter(lambda v: v == v, flattened_stack)))

        result = flattened_stack.reshape((r, r)).T

    # Use this part for the solving/deconvoluting rearranged grids
    else:
        utri_idx = np.triu_indices(r)
        ltri_idx = np.tril_indices(r, -1)

        np.fliplr(lower_grid)[utri_idx] = np.nan
        np.fliplr(upper_grid)[ltri_idx] = np.nan

        stack = np.hstack((lower_grid, upper_grid))

        flattened_stack = stack.flatten()

        if for_integer:
            flattened_stack = flattened_stack[~np.isnan(flattened_stack)]
        else:
            flattened_stack = np.array(list(filter(lambda v: v == v, flattened_stack)))

        result = flattened_stack.reshape((r, r)).T

    return result


def create_multiplex_ms_grid(grid: np.array = None, for_integer: bool = True):
    """
    This function creates or accepts an initial r*r grid.
    The function rearranges the initial grid and returns it.
    :param for_integer: Boolean to indicate if integer values are exclusively used in the grid
    :param grid: Square np.array
    :return: Rearranged grid
    """

    side_length = grid.shape[0]

    first_rotation = gradual_rotate_grid_elements(grid, side_length, direction='rearrange',
                                                  for_integer=for_integer)
    second_rotation = gradual_rotate_grid_elements(first_rotation.T, side_length, direction='rearrange',
                                                   for_integer=for_integer)

    return second_rotation.T


def distribute_samples_in_grids_with_placeholders(sample_count: int, max_grid_size: int, grid_count: int):
    """
    Create a list with evenly distributed sample counts with space for placeholders
    :param sample_count: Number of samples
    :param max_grid_size: Number of positions in the square grid (eg 10x10 -> max_grid_size = 100)
    :param grid_count: Number of grids used in the experiment
    :return: List of sample counts per grid
    """
    # Arrange samples in the number of necessary grids, so that they spread as evenly as possible
    placeholder_sample_count = grid_count * max_grid_size - sample_count
    placeholder_per_grid = placeholder_sample_count // grid_count
    extra_placeholder_per_grid = placeholder_sample_count - placeholder_per_grid * grid_count

    # If placeholders cannot be distributed evenly, fill first grids with more placeholders
    sample_count_per_grid = [max_grid_size - placeholder_per_grid - 1 if extra_placeholder_per_grid > i else
                             max_grid_size - placeholder_per_grid for i in range(int(grid_count))]

    return sample_count_per_grid


def create_initial_grid_with_placeholders(r: int, remaining_samples_count: int, sample_names: np.array, seed: int):
    """
    Based on the square grid side length r, create a square grid and add placeholder
    cells into that grid until the remaining sample count plus placeholder cells equal r * r.
    The placeholder cells are spaced out randomly in that grid.
    :param seed: Seed value to allow reproducible placeholder positions
    :param sample_names: np.array with all sample names
    :param r: Grid side length
    :param remaining_samples_count: Number of samples to place in the grid (r*r) which is less than r*r
    :return: Grid with placeholder samples
    """

    # # Even spread solution
    # placeholder_count = r * r - remaining_samples_count
    # # Every nth step, a placeholder will be used
    # step = r * r // placeholder_count
    # idx = np.arange(0, r * r, step)[0:placeholder_count]

    # Pure random selection, but the first position (top left of the grid) cannot be a placeholder
    placeholder_count = r * r - remaining_samples_count
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(np.arange(1, r * r), placeholder_count, replace=False)

    grid_content_1d = np.array(['sample' for _ in range(r * r)], dtype='object')
    grid_content_1d[idx] = 'placeholder'

    placeholder_grid = np.reshape(grid_content_1d, (r, r))
    indices = np.where(placeholder_grid != 'placeholder')
    placeholder_grid[indices] = sample_names

    return placeholder_grid


# noinspection PyTypeChecker,PyUnboundLocalVariable
def create_user_tables_output(sample_names: list, r: int = 10, rel_path: str = "",
                              randomize_grids: bool = False, seed: int = None,
                              experiment_tag: str = "", verbose: bool = True):
    """
    GUI-function. Function that creates the preparation and grid csv files, the
    user needs in order to proceed with multiplexing the samples and to later on
    deconvolute the pooled-samples feature table with the grids.
    :param verbose: Print information about the procedure (default False)
    :param rel_path: Path where table should be saved in
    :param sample_names: List with all sample names
    :param r: Grid side length
    :param randomize_grids: Boolean to indicate if sample names shall be randomized
    :param seed: Seed to allow reproducible shuffling of sample names
    :param experiment_tag: Optional tag that will be added to the csv file names
    :return: Folder with .csv preparation tables and grids
    """
    max_grid_size = r * r

    # Randomize samples if desired
    if randomize_grids:
        if seed is None:
            np.random.shuffle(sample_names)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(sample_names)

    # Get number of samples
    sample_count = len(sample_names)
    if verbose:
        print(f"{str(sample_count)} samples were found")

    # Modify output experiment tag, if exists
    if experiment_tag != "" and experiment_tag is not None:
        experiment_tag = "_" + experiment_tag
    else:
        experiment_tag = ""

    # Determine the number of grids that need to be created and also determine the number of placeholder
    # per grid if the number of samples does not fit perfectly in all the grids.
    grid_count = np.ceil(sample_count / max_grid_size)
    if sample_count % max_grid_size != 0:
        if verbose:
            print('Placeholder samples will be added to the grids')

        sample_count_per_grid = distribute_samples_in_grids_with_placeholders(sample_count,
                                                                              max_grid_size,
                                                                              grid_count)

        # Obtain a list of lists of with samples per grids and add the placeholder evenly in that list
        grid_list = []
        idx = 0
        for i in sample_count_per_grid:
            grid_list.append(sample_names[idx: idx + int(i)])
            idx += int(i)

    else:
        grid_list = [sample_names[x: x + max_grid_size] for x in
                     range(0, int(max_grid_size * grid_count), max_grid_size)]

    for n, grid in enumerate(grid_list):
        if verbose:
            print(f"Create the initial and corresponding rearranged grid number {n + 1}")

        # If placeholders are needed, add them randomly into the grid
        if len(grid) < max_grid_size:
            initial_grid = create_initial_grid_with_placeholders(r, len(grid), grid, seed=n)
        else:
            initial_grid = np.reshape(grid, (r, r))

        # # This part can be used for all tables
        # Prepare consecutive grid + column/row names
        grid_name = f"grid_{n + 1}_"

        row_names = [f"{grid_name}row_{i}_initial" for i in range(1, r + 1)]
        col_names = [f"{grid_name}column_{i}_initial" for i in range(1, r + 1)]

        # Create pandas dataframe of the initial grid
        df_initial = pd.DataFrame(initial_grid, columns=col_names, index=row_names)

        # Prepare rearranged grid
        sample_names_array_2d_rearranged = create_multiplex_ms_grid(initial_grid, for_integer=False)

        # Prepare consecutive grid + column/row names
        row_names = [f"{grid_name}row_{i}_rearranged" for i in range(1, r + 1)]
        col_names = [f"{grid_name}column_{i}_rearranged" for i in range(1, r + 1)]
        # Create pandas dataframe of the rearranged grid
        df_rearranged = pd.DataFrame(sample_names_array_2d_rearranged, columns=col_names, index=row_names)

        # Prepare table to prepare the pooled samples
        for j, df in enumerate([df_initial, df_rearranged]):
            helper_table = df.copy()
            helper_table_2 = df.T.copy()
            helper_table.columns = helper_table_2.columns = [f"sample_{i}" for i in range(1, r + 1)]
            if j == 0:
                preparation_table_initial = pd.concat([helper_table, helper_table_2])
            else:
                preparation_table_rearranged = pd.concat([helper_table, helper_table_2])

        if verbose:
            print('Produce .csv files.')

        # Create subfolders for grids and preparation tables
        for i in ['grids', 'preparation_tables']:
            temp_path = Path(rel_path, i)
            temp_path.mkdir(exist_ok=True)

        df_initial.to_csv(Path(rel_path, 'grids', f"{grid_name}initial{experiment_tag}.csv"))

        df_rearranged.to_csv(Path(rel_path, 'grids', f"{grid_name}rearranged{experiment_tag}.csv"))

        preparation_table_initial.T.to_csv(
            Path(rel_path, 'preparation_tables', f"{grid_name}initial_preparation_table{experiment_tag}.csv"))

        preparation_table_rearranged.T.to_csv(
            Path(rel_path, 'preparation_tables', f"{grid_name}rearranged_preparation_table{experiment_tag}.csv"))

    return


def combine_grids(path_to_grids: str = ''):
    """
    GUI-function. This function takes a path and finds initial and rearranged grids and merges them, so that they can
    be handled in a for loop. Also does a gut, using the file count as an indicator if files are missing, since
    grids should come as pairs of two.
    :param path_to_grids: Path to the folder containing the grids
    :return:
    """

    INITIAL = '_initial'
    REARRANGED = '_rearranged'

    grid_list = os.listdir(Path(path_to_grids))

    initial = [initial for initial in grid_list if INITIAL in initial]
    rearranged = [rearranged for rearranged in grid_list if REARRANGED in rearranged]

    file_count = len(initial) + len(rearranged)
    if file_count == 0:
        raise FileNotFoundError('No grid files seem to be present in the selected folder or they have been '
                                'renamed - the names should be grid_x_y_z.csv, whereby x indicates the '
                                'grid number, y whether if it is the initial or rearranged version, and z '
                                'the assigned experiment tag (optional)')

    # Match initial.csv files with rearranged.csv files
    combined_grids = []
    for file in initial:
        query = file.replace(INITIAL, REARRANGED)

        corresponding_file = [hit for hit in rearranged if query == hit]
        if len(corresponding_file) == 0:
            raise FileNotFoundError(f'Corresponding rearranged grid file is missing: {query}')

        combined_grids.append((file, query))
        rearranged.remove(query)

    if len(rearranged) != 0:
        raise FileNotFoundError(f'The following rearranged.csv file(s) exist without a correspoding intial.csv file: '
                                f'{" ,".join(rearranged)}')

    return combined_grids


def prepare_user_feature_presence_table(path_to_fpt: str, samples_in_columns: bool = True):
    """
    GUI-function. This function reads the pooled samples feature table and binarizes it.
    :param samples_in_columns: Format of feature presence table ('Samples in columns')
    :param path_to_fpt: Path to feature presence table (.csv format)
    :return:
    """
    # Import feature_presence_table:
    fpt = pd.read_csv(Path(path_to_fpt), index_col=0)

    if fpt.isnull().values.any():
        raise ValueError('Missing values in the feature presence table '
                         'detected! Please fix the table')

    # Check format of feature table -> Samples in columns or rows
    if samples_in_columns:
        fpt = fpt.T

    # convert intensities into presence and absence
    fpt[fpt > 0] = 1

    return fpt


def verify_pooled_sample_names(feature_presence_table: pd.DataFrame, path_to_grids: str):
    """
    GUI function: This function compares the pooled sample names in the automatically generated grids with the
    sample names in the feature table. The tool expects that the grid names represent the true expected names,
    and only the feature table might have wrong names. The missing, expected names are returned.

    :param feature_presence_table:
    :param path_to_grids:
    :return:
    """

    # Get all sample names from the grids that have been created by the tool
    grids = os.listdir(Path(path_to_grids))

    pooled_sample_names = set()
    for grid in grids:
        grid_df = pd.read_csv(Path(path_to_grids, grid), index_col=0)
        [pooled_sample_names.add(i) for i in grid_df.index.values]
        [pooled_sample_names.add(i) for i in grid_df.columns.values]

    # Read the feature table and get all sample names
    sample_names_fpt = set(feature_presence_table.index.values)

    # Compare the expected name list with the feature table name list
    # --> if empty, all files have been found
    check_perfect_overlap = pooled_sample_names - sample_names_fpt

    if check_perfect_overlap:
        raise ValueError(f"The following pooled samples have not been found in the feature table: "
                         f"{', '.join(sorted(list(check_perfect_overlap)))}")
    else:
        print('All pooled samples have been found in the feature table - :-)')


def revert_multiplex_ms_grid(flattened_grid: np.array = None, for_integer: bool = True):
    """
    Takes a flattened squared grid as input.
    :param flattened_grid: 1-D np.array of a flattened MultiplexMS grid
    :param for_integer: Boolean to indicate if integer values are exclusively used in the grid
    :return:
    """

    # Determine side length r
    side_length = int(len(flattened_grid) ** .5)

    grid = flattened_grid.copy().reshape((side_length, side_length))

    first_rotation = gradual_rotate_grid_elements(grid, side_length, direction='solve',
                                                  for_integer=for_integer)
    second_rotation = gradual_rotate_grid_elements(first_rotation, side_length, direction='solve',
                                                   for_integer=for_integer)

    return second_rotation.flatten()


def observed_presence_to_flattened_grid(column_vector: np.array, row_vector: np.array):
    """
    This function uses the presence/absence vectors from the pooled samples as input
    and returns the deconvoluted grid as a linear vector.

    :param column_vector: Column np.array vector
    :param row_vector: Row np.array vector
    :return:
    """
    row_vector = np.reshape(row_vector, (len(row_vector), 1))
    column_vector = np.reshape(column_vector, (1, len(column_vector)))

    return (row_vector @ column_vector).flatten()


def deconvolute_multiplexed_grid(initial_grid: pd.DataFrame, rearranged_grid: pd.DataFrame,
                                 feature_presence_table: pd.DataFrame):
    """
    This function deconvolutes or decodes the pooled samples feature table, using the information of the initial
    and rearranged grids.
    :param initial_grid: Initial grid pd.DataFrame
    :param rearranged_grid: Rearranged grid pd.DataFrame
    :param feature_presence_table: Feature presence table pd.DataFrame
    :return:
    """

    fpt = feature_presence_table.copy()
    ig = initial_grid.copy()
    rg = rearranged_grid.copy()

    sample_names = ig.to_numpy().flatten()

    # Get the feature presence table that only contains the pooled samples
    pooled_df = fpt.T[ig.columns.tolist() + ig.index.tolist() + rg.columns.tolist() + rg.index.tolist()]

    ig_columns = [True if sample in ig.columns else False for sample in pooled_df.columns]
    ig_rows = [True if sample in ig.index else False for sample in pooled_df.columns]
    rg_columns = [True if sample in rg.columns else False for sample in pooled_df.columns]
    rg_rows = [True if sample in rg.index else False for sample in pooled_df.columns]

    # Use numpy array for fast indexing
    pooled_df = pooled_df.to_numpy().astype('int8')

    deconvoluted_grid = []
    for row in pooled_df:
        # Create the observed grid for the initial grid
        column_grid = row[ig_columns]
        row_grid = row[ig_rows]
        deconvoluted_initial_grid = observed_presence_to_flattened_grid(column_grid, row_grid)

        # Create the observed grid for grid 2
        column_grid = row[rg_columns]
        row_grid = row[rg_rows]
        deconvoluted_rearranged_grid = observed_presence_to_flattened_grid(column_grid, row_grid)

        # Since the second grid was 'rearranged', the result has to be transformed to the initial grid form.
        deconvoluted_rearranged_grid = revert_multiplex_ms_grid(deconvoluted_rearranged_grid)

        # We use a logical and gate to determine the overlap between both vectors. Only overlapping ones
        # return a 1
        overlap = np.logical_and(deconvoluted_initial_grid, deconvoluted_rearranged_grid).astype(int)
        deconvoluted_grid.append(overlap)

    deconvoluted_df = pd.DataFrame(deconvoluted_grid, index=fpt.columns, columns=sample_names)

    if 'placeholder' in sample_names:
        deconvoluted_df = deconvoluted_df.drop(["placeholder"], axis=1)

    return deconvoluted_df.astype('int8')


def batch_deconvolute(path_to_grids: str, combined_grids: list, feature_presence_table: pd.DataFrame,
                      file_destination: str):
    """
    GUI-function: This function processes several pairs of initial and rearranged grids that belong to the
    same pooled sample list and produces one concatenated feature_table.
    :param path_to_grids: Path to the folder with initial and rearranged grids
    :param combined_grids: List with tuples of initial and rearranged grids (from combine_grids() function)
    :param feature_presence_table: Feature presence table as pd.DataFrame
    :param file_destination: File path for saving
    :return:
    """

    deconvoluted_df_list = []
    for initial, rearranged in combined_grids:
        initial_grid = pd.read_csv(Path(path_to_grids, initial), index_col=0)
        rearranged_grid = pd.read_csv(Path(path_to_grids, rearranged), index_col=0)

        deconvoluted_df = deconvolute_multiplexed_grid(initial_grid=initial_grid,
                                                       rearranged_grid=rearranged_grid,
                                                       feature_presence_table=feature_presence_table)

        deconvoluted_df_list.append(deconvoluted_df)

    if len(deconvoluted_df_list) > 1:
        deconvoluted_feature_table = pd.concat(deconvoluted_df_list, axis=1)
    else:
        deconvoluted_feature_table = deconvoluted_df_list[0]

    # Check if resulting feature table contains any features that are absent in all samples
    print('Checking the deconvoluted feature table')
    count_absent_features(deconvoluted_feature_table.T, verbose=True)

    # Features in rows, samples in columns
    print('Saving the deconvoluted feature table as deconvoluted_feature_table.csv')
    # noinspection PyTypeChecker
    deconvoluted_feature_table.to_csv(Path(file_destination, 'deconvoluted_feature_table.csv'))
    print('Deconvolution done')


def count_absent_features(feature_presence_table: pd.DataFrame, verbose: bool = True):
    """
    GUI-function. Return number of features that only contain zero values.
    :param verbose:
    :param feature_presence_table:
    :return:
    """

    # Features in columns, samples in rows
    feature_presence_count = feature_presence_table.sum(axis=0).values
    absent_feature_count = int(sum(feature_presence_count == 0))

    if verbose:
        if absent_feature_count > 0:
            print('_' * 50)
            print(f"WARNING! {absent_feature_count} feature(s) have been found that are NOT present in any sample.\n"
                  f"Consider removing those features from the table by using the Cleaning option in the "
                  f"Companion tool.")
            print('_' * 50)
        else:
            print(f"All {feature_presence_table.shape[0]} features are present in at least one sample. :-)")

    else:
        return absent_feature_count


def remove_features_from_table(feature_presence_table: pd.DataFrame,
                               file_path: str, out_path: str, critical_percentage: int = None):
    """
    Check if provided deconvoluted feature table contains any features that are absent in all samples and remove those.
    In addition, a critical percentage allows to remove features that are present in more than x% of the samples.
    :param critical_percentage:
    :param file_path: Used to strip name from file
    :param out_path: Path for saving
    :param feature_presence_table: Feature presence table as pd.DataFrame
    :return:
    """

    # Features in columns, samples in rows
    df = feature_presence_table.copy()

    absent_feature_count = count_absent_features(df, verbose=False)

    if absent_feature_count == 0 and critical_percentage is None:
        print('No totally absent features detected - NO action performed!')

    else:
        present_features = df.sum(axis=0) > 0
        present_features = present_features[present_features].index
        pruned_table = df[present_features]

        print(f'{absent_feature_count} absent features have been removed from the feature table')

        if pruned_table.shape[1] == 0:
            print('No feature can be found in the table after pruning - The feature presence table was empty from '
                  'the beginning')

        if critical_percentage is not None:
            critical_percentage = abs(critical_percentage)
            percentage_presence = pruned_table.sum(axis=0) / df.shape[0] * 100
            features_to_keep = percentage_presence[percentage_presence < critical_percentage].index
            if len(features_to_keep) == 0:
                print('No feature can be found in the table after pruning - Please adjust (raise) critical percentage')
                return
            print(f'{pruned_table.shape[1] - len(features_to_keep)} more features have been removed, using a '
                  f'critical percentage threshold of {critical_percentage}%')
            pruned_table = pruned_table[features_to_keep]
            print(f'In total {absent_feature_count + len(features_to_keep)} features have been removed - '
                  f'{pruned_table.shape[1]} remain in the deconvoluted feature table')

        # Prepare file
        file_name = Path(file_path).name
        new_file_name = Path(out_path, 'cleaned_' + file_name)

        # Features in rows, samples in columns
        pruned_table.T.to_csv(new_file_name)


if __name__ == '__main__':
    pass
