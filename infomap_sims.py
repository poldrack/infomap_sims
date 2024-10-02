# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import scipy.io
import numpy as np
import pandas as pd
from infomap import Infomap
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import networkx as nx


# utility functions
def r_to_z(r):
    # if it's not a numpy array, make it one
    if not isinstance(r, np.ndarray):
        r = np.array([r])
    # fisher transform
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    z[np.where(np.isinf(z))] = 0
    z[np.where(np.isnan(z))] = 0

    return z


def z_to_r(z):
    # inverse transform
    return (np.exp(2.0 * z) - 1) / (np.exp(2.0 * z) + 1)


# functions for simulation
def load_priors(infile="priors.mat", size_ratio=2):
    mat = scipy.io.loadmat(infile)

    # limit size for computational reasons - only use 1/size_ratio of data
    sizelimit = int(mat["Priors"][0][0][0].shape[0] / size_ratio)

    fcpriors = mat["Priors"][0][0][0][:sizelimit]
    spatialpriors = mat["Priors"][0][0][1][:sizelimit]
    names = [str(i[0][0]) for i in mat["Priors"][0][0][2]]
    return mat, fcpriors, spatialpriors, names


def get_maxprob_labels(priors):
    return np.argmax(priors, axis=1).astype(int)


def matching_matrix_to_graph(matching_matrix, density=0.05):
    G = nx.Graph()
    G.add_nodes_from(range(matching_matrix.shape[0]))
    thresh = np.percentile(matching_matrix, 100 * (1 - density))
    match_pairs = np.where(matching_matrix >= thresh)
    match_pairs_clean = [
        (int(i), int(j)) for i, j in zip(match_pairs[0], match_pairs[1]) if i != j
    ]
    G.add_edges_from(match_pairs_clean)
    return G


def get_matching_matrix(maxprob_fc):
    matching_matrix = (maxprob_fc[:, None] == maxprob_fc).astype(np.float32)
    # set diagonal to zero
    np.fill_diagonal(matching_matrix, 0)

    # create a z-scored version for use in generating noisy versions
    matching_matrix_z = r_to_z(matching_matrix)

    return matching_matrix, matching_matrix_z


def get_match_pairs(matching_matrix, thresh=0.99):
    match_pairs = np.where(matching_matrix >= thresh)
    match_pairs_clean = [
        (int(i), int(j)) for i, j in zip(match_pairs[0], match_pairs[1]) if i != j
    ]
    return match_pairs_clean


def run_infomap(G, fcpriors, verbose=False):
    im = Infomap(silent=True)

    _ = im.add_networkx_graph(G)
    im.run()
    if verbose:
        print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    module_id_dict = {node.node_id: node.module_id for node in im.tree if node.is_leaf}
    module_list = np.array([module_id_dict[i] for i in range(fcpriors.shape[0])])
    true_labels = get_maxprob_labels(fcpriors)
    cm = confusion_matrix(true_labels, module_list)

    # relabel modules to match true labels
    cm_argmax = cm.argmax(axis=0)
    module_list_relabeled = np.array([cm_argmax[i] for i in module_list])

    return module_list, module_list_relabeled


def create_noisy_matching_matrix(
    matching_matrix_z, noise_level=1, cutoff=0.6, seed=None
):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, matching_matrix.shape)
    cutoff_z = r_to_z(cutoff)[0]
    matching_matrix_z_noisy = (
        np.clip(matching_matrix_z, -1 * cutoff_z, cutoff_z) + noise
    )
    matching_matrix_noisy = z_to_r(matching_matrix_z_noisy)
    return matching_matrix_noisy


def get_module_counts(module_list, true_labels):
    modules = sorted(np.unique(true_labels).tolist())
    counter = {i: int(np.sum(module_list == i)) for i in modules}
    return counter


if __name__ == "__main__":
    nruns = 100
    size_ratio = 2
    print('running simulations')
    mat, fcpriors, spatialpriors, names = load_priors(size_ratio=size_ratio)

    # we just use functional connectivity priors from WashU team
    # to create ground truth for this simulation
    maxprob_fc = get_maxprob_labels(fcpriors)

    matching_matrix, matching_matrix_z = get_matching_matrix(maxprob_fc)

    results = []

    for noise_level in np.arange(0.05, 0.4, 0.1):
        for run in range(nruns):
            matching_matrix_noisy = create_noisy_matching_matrix(
                matching_matrix_z, noise_level=noise_level
            )
            G = matching_matrix_to_graph(matching_matrix_noisy, density=0.05)
            module_list, module_list_relabeled = run_infomap(G, fcpriors, verbose=False)
            ari = adjusted_rand_score(maxprob_fc, module_list)
            results.append(
                [noise_level, ari]
                + list(get_module_counts(module_list_relabeled, maxprob_fc).values())
            )
            print(f"Noise level: {noise_level}, ARI: {ari}")

    results_df = pd.DataFrame(
        results,
        columns=["noise_level", "ari"]
        + [
            f"size_mod{i}"
            for i in list(get_module_counts(module_list_relabeled, maxprob_fc).keys())
        ],
    )
    results_df.to_csv("results.csv")
