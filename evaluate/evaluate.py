import numpy as np
import tqdm
import argparse
from pathlib import Path

# Read IPD and ILD from .npy files

def read_IPD_ILD_from_npy(npy_path):
    """
    Read IPD and ILD matrices from a .npy file.

    Args:
        npy_path (str): Path to a numeric .npy array with shape (2, time, mel).

    Returns:
        (IPD_mel, ILD_mel): Tuple containing two 2D numpy arrays.
    """
    data = np.load(npy_path, allow_pickle=False)
    if data.ndim != 3 or data.shape[0] != 2:
        raise ValueError(f"Expected numeric IPD/ILD features with shape (2, time, mel): {npy_path}")
    return data[0], data[1]


def calculate_mae(matrix_a, matrix_b):
    """
    Compute Mean Absolute Error (MAE) between two matrices.
    Automatically clips the larger matrix (top-left corner) to align with the smaller one.

    Args:
        matrix_a : array_like
            First input 2D array.
        matrix_b : array_like
            Second input 2D array.

    Returns:
        float: MAE value over the aligned area.
    """
    a = np.asarray(matrix_a)
    b = np.asarray(matrix_b)
    if a.ndim != 2 or b.ndim != 2 or not a.size or not b.size:
        raise ValueError("MAE requires nonempty two-dimensional feature arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("MAE features must contain only finite values")

    min_rows = min(a.shape[0], b.shape[0])
    min_cols = min(a.shape[1], b.shape[1])

    a_trunc = a[:min_rows, :min_cols]
    b_trunc = b[:min_rows, :min_cols]

    return np.mean(np.abs(a_trunc - b_trunc))


def calculate_mae_time(matrix_a, matrix_b, normalization="mean"):
    """Compute elementwise MAE; legacy reproduces the old extra frame division."""
    mae_value = calculate_mae(matrix_a, matrix_b)
    if normalization == "legacy":
        return mae_value / min(np.asarray(matrix_a).shape[0], np.asarray(matrix_b).shape[0])
    if normalization != "mean":
        raise ValueError(f"Unknown MAE normalization: {normalization}")
    return mae_value


def matrix_cosine_similarity(matrix_a, matrix_b, epsilon=1e-8):
    """
    Compute cosine similarity between two matrices (flattened) with phase wrapping support for IPD.

    Args:
        matrix_a : ndarray - Reference matrix (ground truth)
        matrix_b : ndarray - Comparison matrix (prediction)
        epsilon : float - Small constant to prevent division by zero

    Returns:
        float - Cosine similarity score (can be in [-1, 1])
    """
    min_rows = min(matrix_a.shape[0], matrix_b.shape[0])
    min_cols = min(matrix_a.shape[1], matrix_b.shape[1])
    a = matrix_a[:min_rows, :min_cols].flatten()
    b = matrix_b[:min_rows, :min_cols].flatten()

    # Phase wrapping (for IPD) if any values exceed pi (heuristic detection)
    if np.any(a > np.pi):
        a = np.angle(np.exp(1j * a))
        b = np.angle(np.exp(1j * b))

    a_norm = a / (np.linalg.norm(a) + epsilon)
    b_norm = b / (np.linalg.norm(b) + epsilon)

    raw_score = np.dot(a_norm, b_norm)

    return raw_score


def frequency_band_similarity(matrix_a, matrix_b, sr=16000, n_fft=512):
    """
    Compute weighted cosine similarity across coarse frequency bands.

    Args:
        matrix_a : ndarray
        matrix_b : ndarray
        sr : int - Sample rate assumed for frequency axis construction
        n_fft : int - FFT size (used to reconstruct frequency bins)

    Returns:
        float - Weighted average similarity.
    """
    min_rows = min(matrix_a.shape[0], matrix_b.shape[0])
    min_cols = min(matrix_a.shape[1], matrix_b.shape[1])
    a = matrix_a[:min_rows, :min_cols]
    b = matrix_b[:min_rows, :min_cols]

    # Frequency band definitions (approximate auditory bands)
    freq_bands = [
        (0, 500),      # Low
        (500, 2000),   # Mid
        (2000, 20000)  # High
    ]

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    bin_indices = [np.where((freqs >= low) & (freqs < high))[0] for low, high in freq_bands]

    # Band weights (emphasize mid/high frequencies)
    weights = [0.2, 0.5, 0.3]
    total_score = 0
    for idx, w in zip(bin_indices, weights):
        a_band = a[:, idx].flatten()
        b_band = b[:, idx].flatten()

        if np.all(a_band == 0) and np.all(b_band == 0):
            continue

        score = matrix_cosine_similarity(a_band, b_band)
        total_score += w * score

    return total_score / sum(weights)


def paired_files(gt_dir, pred_dir, suffix):
    gt_files = {path.name: path for path in Path(gt_dir).glob(f"*{suffix}")}
    pred_files = {path.name: path for path in Path(pred_dir).glob(f"*{suffix}")}
    if not gt_files or gt_files.keys() != pred_files.keys():
        missing = sorted(gt_files.keys() - pred_files.keys())
        extra = sorted(pred_files.keys() - gt_files.keys())
        raise ValueError(
            f"Expected nonempty matching {suffix} files; missing predictions: {missing}; "
            f"extra predictions: {extra}"
        )
    return [(gt_files[name], pred_files[name]) for name in sorted(gt_files)]


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired spatial audio features.")
    parser.add_argument("--gt-dir", default="./outputs/gt/npy")
    parser.add_argument("--pred-dir", default="./outputs/infer/npy")
    parser.add_argument("--mae-normalization", choices=("mean", "legacy"), default="mean")
    parser.add_argument("--skip-embeddings", action="store_true", help="Evaluate IPD/ILD only.")
    args = parser.parse_args()
    feature_pairs = paired_files(args.gt_dir, args.pred_dir, "_feature.npy")
    dis_pairs = [] if args.skip_embeddings else paired_files(args.gt_dir, args.pred_dir, "_dis.npy")
    azi_pairs = [] if args.skip_embeddings else paired_files(args.gt_dir, args.pred_dir, "_azi.npy")
    if not args.skip_embeddings:
        names = {gt.name[:-len("_feature.npy")] for gt, _ in feature_pairs}
        for pairs, suffix in ((dis_pairs, "_dis.npy"), (azi_pairs, "_azi.npy")):
            if {gt.name[:-len(suffix)] for gt, _ in pairs} != names:
                raise ValueError(f"{suffix} embeddings must cover the same audio items as IPD/ILD features")

    IPD_MAE_LIST = []
    ILD_MAE_LIST = []

    # Compute IPD and ILD MAE
    for gt_npy_file, pred_npy_file in tqdm.tqdm(feature_pairs, desc='evaluate IPD and ILD'):
        gt_IPD, gt_ILD = read_IPD_ILD_from_npy(gt_npy_file)
        pred_IPD, pred_ILD = read_IPD_ILD_from_npy(pred_npy_file)
        IPD_mae = calculate_mae_time(gt_IPD, pred_IPD, args.mae_normalization)
        ILD_mae = calculate_mae_time(gt_ILD, pred_ILD, args.mae_normalization)
        IPD_MAE_LIST.append(IPD_mae)
        ILD_MAE_LIST.append(ILD_mae)

    ipd_mae_loss = round(sum(IPD_MAE_LIST) / len(IPD_MAE_LIST) * 100, 4)
    ild_mae_loss = round(sum(ILD_MAE_LIST) / len(ILD_MAE_LIST) * 100, 4)

    print(f'MAE normalization: {args.mae_normalization}')
    print(f'IPD mae(x100): {ipd_mae_loss}')
    print(f'ILD mae(x100): {ild_mae_loss}')
    if args.skip_embeddings:
        return

    # Compute distance and azimuth cosine similarity
    dis_scores = []
    azi_scores = []

    for gt_dis_npy_file, pred_dis_npy_file in tqdm.tqdm(dis_pairs, desc='evaluate distance cos'):
        gt_dis_tokens = np.load(gt_dis_npy_file, allow_pickle=False)
        pred_dis_tokens = np.load(pred_dis_npy_file, allow_pickle=False)
        dis_score = matrix_cosine_similarity(gt_dis_tokens, pred_dis_tokens)
        dis_scores.append(dis_score)

    for gt_azi_npy_file, pred_azi_npy_file in tqdm.tqdm(azi_pairs, desc='evaluate azimuth cos'):
        gt_azi_tokens = np.load(gt_azi_npy_file, allow_pickle=False)
        pred_azi_tokens = np.load(pred_azi_npy_file, allow_pickle=False)
        azi_score = matrix_cosine_similarity(gt_azi_tokens, pred_azi_tokens)
        azi_scores.append(azi_score)

    dis_cos = np.mean(dis_scores) if dis_scores else float('nan')
    azi_cos = np.mean(azi_scores) if azi_scores else float('nan')

    print(f"dis Cosine Similarity: {dis_cos:.4f} (-1~1 scale)")
    print(f"azi Cosine Similarity: {azi_cos:.4f} (-1~1 scale)")


if __name__ == "__main__":
    main()
