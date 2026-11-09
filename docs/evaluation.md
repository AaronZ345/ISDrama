[Back to ISDrama](../README.md)

# Evaluate paired spatial audio

> The evaluation process is based on the code and models of "BAT: Learning to Reason about Spatial Sounds with Large Language Models" .

### Dependencies

A suitable [conda](https://conda.io/) environment named `isdrama_eva` can be created
and activated with:

```bash
conda env create -f environment.yml
conda activate isdrama_eva
bash timm_patch/patch.sh
```

## Preparation

### Checkpoint Preparation

Please download the finetuned `BAT` encoder [checkpoint](https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth) and place it at:

```bash
./ckpt/finetuned.pth
```
Make sure the path exists (create the `ckpt` directory if necessary).
The evaluator verifies the published checkpoint's SHA-256 before deserializing a private copy. Custom or modified checkpoint files are rejected.

### Data Preparation

For evaluation, you must prepare paired ground‑truth audio and generated audio.
Place them respectively in:

```bash
./data/gt
./data/infer
```
The expected directory layout is:

```yaml
.
├── gt
│   ├── 0000.wav
│   ├── 0001.wav
│   ├── 0002.wav
│   └── 0003.wav
└── infer
    ├── 0000.wav
    ├── 0001.wav
    ├── 0002.wav
    └── 0003.wav
```

Important:

- The files inside gt and infer must correspond one‑to‑one.
- Filenames and counts must match exactly (e.g., `gt/0002.wav` pairs with `infer/0002.wav`).
- Ensure sampling rates and channel configurations are consistent if required by downstream metrics.

## Metrics

### Semantic and acoustic metrics (separate tools)

The supplied runner reports the spatial metrics below. CER, speaker similarity, and FFE require separate recognizer, speaker-embedding, and pitch-evaluation tools.
- **Character Error Rate (CER)**: Assesses transcript/content accuracy.
- **Cosine Similarity (SIM)**: Measures speaker timbre similarity between the generated audio and the prompt/reference audio (e.g., via speaker embeddings).
- **F0 Frame Error (FFE)**: Evaluates prosody fidelity by comparing voiced/unvoiced decisions and pitch (F0) frames.

### Spatial Metrics
- **IPD MAE**: Mean Absolute Error between ground‑truth and generated Interaural Phase Differences.
- **ILD MAE**: Mean Absolute Error between ground‑truth and generated Interaural Level Differences.
- **Angle Cosine Similarity (ANG Cos)**: Cosine similarity between ground‑truth and generated direction (azimuth / elevation) angle embeddings.
- **Distance Cosine Similarity (Dis Cos)**: Cosine similarity between ground‑truth and generated distance embeddings.

> **Note:** Cosine‑based spatial scores are in the range \[-1, 1\], with higher values indicating closer alignment of spatial embeddings.

## Running the Evaluation

Run the following script to perform the evaluation pipeline:

```bash
bash evaluate/eval.sh --gt-dir data/gt --pred-dir data/infer \
  --checkpoint ckpt/finetuned.pth --output-dir outputs/run-1
```

The script evaluate/eval.sh executes the following three stages:

1. Extract angle and distance embeddings using the BAT encoder.

2. Extract IPD & ILD features from paired ground‑truth and generated stereo audio.

3. Compute metrics: MAE (for IPD / ILD) and cosine similarities (for angle and distance).

The runner requires matching, nonempty WAV filename sets and an output directory that does not already exist. It reserves that directory before extraction and removes its partial output if a stage fails. Use `--device cpu` for CPU embedding extraction, or set `CUDA_VISIBLE_DEVICES` before the command to select an existing GPU. The BAT preprocessing profile is fixed by the encoder; it does not select or load a training dataset.

To calculate IPD and ILD without the BAT checkpoint:

```bash
bash evaluate/eval.sh --gt-dir data/gt --pred-dir data/infer \
  --output-dir outputs/ipd-ild-run-1 --skip-embeddings
```

The default `--mae-normalization mean` computes the mean absolute error once over aligned time-frequency elements, then multiplies the printed score by 100. It corrects the previous extra division by frame count. Scores from this version should be labeled `mean`; use `--mae-normalization legacy` to reproduce the former normalization when comparing historical runs. Both modes retain the existing feature extraction and alignment convention.

Existing features can be evaluated independently:

New feature files contain numeric arrays with shape `(2, time, mel)`, ordered as IPD then ILD. Regenerate older pickled dictionary feature files with the updated extractor; object arrays are no longer loaded. All requested metrics must cover the same audio items.

```bash
python evaluate/evaluate.py --gt-dir outputs/run-1/gt/npy \
  --pred-dir outputs/run-1/infer/npy --mae-normalization mean
```
