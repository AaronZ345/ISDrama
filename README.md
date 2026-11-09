#  ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting

#### Yu Zhang*, Wenxiang Guo*, Changhao Pan*, Zhiyuan Zhu*, Tao Jin, Zhou Zhao | Zhejiang University

Dataset and evaluation code of [ISDrama (ACM-MM 2025)](https://arxiv.org/abs/2504.20630): Immersive Spatial Drama Generation through Multimodal Prompting.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.20630) 
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/ISDramaDemo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Dataset)](https://huggingface.co/datasets/AaronZ345/MRSDrama)
[![weixin](https://img.shields.io/badge/-WeChat@语音之家-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/jrKT16zH115ZDYxa0b492w)
[![weixin](https://img.shields.io/badge/-WeChat@PaperWeekly-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/ZIC8rHDkwsKwcBwEi4v4DQ)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/1930597017840779306)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Link-blue?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1ALpnOD4b_zgn8GSu4aS-cjdDK8vY6Lyy?usp=sharing)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/ISDrama?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/ISDrama)

We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts.
Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting.

We provide the **evaluation code** in this repository. 

Moreover, you can visit our [Demo Page](https://aaronz345.github.io/ISDramaDemo) for the audio samples of our dataset as well as the results of our model.

## Updates

- 2025.07: We released the evaluation code of MRSDrama!
- 2025.07: We released the full dataset of MRSDrama!
- 2025.07: ISDrama is accepted by ACMMM 2025!

## Key Features

- We develop MRSDrama, the first **multimodal recorded spatial drama dataset**, accompanying videos, scripts, alignments, positions, and textual prompts.
- We introduce ISDrama, the first **immersive spatial drama generation model through multimodal prompting**. We design the **Multimodal Pose Encoder** to extract pose from multimodal inputs, while the **Immersive Drama Transformer** produces binaural speech.
- Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics.

## Training datasets

MRSDrama and MRSAudio are both publicly released training data resources for spatial audio generation and understanding:

| Dataset | Training data | Download |
| --- | --- | --- |
| MRSDrama | Recorded binaural drama, scripts, video, geometric poses, and textual prompts | [Hugging Face](https://huggingface.co/datasets/AaronZ345/MRSDrama) |
| MRSAudio | MRSLife, MRSSpeech, MRSSing, and MRSMusic, with spatial recordings and multimodal annotations | [Hugging Face](https://huggingface.co/datasets/MRSAudio/MRSAudio) · [Project](https://mrsaudio.github.io) |

Prepare the selected dataset for your model's training pipeline and follow its own data card and license. This repository contains the ISDrama evaluation tools; the paired audio directories used by those tools are described separately below.

### Where to download

Click [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Dataset)](https://huggingface.co/datasets/AaronZ345/MRSDrama) to access our **full dataset** (videos, scripts, alignments, positions, and textual prompts) on Hugging Face **for free**! Hope our data is helpful for your research.

Besides, we also provide our dataset on [![Google Drive](https://img.shields.io/badge/Google%20Drive-Link-blue?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1ALpnOD4b_zgn8GSu4aS-cjdDK8vY6Lyy?usp=sharing).

**Please note that, if you are using MRSDrama, it means that you have accepted the terms of [license](https://github.com/AaronZ345/ISDrama/blob/master/dataset_license.md).**

### Data Architecture

Our dataset is organized hierarchically. 

Each top-level folder contains a set of dramas. Each folder contains a subfolder with cut WAV files, an MP4 video file, and a JSON file containing all annotation information. 
Additionally, the geometric_pose subdirectory stores NumPy (.npy) sequences—listener‑centric 3D positions, head-orientation quaternions, and radial velocities with respect to the left and right ears. These sequences are aligned at the frame level and generated with a 48 kHz sample rate and a 256-sample hop size.

## Evaluation of ISDrama

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

### Semantic & Acoustic Metrics
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

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{zhang2025isdrama,
  title={ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting},
  author={Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Zhu, Zhiyuan and Jin, Tao and Zhao, Zhou},
  journal={arXiv preprint arXiv:2504.20630},
  year={2025}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/ISDrama)
