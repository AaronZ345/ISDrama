#  ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting

#### Yu Zhang*, Wenxiang Guo*, Changhao Pan*, Zhiyuan Zhu*, Tao Jin, Zhou Zhao | Zhejiang University

Dataset and evaluation code of [ISDrama (ACMMM 2025)](https://arxiv.org/abs/2504.20630): Immersive Spatial Drama Generation through Multimodal Prompting.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.20630) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Dataset)](https://huggingface.co/datasets/AaronZ345/MRSDrama)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/ISDrama?style=social)](https://github.com/AaronZ345/ISDrama)

We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts.
Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting.

We provide the **evaluation code** in this repository. 

Moreover, you can visit our [Demo Page](https://aaronz345.github.io/ISDramaDemo) for the audio samples of our dataset as well as the results of our model.

## Updates

- 2025.07: We released the evaluation code of MRSDrama!
- 2025.07: We released the full dataset of MRSDrama!
- 2025.07: ISDrama is accepted by ACMMM 2025!

## TODO List

✅ Release the evaluation code.

✅ Release the full dataset.

🔲 Release the main model code.

## Key Features

- We develop MRSDrama, the first **multimodal recorded spatial drama dataset**, accompanying videos, scripts, alignments, positions, and textual prompts.
- We introduce ISDrama, the first **immersive spatial drama generation model through multimodal prompting**. We design the **Multimodal Pose Encoder** to extract pose from multimodal inputs, while the **Immersive Drama Transformer** produces binaural speech.
- Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics.

### Where to download

Click [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Dataset)](https://huggingface.co/datasets/AaronZ345/MRSDrama) to access our **full dataset** (videos, scripts, alignments, positions, and textual prompts) on Hugging Face **for free**! Hope our data is helpful for your research.

**Please note that, if you are using MRSDrama, it means that you have accepted the terms of [license](https://github.com/AaronZ345/ISDrama/blob/master/dataset_license.md).**

### Data Architecture

Our dataset is organized hierarchically. 

Each top-level folder contains a set of dramas. Each folder contains a subfolder with cut WAV files, an MP4 video file, and a JSON file containing all annotation information.

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
