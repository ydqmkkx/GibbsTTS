# GibbsTTS
Official PyTorch implementation of the paper: \
Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech \
<a href='https://arxiv.org/abs/2605.09386'><img src='https://img.shields.io/badge/arXiv-red'></a>
<a href='https://ydqmkkx.github.io/GibbsTTSProject/'><img src='https://img.shields.io/badge/Demo-blue'></a> \
(Updating)

## Overview
1. GibbsTTS is a zero-shot text-to-speech model based on metric-induced discrete flow matching, incorporating the proposed kinetic-optimal scheduler and finite-step CTMC correction. \
The released checkpoint was trained for about 46 hours on 32 NVIDIA H100 GPUs.

2. For greater flexibility and to better fit the ARM-based GPU cluster ([Miyabi](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/system.php)), the model architecture, training framework, and inference pipeline are built from scratch. \
This makes the code easy to modify, but some implementation details, such as variable precision, may require extra care compared with mature training frameworks.

3. For the open-source release, I have simplified the code and reduced the required packages as much as possible.
If you encounter any bugs, please open an issue.

4. The model was trained on the [Emilia-en/zh](https://huggingface.co/datasets/amphion/Emilia-Dataset), so it only supports English and Chinese Mandarin now. \
A Japanese fine-tuned version will be trained these days.

5. Data cleaning scripts and training examples will also be released gradually.

Contact: I am currently seeking full-time positions in speech, audio, or multimodal generative modeling.
If you are interested in this work, please feel free to visit my [homepage](https://ydqmkkx.github.io/) for more information.


## Environment and installation
1. Python >= 3.9, Python 3.10 and 3.12 are recommended.
2. Install PyTorch and torchaudio following the official PyTorch installation instructions.
3. 
```bash
git clone https://github.com/ydqmkkx/GibbsTTS
cd GibbsTTS
pip install -r requirements.txt
```

## Download pre-trained weights
Use ```hf download```
```bash
hf download ydqmkkx/GibbsTTS --local-dir ./pretrained --include "*.safetensors"
```
or download the safetensors via the [huggingface page](https://huggingface.co/ydqmkkx/GibbsTTS/tree/main). \
The codec weights are from [MaskGCT](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec).

## Quick start
```python
from models import GibbsTTS
from config import ModelConfig
from IPython.display import Audio
import soundfile as sf

configs = ModelConfig()
model = GibbsTTS(configs)
```
English:
```python
prompt_audio = "./prompt_examples/common_voice_en_188092-common_voice_en_188093.wav"
prompt_text = "This man looked exactly the same, except that now the roles were reversed."
target_text = "He also tried to remember some good stories to relate as he sheared the sheep."
language = "en"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, language=language)

sf.write('target.wav', audio, 24000, 'PCM_24')
Audio(data=audio, rate=24000) 

```

Chinese Mandarin:
```python
prompt_audio = "./prompt_examples/00005476-00000047.wav"
prompt_text = "该委员会的角色，类似新总统的亲友团或后援团。"
target_text = "上发条的弹簧钟发明之前，没有准点时间来确保远程航行的安全。"
language = "zh"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, language=language)

sf.write('target.wav', audio, 24000, 'PCM_24')
Audio(data=audio, rate=24000) 
```

A very small proportion of the training data contains mixed English-Chinese sentences, which are labeled as `mixed`. \
Under this setting, the model can synthesize mixed English-Chinese sentences, and cross-lingual synthesis is also handled in the same way. \
Since such data is rare in the training set, this functionality has not been specifically optimized or evaluated. It is provided for experimentation and fun.
```python
prompt_audio = "./prompt_examples/common_voice_en_188092-common_voice_en_188093.wav"
prompt_text = "This man looked exactly the same, except that now the roles were reversed."
target_text = "我做完这个pre，你帮我download点document。O不OK？"
language = "mixed"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, language=language)

sf.write('target.wav', audio, 24000, 'PCM_24')
Audio(data=audio, rate=24000) 
```

## Kinetic-optimal scheduler
The proposed numerical kinetic-optimal scheduler is demonstrated in `Num_KO.ipynb`. \
It is easy to use and can produce the results with a single run. The computed results may show slight numerical differences across devices.

## Moment correction
The proposed solver with finite-step moment correction is shown in `models.model`.
The standard first-order CTMC solver is also provided.

## Acknowledgements
1. We improve and use the text frontend of [StableTTS](https://github.com/KdaiP/StableTTS).
2. For the implementation of masked discrete flow matching, we refer to its [implementation](https://github.com/RobinKa/discrete-flow-matching-pytorch) and [DiFlow-TTS](https://github.com/ishine/DiFlow-TTS).
3. For the implementation of masked discrete diffusion, we refer to [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct).


## Citation
```bibtex
This work:
@article{GibbsTTS,
 author    = {Dong Yang and Yiyi Cai and Haoyu Zhang and Yuki Saito and Hiroshi Saruwatari},
 title     = {Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech},
 year      = {2026},
 journal   = {arXiv preprint arXiv:2605.09386},
}

Codec we use:
@inproceedings{MaskGCT,
 author    = {Yuancheng Wang and Haoyue Zhan and Liwei Liu and Ruihong Zeng and Haotian Guo and Jiachen Zheng and Qiang Zhang and Xueyao Zhang and Shunsi Zhang and Zhizheng Wu},
 title     = {{MaskGCT}: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
 year      = {2025},
 booktitle = {International Conference on Learning Representations (ICLR)},
}

Metric-induced discrete flow matching:
@inproceedings{MI-DFM,
 author    = {Neta Shaul and Itai Gat and Marton Havasi and Daniel Severo and Anuroop Sriram and Peter Holderrieth and Brian Karrer and Yaron Lipman and Ricky T. Q. Chen},
 title     = {Flow Matching with General Discrete Paths: {A} Kinetic-Optimal Perspective},
 year      = {2025},
 booktitle = {International Conference on Learning Representations (ICLR)},
}
```