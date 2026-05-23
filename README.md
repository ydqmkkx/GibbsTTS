# GibbsTTS: a zero-shot voice cloning TTS model
GibbsTTS is a zero-shot text-to-speech model based on metric-induced discrete flow matching, incorporating the proposed kinetic-optimal scheduler and finite-step moment correction. \
This is the official PyTorch implementation of the paper: \
Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech \
<a href='https://arxiv.org/abs/2605.09386'><img src='https://img.shields.io/badge/arXiv-2605.09386-red'></a>
<a href='https://ydqmkkx.github.io/GibbsTTSProject/'><img src='https://img.shields.io/badge/Demo-blue'></a>
<a href='https://huggingface.co/spaces/ydqmkkx/GibbsTTS'>
  <img src='https://img.shields.io/badge/🤗%20Interactive%20Demo-Hugging%20Face-yellow'>
</a>

## Overview
1. The released checkpoint was trained on the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset)-EN/ZH, for about 46 hours using 32 NVIDIA H100 GPUs. \
The Japanese LoRA checkpoint was fine-tuned on the Emilia-JA and Emilia-[Yodas](https://huggingface.co/datasets/espnet/yodas)-JA, for about 70 mins using 32 GPUs.

2. The model supports English, Chinese Mandarin, and Japanese (LoRA fine-tuned), also supports cross-lingual synthesis.

3. For greater flexibility and to better fit the ARM-based GPU cluster ([Miyabi](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/system.php)), the model architecture, training framework, and inference pipeline are built from scratch. \
This makes the code easy to modify, but some implementation details, such as variable precision, may require extra care compared with mature training frameworks.

4. For the open-source release, I have simplified the code and reduced the required packages as much as possible.
If you encounter any bugs, please open an issue.

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

## Using WebUI
The WebUI uses the same script as the [interactive demo](https://huggingface.co/spaces/ydqmkkx/GibbsTTS).
```python
python webui.py
```

## Start without WebUI
### Download pre-trained weights
Use ```hf download```
```bash
hf download ydqmkkx/GibbsTTS --local-dir ./pretrained --include "*.safetensors"
```
or download the safetensors via the [huggingface page](https://huggingface.co/ydqmkkx/GibbsTTS/tree/main). \
The codec weights are from [MaskGCT](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec).

### Find more examples in samples.ipynb
```python
from models import GibbsTTS
from config import ModelConfig, LoRAConfig
import soundfile as sf

configs = ModelConfig()
model = GibbsTTS(configs)

# English:
prompt_audio = "./prompt_examples/common_voice_en_188092-common_voice_en_188093.wav"
prompt_text = "This man looked exactly the same, except that now the roles were reversed."
target_text = "He also tried to remember some good stories to relate as he sheared the sheep."
prompt_lang = "en"
target_lang = "en"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, prompt_lang=prompt_lang, target_lang=target_lang)
sf.write(f'en.wav', audio, 24000, 'PCM_24')

# Chinese Mandarin:
prompt_audio = "./prompt_examples/00005476-00000047.wav"
prompt_text = "该委员会的角色，类似新总统的亲友团或后援团。"
target_text = "上发条的弹簧钟发明之前，没有准点时间来确保远程航行的安全。"
prompt_lang = "zh"
target_lang = "zh"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, prompt_lang=prompt_lang, target_lang=target_lang)
sf.write(f'zh.wav', audio, 24000, 'PCM_24')

# Cross-lingual: English to Chinese
prompt_audio = "./prompt_examples/common_voice_en_188092-common_voice_en_188093.wav"
prompt_text = "This man looked exactly the same, except that now the roles were reversed."
target_text = "上发条的弹簧钟发明之前，没有准点时间来确保远程航行的安全。"
prompt_lang = "en"
target_lang = "zh"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, prompt_lang=prompt_lang, target_lang=target_lang)
sf.write(f'en2zh.wav', audio, 24000, 'PCM_24')

# Japanese
configs.lora = LoRAConfig()
model = GibbsTTS(configs)

prompt_audio = "./prompt_examples/prompt_audio_1.wav"
prompt_text = "この料理は家庭で作れます。"
target_text = "アラバマ州の最大都市はバーミングハムである"
prompt_lang = "ja"
target_lang = "ja"

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, prompt_lang=prompt_lang, target_lang=target_lang)
sf.write(f'ja.wav', audio, 24000, 'PCM_24')
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