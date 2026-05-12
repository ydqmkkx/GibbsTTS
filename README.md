# GibbsTTS
Official PyTorch implementation of the paper: Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech \
<a href='https://ydqmkkx.github.io/GibbsTTSProject/'><img src='https://img.shields.io/badge/Demo-blue'></a>

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
Use ```hf download```:
```bash
hf download ydqmkkx/GibbsTTS --local-dir ./pretrained --include "*.safetensors"
```
or download the safetensors files via the [huggingface page](https://huggingface.co/ydqmkkx/GibbsTTS/tree/main). \
The codec weights are from [MaskGCT](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec).

## Quick start
```bash
from models import GibbsTTS
from config import ModelConfig
from IPython.display import Audio
import soundfile as sf

configs = ModelConfig()
model = GibbsTTS(configs)

audio = model.synthesize(prompt_audio=prompt_audio, prompt_text=prompt_text, target_text=target_text, language=language)

sf.write(f'target.wav', audio, 24000, 'PCM_24')
Audio(data=audio, rate=24000) 
```