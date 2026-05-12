import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_model

from text import cleaned_text_to_sequence
from text.phonemize import phonemize

from .model import GibbsTTS_Model
from amphion_utils import amphion_codec

class AudioResampler:
    def __init__(self, device, target_sr=24000):
        self.device = device
        self.target_sr = target_sr
        self.resamplers = {}

    def __call__(self, wav, sr):
        if sr == self.target_sr:
            return wav

        if sr not in self.resamplers:
            self.resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr
            ).to(self.device)

        return self.resamplers[sr](wav)
    
class GibbsTTS(nn.Module):
    def __init__(self, configs):
        super().__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.configs = configs

        self.model = GibbsTTS_Model(configs)
        load_model(self.model, configs.infer_ckpt_path)
        self.model = self.model.to(device)
        self.model.eval()

        self.codec = amphion_codec(configs, device)

        self.resampler = AudioResampler(device, target_sr=24000)

        self.language_dict = {"en": 0, "zh": 1, "mixed": 2}
        self.space_id = cleaned_text_to_sequence([" "])

    @torch.no_grad()
    def synthesize(self, prompt_audio, prompt_text, target_text, language):
        prompt_phone, _ = phonemize(prompt_text)
        target_phone, _ = phonemize(target_text)
        prompt_phone = cleaned_text_to_sequence(prompt_phone)
        target_phone = self.space_id + cleaned_text_to_sequence(target_phone)
        text = prompt_phone + target_phone
        text = torch.tensor(text, dtype=torch.long, device=self.device).unsqueeze(0)

        prompt_wav, sr = torchaudio.load(prompt_audio)
        prompt_wav = self.resampler(prompt_wav.to(self.device), sr).unsqueeze(0)
        prompt_token = self.codec.encode(prompt_wav)

        ratio = prompt_token.shape[1] / len(prompt_phone)
        if language == "en":
            ratio = max(3.224 * 0.8, min(ratio, 3.224 * 1.25))
        elif language == "zh":
            ratio = max(3.286 * 0.8, min(ratio, 3.286 * 1.25))
        elif language == "mixed":
            ratio = max(3.255 * 0.8, min(ratio, 3.255 * 1.25))
        length = int(len(target_phone) * ratio)

        lang = torch.tensor([self.language_dict[language]], dtype=torch.long, device=self.device)

        outputs = self.model.synthesize(text, lang, length, prompt_token, 
                                   n_timesteps=self.configs.steps, temperature=self.configs.temperature, top_p=self.configs.top_p, rescale_cfg=self.configs.rescale_cfg, cfg=self.configs.cfg)
        codes = outputs["x"].clamp(min=0, max=1023)
    
        audio = self.codec.decode(codes)
        return audio