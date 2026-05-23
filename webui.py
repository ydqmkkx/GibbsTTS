"""
GibbsTTS Gradio Space — zero-shot voice cloning TTS.

Paper: Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced
Discrete Flow Matching in Zero-Shot Text-to-Speech.
"""
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

PRETRAINED_DIR = REPO_ROOT / "pretrained"
HF_REPO_ID = "ydqmkkx/GibbsTTS"
ASR_MODEL_ID = "openai/whisper-large-v3-turbo"

# MaskGCT codec decoder has a shared buffer (model.head.istft.window).
# Newer safetensors.load_model refuses such state dicts. Patch with a tolerant loader.
import safetensors.torch as _st_torch


def _tolerant_load_model(model, filename, strict=False, device="cpu"):
    state = _st_torch.load_file(filename, device=device)
    return model.load_state_dict(state, strict=False)


_st_torch.load_model = _tolerant_load_model

try:
    import spaces  # ZeroGPU runtime
except ImportError:
    class _DummySpaces:
        def GPU(self, *args, **kwargs):
            def _deco(fn):
                return fn
            return _deco

    spaces = _DummySpaces()


def ensure_weights():
    """Download safetensors from the GibbsTTS HF repo if missing."""
    needed = [
        "GibbsTTS_large_ema.safetensors",
        "GibbsTTS_lora_ja.safetensors",
        "MaskGCT_codec_encoder.safetensors",
        "MaskGCT_codec_decoder.safetensors",
    ]
    missing = [n for n in needed if not (PRETRAINED_DIR / n).exists()]
    if not missing:
        return

    print(f"Downloading missing weights from {HF_REPO_ID}: {missing}")

    from huggingface_hub import hf_hub_download

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    for fname in missing:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            local_dir=str(PRETRAINED_DIR),
        )


ensure_weights()

from config import ModelConfig, LoRAConfig
from models import GibbsTTS_webui

_model = None
_asr = None


def get_model(cfg=None):
    """Lazy singleton — instantiated on first call."""
    global _model
    if _model is None:
        configs = cfg or ModelConfig()
        _model = GibbsTTS_webui(configs)
    return _model


def get_asr():
    """Lazy Whisper pipeline — instantiated on first transcription call."""
    global _asr
    if _asr is None:
        from transformers import pipeline

        cuda_ok = torch.cuda.is_available()
        _asr = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            torch_dtype=torch.float16 if cuda_ok else torch.float32,
            device=0 if cuda_ok else -1,
        )
    return _asr


# TTS language choices.
LANG_LABEL_TO_KEY = {
    "English": "en",
    "Chinese (Mandarin)": "zh",
    "Mixed English/Chinese": "mixed",
    "Japanese": "ja"
}

# ASR language hints for Whisper.
# "None" means no language hint is passed to Whisper.
ASR_LABEL_TO_WHISPER = {
    "None": None,
    "English": "english",
    "Chinese": "chinese",
    "Japanese": "japanese"
}


def _load_audio_16k_mono(path):
    """Read audio at 16 kHz mono via soundfile + torch resampler. Avoids ffmpeg."""
    import soundfile as sf_local
    import torchaudio

    wav, sr = sf_local.read(path, dtype="float32", always_2d=False)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != 16000:
        t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(
            t,
            orig_freq=sr,
            new_freq=16000,
        ).squeeze(0).numpy()

    return wav.astype(np.float32)


@spaces.GPU(duration=60)
def transcribe(prompt_audio, asr_language):
    if not prompt_audio:
        raise gr.Error("Please upload a reference audio clip first.")

    asr = get_asr()

    gen_kwargs = {"task": "transcribe"}
    lang_hint = ASR_LABEL_TO_WHISPER.get(asr_language)

    if lang_hint is not None:
        gen_kwargs["language"] = lang_hint

    audio_np = _load_audio_16k_mono(prompt_audio)

    out = asr(
        {"raw": audio_np, "sampling_rate": 16000},
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=False,
        generate_kwargs=gen_kwargs,
    )

    text = (out.get("text") or "").strip()

    if not text:
        raise gr.Error(
            "Whisper returned empty text. Please type the reference transcript manually."
        )

    return text


DEFAULT_PROMPT_AUDIO_EN = str(
    REPO_ROOT / "prompt_examples" / "common_voice_en_188092-common_voice_en_188093.wav"
)
DEFAULT_PROMPT_TEXT_EN = (
    "This man looked exactly the same, except that now the roles were reversed."
)

DEFAULT_PROMPT_AUDIO_ZH = str(
    REPO_ROOT / "prompt_examples" / "00005476-00000047.wav"
)
DEFAULT_PROMPT_TEXT_ZH = "该委员会的角色，类似新总统的亲友团或后援团。"

DEFAULT_PROMPT_AUDIO_JA = str(
    REPO_ROOT / "prompt_examples" / "prompt_audio_1.wav"
)
DEFAULT_PROMPT_TEXT_JA = (
    "この料理は家庭で作れます。"
)


@spaces.GPU(duration=120)
def synthesize(
    prompt_audio,
    prompt_text,
    target_text,
    reference_language,
    target_language,
    asr_language,
    steps,
    cfg,
    rescale_cfg,
    temperature,
    top_p,
    speed,
    seed,
):
    if not prompt_audio:
        raise gr.Error("Please provide a reference audio clip.")

    if not (target_text or "").strip():
        raise gr.Error("Please provide the text you want to synthesize.")

    used_text = (prompt_text or "").strip()

    if not used_text:
        # Fall back to Whisper auto-transcription.
        asr = get_asr()

        gen_kwargs = {"task": "transcribe"}
        lang_hint = ASR_LABEL_TO_WHISPER.get(asr_language)

        if lang_hint is not None:
            gen_kwargs["language"] = lang_hint

        audio_np = _load_audio_16k_mono(prompt_audio)

        out = asr(
            {"raw": audio_np, "sampling_rate": 16000},
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=False,
            generate_kwargs=gen_kwargs,
        )

        used_text = (out.get("text") or "").strip()

        if not used_text:
            raise gr.Error(
                "Whisper returned empty text. Please type the reference transcript manually."
            )

    prompt_lang_key = LANG_LABEL_TO_KEY[reference_language]
    target_lang_key = LANG_LABEL_TO_KEY[target_language]

    cfg_obj = ModelConfig()
    cfg_obj.lora = LoRAConfig()
    cfg_obj.steps = int(steps)
    cfg_obj.cfg = float(cfg)
    cfg_obj.rescale_cfg = float(rescale_cfg)
    cfg_obj.temperature = float(temperature)
    cfg_obj.top_p = float(top_p)

    model = get_model(cfg_obj)

    # Update inference-time knobs in place.
    model.configs.steps = cfg_obj.steps
    model.configs.cfg = cfg_obj.cfg
    model.configs.rescale_cfg = cfg_obj.rescale_cfg
    model.configs.temperature = cfg_obj.temperature
    model.configs.top_p = cfg_obj.top_p

    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    audio = model.synthesize(
        prompt_audio=prompt_audio,
        prompt_text=used_text,
        target_text=target_text,
        prompt_lang=prompt_lang_key,
        target_lang=target_lang_key,
        speed=float(speed)
    )

    audio = np.asarray(audio, dtype=np.float32)

    return (24000, audio), used_text


CSS = """
.gradio-container { max-width: 1100px !important; }
footer { visibility: hidden; }
"""

INTRO_MD = """
# 🎙️ GibbsTTS — Zero-Shot Voice Cloning TTS

Official interactive demo for **Kinetic-Optimal Scheduling with Moment Correction for
Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech**.

- Paper: <https://arxiv.org/abs/2605.09386>
- Code: <https://github.com/ydqmkkx/GibbsTTS>
- Weights: <https://huggingface.co/ydqmkkx/GibbsTTS>

Upload a short reference speech audio (a few seconds is enough). \\
The reference transcript is optional. Leave it blank, choose ASR language, (and click the **Auto-transcribe** button,) Whisper will transcribe automatically. \\
Then type the text you want to synthesize, choose reference and TTS languages, and click the **Synthesize** button, the model
will speak it in the reference voice. \\
Supports **English**, **Chinese Mandarin**, **English/Chinese mixing**, and **Japanese** (LoRA fine-tuned). \\
Also supports cross-lingual synthesis.
"""


with gr.Blocks(title="GibbsTTS Demo", css=CSS) as demo:
    gr.Markdown(INTRO_MD)

    with gr.Row():
        with gr.Column():
            prompt_audio = gr.Audio(
                label="Reference audio (prompt)",
                type="filepath",
                value=DEFAULT_PROMPT_AUDIO_EN,
            )

            with gr.Group():
                prompt_text = gr.Textbox(
                    label="Reference transcript (optional)",
                    info=(
                        "What the reference clip says. Leave blank to "
                        "auto-transcribe with Whisper."
                    ),
                    lines=2,
                    value=DEFAULT_PROMPT_TEXT_EN,
                    placeholder="(leave blank to auto-transcribe)",
                )

                asr_language = gr.Radio(
                    choices=list(ASR_LABEL_TO_WHISPER.keys()),
                    value="None",
                    label="ASR language",
                    info=(
                        "Language hint for Whisper. Choose None to use "
                        "auto-detection."
                    ),
                )

                transcribe_btn = gr.Button(
                    "Auto-transcribe",
                    size="sm",
                    variant="secondary",
                )

            target_text = gr.Textbox(
                label="Target text (what you want the model to speak)",
                lines=3,
                value=(
                    "He also tried to remember some good stories to relate "
                    "as he sheared the sheep."
                ),
            )

            with gr.Row():
                reference_language = gr.Radio(
                    choices=list(LANG_LABEL_TO_KEY.keys()),
                    value="English",
                    label="Reference language",
                    info="Language of the reference audio/transcript.",
                )

                target_language = gr.Radio(
                    choices=list(LANG_LABEL_TO_KEY.keys()),
                    value="English",
                    label="TTS language",
                    info="Language used by GibbsTTS for synthesis.",
                )

            with gr.Accordion("Advanced settings", open=False):
                steps = gr.Slider(
                    16,
                    64,
                    value=32,
                    step=1,
                    label="Sampling steps",
                )

                cfg = gr.Slider(
                    1.0,
                    5.0,
                    value=2.5,
                    step=0.1,
                    label="Classifier-free guidance scale",
                )

                rescale_cfg = gr.Slider(
                    0.0,
                    1.0,
                    value=0.75,
                    step=0.05,
                    label="CFG rescale",
                )

                temperature = gr.Slider(
                    0.1,
                    1.0,
                    value=0.6,
                    step=0.05,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    0.5,
                    1.0,
                    value=1.0,
                    step=0.01,
                    label="Top-p",
                )

                speed = gr.Slider(
                    0.5,
                    2.0,
                    value=1.0,
                    step=0.1,
                    label="Speed",
                    info="Speech speed: 1.0 = original, 1.5 = 1.5× faster.",
                )

                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed",
                    info="Random seed. -1 for non-fixed seed."
                )

            go = gr.Button("Synthesize", variant="primary")

        with gr.Column():
            out_audio = gr.Audio(
                label="Synthesized speech",
                type="numpy",
            )

    go.click(
        synthesize,
        inputs=[
            prompt_audio,
            prompt_text,
            target_text,
            reference_language,
            target_language,
            asr_language,
            steps,
            cfg,
            rescale_cfg,
            temperature,
            top_p,
            speed,
            seed,
        ],
        outputs=[out_audio, prompt_text],
    )

    transcribe_btn.click(
        transcribe,
        inputs=[prompt_audio, asr_language],
        outputs=prompt_text,
    )

    gr.Examples(
        label="Examples",
        examples=[
            [
                DEFAULT_PROMPT_AUDIO_EN,
                DEFAULT_PROMPT_TEXT_EN,
                "He also tried to remember some good stories to relate as he sheared the sheep.",
                "English",
                "English",
                "English",
            ],
            [
                DEFAULT_PROMPT_AUDIO_ZH,
                DEFAULT_PROMPT_TEXT_ZH,
                "上发条的弹簧钟发明之前，没有准点时间来确保远程航行的安全。",
                "Chinese",
                "Chinese (Mandarin)",
                "Chinese (Mandarin)",
            ],
            [
                DEFAULT_PROMPT_AUDIO_EN,
                DEFAULT_PROMPT_TEXT_EN,
                "我做完这个pre，你帮我download点document。O不OK？",
                "English",
                "English",
                "Mixed English/Chinese",
            ],
            [
                DEFAULT_PROMPT_AUDIO_JA,
                DEFAULT_PROMPT_TEXT_JA,
                "アラバマ州の最大都市はバーミングハムである",
                "Japanese",
                "Japanese",
                "Japanese",
            ],
        ],
        inputs=[
            prompt_audio,
            prompt_text,
            target_text,
            asr_language,
            reference_language,
            target_language,
        ],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        show_error=True,
    )