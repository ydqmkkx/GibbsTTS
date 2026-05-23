import re

from text.english import english_to_ipa2
from text.mandarin import chinese_to_cnm3
from text.japanese import japanese_to_ipa2

ZH_PATTERN = re.compile(r'[\u3400-\u4DBF\u4e00-\u9FFF\uF900-\uFAFF\u3000-\u303F]')
EN_PATTERN = re.compile(r'[a-zA-Z]+')

def detect_language(text: str, prev_lang=None):
    if ZH_PATTERN.search(text): return 'zh'
    if EN_PATTERN.search(text): return 'en'
    return prev_lang 

def strip_trailing_space(xs):
    while xs and xs[-1].isspace():
        xs.pop()
    return xs

END_PUNCS = {'.', ',', '!', '?', '-', '…', '~'}
def ensure_ending_punc(xs):
    if not xs:
        return ['.']
    if xs[-1] not in END_PUNCS:
        xs.append('.')
    return xs

def language_tag(tags):
    s = set(tags)

    has_en = 'en' in s
    has_zh = 'zh' in s

    if has_en and has_zh:
        return 'mixed'
    elif has_en:
        return 'en'
    elif has_zh:
        return 'zh'
    else:
        return None

# auto detect language using re
def phonemize(text, lang=None):
    if lang == "en":
        output = english_to_ipa2(text)
    elif lang == "zh":
        output = chinese_to_cnm3(text)
    elif lang == "ja":
        output = japanese_to_ipa2(text)
    else:
        # auto detection for en/zh
        pointer = 0
        output = []
        languages = []
        current_language = detect_language(text[pointer])
        
        while pointer < len(text):
            temp_text = ''
            while pointer < len(text) and detect_language(text[pointer], current_language) == current_language:
                temp_text += text[pointer]
                pointer += 1
            if current_language == 'zh':
                languages += ['zh']
                output += chinese_to_cnm3(temp_text)
            elif current_language == 'en':
                languages += ['en']
                output += english_to_ipa2(temp_text)
                output += [" "]
            if pointer < len(text):
                current_language = detect_language(text[pointer])
            
        lang = language_tag(languages)

    output = strip_trailing_space(output)
    output = ensure_ending_punc(output)

    return output, lang
