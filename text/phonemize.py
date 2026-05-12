import re

from text.english import english_to_ipa2
from text.mandarin import chinese_to_cnm3
# from text.japanese import japanese_to_ipa2

ZH_PATTERN = re.compile(r'[\u3400-\u4DBF\u4e00-\u9FFF\uF900-\uFAFF\u3000-\u303F]')
EN_PATTERN = re.compile(r'[a-zA-Z]+')

def detect_language(text: str, prev_lang=None):
    if ZH_PATTERN.search(text): return 'ZH'
    if EN_PATTERN.search(text): return 'EN'
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

    has_en = 'EN' in s
    has_zh = 'ZH' in s

    if has_en and has_zh:
        return 'mixed'
    elif has_en:
        return 'EN'
    elif has_zh:
        return 'ZH'
    else:
        return None

# auto detect language using re
def phonemize(text: str):
    pointer = 0
    output = []
    languages = []
    current_language = detect_language(text[pointer])
    
    while pointer < len(text):
        temp_text = ''
        while pointer < len(text) and detect_language(text[pointer], current_language) == current_language:
            temp_text += text[pointer]
            pointer += 1
        if current_language == 'ZH':
            languages += ['ZH']
            output += chinese_to_cnm3(temp_text)
        # elif current_language == 'JA':
        #     output += japanese_to_ipa2(temp_text)
        elif current_language == 'EN':
            languages += ['EN']
            output += english_to_ipa2(temp_text)
            output += [" "]
        if pointer < len(text):
            current_language = detect_language(text[pointer])

    output = strip_trailing_space(output)
    output = ensure_ending_punc(output)

    return output, language_tag(languages)
