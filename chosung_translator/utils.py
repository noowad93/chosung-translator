from typing import List
import magic

CHOSUNG_LIST = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def load_data(file_path: str) -> List[str]:
    texts = []
    encoding_type = _get_file_encoding_type(file_path)
    with open(file_path, "r", encoding=encoding_type) as f:
        for line in f:
            texts.append(line.strip())
    return texts


def convert_text_to_chosung(text: str) -> str:
    chosung_text = ""
    for c in text:
        if ord("가") <= ord(c) <= ord("힣"):
            chosung_text += CHOSUNG_LIST[(ord(c) - ord("가")) // 588]
        else:
            chosung_text += c
    return chosung_text


def _get_file_encoding_type(file_path: str) -> str:
    blob = open(file_path, 'rb').read()
    m = magic.Magic(mime_encoding=True)
    return m.from_buffer(blob)
