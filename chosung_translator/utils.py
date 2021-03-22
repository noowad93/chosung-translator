from typing import List

CHOSUNG_LIST = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def load_data(file_path: str) -> List[str]:
    texts = []
    with open(file_path, "r") as f:
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
