# 초성 해석기
## 개요
한국어 초성만으로 이루어진 문장을 입력하면, 완성된 문장을 예측하는 초성 해석기입니다.
```text
초성: ㄴㄴ ㄴㄹ ㅈㅇㅎ
예측 문장: 나는 너를 좋아해
```
## 모델
모델은 SKT-AI에서 공개한 [Ko-BART](https://github.com/SKT-AI/KoBART)를 이용합니다.
## 데이터
문장 단위로 이루어진 아무 코퍼스나 사용가능합니다. 단, 모델의 추론 성능은 데이터의 도메인이나 데이터의 양에 크게 의존하기 때문에 원하는 모델 성능에 맞는 코퍼스를 사용해주세요.
`./data` 디렉토리에 더미 데이터셋을 추가해두었으니, 더미 데이터셋과 동일한 형식의 코퍼스를 준비해두시면 됩니다.
## 학습

```sh
python run_train.py
```

## 추론
```sh
python run_inference.py --finetuned-model-path $FINETUNED_MODEL_PATH
```

## Notes
- 본 레포는 별도의 학습 데이터를 포함하고 있지 않습니다.
- 본 레포의 라이센스는 [Ko-BART](https://github.com/SKT-AI/KoBART)의 `modified-MIT` 라이센스를 따릅니다.

## Todo
- 테스트 코드 추가
