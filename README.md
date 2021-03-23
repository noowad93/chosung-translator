# 초성 해석기

## 개요
한국어 초성만으로 이루어진 문장을 입력하면, 완성된 문장을 예측하는 초성 해석기입니다.

## 모델
모델은 SKT-AI에서 공개한 [Ko-BART](https://github.com/SKT-AI/KoBART)를 이용합니다.

## 데이터
크롤링한 드라마 대사 코퍼스를 이용합니다.

## 학습

```sh
python run_train.py
```

## 추론
```sh
python run_inference.py --finetuned-model-path $FINETUNED_MODEL_PATH
```
