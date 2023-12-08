#  DeepFM 라이브러리를 이용하여 영화 추천 모델 만들기

* 영화 데이터 : https://grouplens.org/datasets/movielens/

![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-blue.svg)
![TensorFlow 2.12.0](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)

---

|column name|설명|
|---	|---	|
|userId|사용자ID (총 1000명)|
|title|영화제목 (총 100개)|
|genres|영화 장르|
|tag|영화 태그|
|rating|영화 평점|
|target|영화를 봤는지 안봤는지 (0=안봄, 1=영화를 봄)|


## 결과
```python
test LogLoss 0.3888
test AUC 0.8765
```
