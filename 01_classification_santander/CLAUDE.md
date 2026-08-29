# 01_classification_santander — 프로젝트 규칙

레포 공통 규칙은 `../CLAUDE.md` 참고.

## 문제

- **과제**: 약 370개 익명화 피처로 은행 고객의 불만족(`TARGET=1`) 여부를 사전 감지하는 이진분류
- **데이터**: `../data/santander-customer-satisfaction/`
- **평가지표**: ROC-AUC

## 접근

- 분산 0 / 상관계수 과다 중복 피처 제거로 차원 축소
- XGBoost / LightGBM / 로지스틱 회귀 비교
- Hyperopt로 트리 깊이·샘플링 비율 등 튜닝

## 구조

- `Santander.ipynb` — 단일 노트북 (전처리~모델링~튜닝 전체)
- `models/` — 튜닝된 최종 모델(`.pkl`) 저장 위치
