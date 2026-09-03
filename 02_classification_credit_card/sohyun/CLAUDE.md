# 02_classification_credit_card — 프로젝트 규칙

레포 공통 규칙은 `../CLAUDE.md` 참고.

## 문제

- **과제**: 28개 PCA 변수 + 거래정보로 극소수(약 0.17%) 사기 거래를 포착하는 극불균형 이진분류
- **데이터**: `../data/creditcard.csv`
- **평가지표**: Precision-Recall AUC / F1-Score (Accuracy는 착시가 크므로 배제)

## 접근

- SMOTE / ADASYN 오버샘플링, 언더샘플링 비교
- Recall(탐지율) 중심 개선

## 구조

- `CreditCard.ipynb` — 단일 노트북
- `models/` — 최종 모델(`.pkl`) 저장 위치
