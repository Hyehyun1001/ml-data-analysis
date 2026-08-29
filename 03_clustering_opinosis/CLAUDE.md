# 03_clustering_opinosis — 프로젝트 규칙

레포 공통 규칙은 `../CLAUDE.md` 참고.

## 문제

- **과제**: IT 기기/자동차 리뷰 텍스트(UCI Opinosis)에서 라벨 없이 의미 있는 그룹을 발견하는 비지도 군집화
- **데이터**: `../data/topics/` (또는 노트북 내 경로 참고)
- **평가지표**: Silhouette Score

## 접근

- 불용어 제거 + 형태소 분석 후 TF-IDF / 임베딩 벡터화
- K-Means, DBSCAN 적용해 최적 클러스터 수 도출 및 핵심 키워드 추출

## 구조

- `Opinosis_Preprocessing.ipynb` — 단일 노트북
