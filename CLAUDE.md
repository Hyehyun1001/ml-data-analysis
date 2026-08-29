# ml-data-analysis — 공통 규칙

4개 ML 프로젝트(분류 2 · 군집 1 · 회귀 1)를 각자 폴더에서 독립적으로 진행하는 팀 레포.
프로젝트별 세부 규칙/현황은 각 폴더의 `CLAUDE.md`(및 있다면 `HANDOFF.md`) 참고.

## 구조

| 폴더 | 내용 |
|---|---|
| `01_classification_santander/` | 은행 고객 불만족 예측 (이진분류) |
| `02_classification_credit_card/` | 신용카드 이상거래 탐지 (극불균형 이진분류) |
| `03_clustering_opinosis/` | 리뷰 텍스트 군집화 (비지도) |
| `04_regression_mercari/` | 중고거래 가격 예측 (회귀) |
| `data/` | 원본 데이터. **git 추적 안 됨**(`.gitignore`) — 각자 로컬에 받아서 채워야 함 |
| `reports/meetings/` | 팀 회의록 |

## 문서 작성 원칙

- `CLAUDE.md` / `PIPELINE*.md` / `HANDOFF.md` 등은 **담백하고 solid하게** 쓴다 — 필요한 정보만, 장황한 서술 지양.
- 깊게 쓸 필요가 있는 부분은 그때그때 명시적으로 요청받았을 때만 상세히.
- 노트북 1개 : 설계 문서 1개 대응이 원칙 — 한 문서가 여러 노트북 내용을 섞어 다루지 않는다.

## 공통 관례

- 팀원별 노트북은 `이름_프로젝트명.ipynb` 형식 (폴더 구조는 프로젝트마다 다름, 하위 `CLAUDE.md` 참고).
- git push는 명시적으로 요청받았을 때만 수행. 로컬 커밋과 push는 별개로 취급.
