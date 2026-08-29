# 04_regression_mercari — 프로젝트 규칙

레포 공통 규칙은 `../CLAUDE.md` 참고. 현재 작업 인수인계는 `HANDOFF.md` 참고.

## 문제

- **과제**: 상품명/카테고리/브랜드/설명글/상태/배송비로 적정 판매가(`price`) 예측 (회귀)
- **데이터**: `../data/mercari_train.tsv` (tab-separated)
- **평가지표**: RMSLE — 타깃을 `log1p` 변환해두고 일반 RMSE 최소화로 대체

## 노트북·문서 매핑 (중요)

`teamipynb/`에는 팀원별 개인 노트북이 각자 있지만, **정식 파이프라인 문서화 대상은 소현 노트북뿐**이며 base/feature 두 갈래로 분리되어 있다:

| 노트북 | 대응 문서 | 역할 |
|---|---|---|
| `teamipynb/소현_mercari_base.ipynb` | `docs/pipelines/PIPELINE_base.md` | 원본 피처 전처리/인코딩/벡터화 + 베이스라인 모델링 |
| `teamipynb/소현_mercari_feature.ipynb` | `docs/pipelines/PIPELINE_feature.md` | base 위에 새 파생피처 추가 실험 |

**절대 서로 섞지 않는다**: base 작업 내용은 `PIPELINE_base.md`에만, feature 작업 내용은 `PIPELINE_feature.md`에만 기록.

## 문서 구조

- `docs/pipelines/` — 파이프라인 설계 문서 (위 표 참고)
- `docs/reports_claude/` — 그 외 분석 리포트 초안

## 방법론 관례

- `random_state=156`을 모든 split/모델에서 고정 (노트북 간 비교 가능하게)
- train/test 분할 이전에 fit하는 벡터화·인코딩은 leakage trade-off로 인지하고 진행 (base에서 실측 영향 미미함 확인됨, `PIPELINE_base.md` 6장 참고)
- target encoding 등 타깃을 쓰는 피처는 반드시 train 쪽 통계만으로 계산 (base `PIPELINE_base.md` 9-3/9-5의 leakage-free 2-stage 방식 따를 것)
