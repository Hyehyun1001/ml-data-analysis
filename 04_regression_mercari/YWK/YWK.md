# 1. mercari 가격 추천 모델
- 이 프로젝트는 mercari 마켓의 기존 데이터를 활용하여 이후 추가되는 제품 혹은 상품의 가격을 추천해주는 시스템을 만든다.

## 코드 진행 순서 (YWK.ipynb 실제 구현 기준)
1. 데이터 로드 & 정제 — 가격 이상치 제거, log1p 변환 (+ 카테고리/컨디션/배송 분포 시각화)
2. category_name 3단계 분리
3. 학습 / 테스트 데이터 분리 (`train_test_split`)
4. 브랜드 보강 — 사전 생성 → train 매칭 → test 적용 → Unknown 통일 (+ 처리 전후 시각화)
5. 텍스트 클렌징 — `name`, `item_description` 공통 전처리
6. 인코딩 / 벡터화 — TF-IDF(name, description), 원-핫(brand/category), 스케일링(item_condition_id)
7. Feature 결합 & 파이프라인 저장 (+ feature 구성비 시각화)
8. 모델 학습 & 평가 — Ridge 베이스라인, LightGBM (+ 학습곡선/비교/importance/예측 산점도 시각화)

## 데이터 전처리 (구현 내용)

### 1. 가격 이상치 처리 & 타깃 변환
- 원본 1,482,535행 중 `price ≤ 0`(874건, 무효 데이터)과 `price > 2,000`(3건, Mercari 실제 최대 등록가 정책 기준)을 제거.
- `train_id` 제거, 중복 행 제거 → 정제 후 **1,481,609행**.
- `price`는 분포가 크게 치우쳐 있어 `log1p` 변환 후 학습에 사용하고, 예측 후 `expm1()`로 복원. 평가지표는 **RMSLE**(y가 이미 log1p 상태라 로그 스케일 RMSE와 수학적으로 동일하게 계산).

### 2. category_name 3단계 분리
- `category_name`(`Main/Sub/SubSub` 형태)을 `cat_main` / `cat_sub` / `cat_sub2` 3개 컬럼으로 분리.
- 원본 결측치는 분리 전 `Unknown/Unknown/Unknown`으로 채운 뒤 분리, `str.split('/', n=2, expand=True)`로 3단계까지만 자름.
- brand_name과 동일한 원칙: train에 등장한 조합만 known set으로 인정하고, test에만 있는 새 조합은 `Unknown`으로 통일.

### 3. brand_name 결측치 처리
- **train_test_split 이후, train에서만** `brand_name`이 이미 채워진 값들로 브랜드 사전(`known_brands`)을 생성 → 실제 **4,535개** 브랜드.
- 결측 행은 `item_description`(클렌징 이전 원문 상태)에서 사전 속 브랜드를 탐색해 채움.
  - 문자열 정규화(소문자화) + 단어 경계 정규식(`\b`)으로 오탐 방지.
  - **긴 브랜드명을 먼저 매칭**하도록 길이 내림차순으로 정렬 후 정규식 alternation을 구성 ("Kate"가 "Kate Spade"보다 먼저 매칭되는 문제 방지).
- 매칭에 끝내 실패한 나머지는 `Unknown`으로 통일. test에는 train 사전을 **적용만**(재생성 X) 하며, train 사전에 없는 브랜드는 test에서 원래 값이었든 매칭으로 찾은 값이든 전부 `Unknown`으로 처리.
- 브랜드 사전은 `brand_dictionary.pkl`로 저장.
- **실제 처리 결과**: train/test 모두 결측 0건. 상위 빈도 — `Unknown` 196,041 · `AND` 99,396 · `PINK` 55,325 · `Nike` 43,972 · `Victoria's Secret` 39,366.

### 4. item_condition_id / shipping
- `item_condition_id`(1~5, 결측 없음)는 순서형 수치로 그대로 사용, `shipping`(0/1)도 그대로 사용.
- EDA 단계에서 `cat_main` 상위 분포, `item_condition_id`/`shipping` 분포를 시각화해 확인만 하고, 수동 구간화는 하지 않음 — 해당 패턴은 모델(GBM)이 자동으로 학습하도록 위임.

### 5. item_description / name 텍스트 feature화
- `name`, `item_description` 둘 다 클렌징(소문자화 → **영문자+숫자 외 제거** → 불용어 제거 → 토큰화 → 재조합) 후 `TfidfVectorizer`로 벡터화.
  - 초기엔 `[^a-z\s]`로 숫자까지 제거했으나, 실험 결과(아래 "실험: 숫자 유지 여부 비교" 참고) 숫자를 남기는 쪽(`[^a-z0-9\s]`)이 더 좋아서 **숫자를 유지하는 것으로 변경**했다. `4gb`, `32gb`처럼 용량/사이즈 정보가 가격 신호로 작동하는 것으로 보임.
  - `name_vectorizer`: `max_features=30000`, `ngram_range=(1,2)`
  - `desc_vectorizer`: `max_features=50000`, `ngram_range=(1,2)`
- 불용어 처리 시 부정어(`no/not/nor/never/without`)는 보존하고, `[rm]` 마스킹 잔여 토큰(`rm`)은 불용어로 추가 제거.
- train에서만 `fit()`, test는 `transform()`만 적용. 벡터라이저 객체는 `name_vectorizer.pkl`/`desc_vectorizer.pkl`로 저장해 재사용.

### 6. 인코딩 / 스케일링
- `brand_name`, `cat_main`, `cat_sub`, `cat_sub2` → `OneHotEncoder(handle_unknown='ignore')`로 인코딩 (5,527차원). unseen 값은 전부 0벡터 처리되어 6번 항목의 `Unknown` 통일과 이중으로 안전장치가 걸림.
- `item_condition_id` → `StandardScaler`로 표준화 (Ridge 베이스라인을 위함, GBM에는 불필요하지만 공통 파이프라인 유지).

## 최종 Feature 구성
- `X_train` : **1,185,287 × 85,529**, `X_test` : 296,322 × 85,529
- 구성: brand/category 원-핫 5,527 · item_condition_id 1 · shipping 1 · name TF-IDF 30,000 · description TF-IDF 50,000
- 모든 전처리 객체(브랜드 사전, known category set, 인코더, 스케일러, 벡터라이저)를 `mercari_pipeline.pkl` 하나로 묶고, `preprocess_new_data()` 함수로 신규 데이터에도 동일하게 재현되도록 구성.

## 실험: 숫자 유지 여부 비교
- `text_cleaning`에서 숫자를 제거(`[^a-z\s]`)할지 유지(`[^a-z0-9\s]`)할지를, 나머지 파이프라인(같은 `train_test_split(random_state=42)`, 같은 brand/category 전처리)은 전부 동일하게 고정한 채 Ridge로 A/B 비교.

| 버전 | Ridge RMSLE (alpha=1.0) |
|---|---|
| 숫자 제거 (기존) | 0.45892 |
| **숫자 유지 (채택)** | **0.45597** |
| 차이 | -0.00295 (숫자 유지가 더 좋음) |

- 결론: 숫자를 남기는 쪽을 최종 방식으로 채택 (5번 항목 반영). YWK.ipynb의 Ridge 셀 바로 다음 셀에서 이 비교를 확인할 수 있음.

## 실험: Ridge alpha 스윕 (1, 2, 3, 4, 5)
- 숫자 유지 버전 기준으로, 나머지는 전부 고정한 채 alpha만 바꿔가며 Ridge RMSLE 비교.

| alpha | RMSLE |
|---|---|
| 1 | 0.45597 |
| 2 | 0.45483 |
| 3 | **0.45447** (최적) |
| 4 | 0.45448 |
| 5 | 0.45468 |

- alpha가 커질수록 좋아지다가 3에서 최저점을 찍고 다시 나빠지는 U자 형태 — 과적합(낮은 alpha)과 과소적합(높은 alpha) 사이의 균형점. **alpha=1.0 대신 alpha=3.0을 최종 베이스라인으로 쓰는 걸 고려할 것.**

## 모델링 결과
- **타깃**: `log1p(price)`, **평가지표**: RMSLE
- **베이스라인 (Ridge, alpha=1.0)**: RMSLE **0.45597** (숫자 유지 버전 기준. 숫자 제거 버전은 0.45892였음 — 위 실험 참고). alpha 스윕 결과 **alpha=3.0에서 0.45447**로 더 좋음 — 아직 기본 셀은 alpha=1.0 유지 중.
- **본 모델 (LightGBM)**: `learning_rate=0.1`, `num_leaves=31`. 원래 `num_boost_round=2000`으로 학습 시 **early stopping 없이 종료**(valid rmse가 끝까지 하락 중, 학습 시간 약 97분) → RMSLE **0.45122**(숫자 제거 버전 기준)였으나, 매번 97분씩 기다리지 않도록 `FAST_LGBM_CHECK` 스위치를 추가해 기본값은 `num_boost_round=150`으로 빠르게 검증하도록 변경. **최종 성능을 다시 확인하려면 `FAST_LGBM_CHECK=False`로 바꿔 2000라운드 풀 학습을 재실행해야 함**.
- **Feature Importance 상위(gain 기준, 숫자 제거 버전 기준 기록)**: `shipping` · `name__lularoe` · `cat_sub_Shoes` · `desc__box` · `desc__authentic` · `item_condition_id` · `cat_sub_Women's Handbags` 등 — 브랜드/카테고리/텍스트가 고르게 상위권에 분포.
- 학습된 모델은 `mercari_gbm_model.txt`(LightGBM), `mercari_ridge_model.pkl`(Ridge)로 저장하고, `predict_price(raw_df)` 함수로 신규 상품 하나에 대해서도 바로 가격을 예측할 수 있도록 마무리.
- Ridge 계수(`coef_`)를 brand_name / category / item_condition_id / shipping / name / description **6개 그룹**으로 묶어 그룹별 대표 가중치(`sum(|coef|)`, `mean(|coef|)`)를 확인하는 셀을 노트북 맨 끝에 추가함.

## 시각화
- **YWK.ipynb 내 9개 셀** (전부 영문 라벨): 카테고리/컨디션/배송 분포 → 브랜드 결측치 처리 전후 → feature 구성비 → Ridge 숫자 제거 vs 유지 비교 → Ridge alpha 스윕(1/2/3/4/5) → LightGBM 학습곡선(train vs valid) → Ridge vs GBM 비교 → Feature Importance Top 15 → 실제값 vs 예측값 산점도(log1p 스케일, test 5,000건 샘플).
- 모든 시각화는 `YWK/visualizations/` 폴더에 번호 순서(01~11) PNG로도 저장해둠 — `10_ridge_digits_removed_vs_kept.png`, `11_ridge_alpha_sweep.png`가 최신 추가분.
- 파이프라인 전체를 정리한 별도 리포트 `mercari_report.html` 제작 (진행 순서 · 핵심 개념 · 설계 결정 · 위 시각화를 실제 데이터로 재구성).

## 다음 단계 제안
- **Ridge alpha를 1.0 → 3.0으로 교체 검토** — alpha 스윕에서 3.0이 가장 좋았음 (0.45447).
- **숫자 유지 버전으로 LightGBM 풀 재학습** — `FAST_LGBM_CHECK=False`로 바꿔서 2000라운드로 다시 돌려 최종 RMSLE/학습곡선/feature importance를 갱신할 것 (현재 기록된 LightGBM 수치는 숫자 제거 버전 기준으로 남아있음).
- LightGBM `num_boost_round` 증량 검토 — 2000라운드에서도 valid rmse가 계속 하락 중이라 여지가 있음.
- 브랜드 매칭 오탐 점검 — `AND`, `PINK`처럼 일반 단어와 겹치는 상위 브랜드의 매칭 정확도 표본 확인.
- 브랜드 수가 크게 늘어날 경우 매칭 속도 최적화(Aho-Corasick 등) 검토.
