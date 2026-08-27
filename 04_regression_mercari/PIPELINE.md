# Mercari 중고거래 가격 예측 — 파이프라인 설계 문서

`Mercari.ipynb` 하나만 보고도 전체 흐름을 재구성할 수 있도록, 데이터 특성 → 전처리 → 피처 엔지니어링 →
모델 학습/평가까지의 설계와 근거를 정리한 문서입니다. 이 문서 + 노트북 코드만 있으면 동일한 파이프라인을
처음부터 다시 구현할 수 있는 것을 목표로 합니다.

## 0. 목표와 평가지표

- **문제**: 상품명, 카테고리, 브랜드, 설명글, 상품 상태, 배송비 부담 여부로 **적정 판매가(price)** 를 예측하는 회귀 문제
- **데이터**: `../data/mercari_train.tsv` (탭 구분, 원본 shape `(1482535, 8)`)
- **평가지표**: RMSLE (Root Mean Squared Logarithmic Error) — 가격처럼 왜도가 큰 타깃에서 절대오차보다 로그오차를 재는 게 합리적
- **핵심 설계 선택**: 타깃 `price`를 미리 `log1p`로 변환해두면, 이후 일반 회귀모델의 RMSE 최소화가 곧 RMSLE 최소화와 수학적으로 동일해짐 → 별도의 RMSLE 전용 손실함수 없이 표준 회귀 파이프라인 그대로 사용 가능

---

## 1. 데이터 특성 (EDA 요약)

원본 컬럼: `train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description`

**결측치 (원본 기준)**

| 컬럼 | 결측 개수 |
|---|---|
| category_name | 6,327 |
| brand_name | 632,682 (전체의 약 43%) |
| item_description | 6 |

**price 분포**

- 오른쪽으로 심하게 치우친 분포 (min 0, max 2009, mean ≈ 26.7)
- `price == 0`인 행 존재 (증정품/입력 오류로 추정) → 노이즈로 보고 제거 대상
- 소수의 극단적 고가(2000달러 이상) 행 존재 → 이상치로 보고 제거 대상

**category_name 계층 구조**

- `대분류/중분류/소분류` 형태로 `/`로 구분된 문자열 (예: `Men/Tops/T-shirts`)
- 분할 후 유니크 개수: 대분류(cat_dae) 11개, 중분류(cat_jung) 114개, 소분류(cat_so) 871개

---

## 2. 전처리 파이프라인 (실행 순서 그대로)

전처리는 **전체 데이터(`mercari_df`) 기준으로 한 번에 수행**하고, train/test 분할은 모델 학습 직전(섹션 11)에서만 이루어집니다.
(이 설계의 data leakage 트레이드오프는 6장 참고)

| 단계 | 처리 내용 | shape 변화 |
|---|---|---|
| 0 | 원본 로드 (`pd.read_csv(sep='\t')`) | `(1482535, 8)` |
| 1 | `price > 0` 필터링 (0원 노이즈 제거) | `(1481661, 8)` |
| 2 | `price < 2000` 필터링 (고가 이상치 제거) | `(1481652, 8)` |
| 3 | `train_id` 컬럼 제거 + 완전 중복 행 제거 + `reset_index` | `(1481603, 7)` |
| 4 | `brand_name`/`category_name`/`item_description` 결측치 → `'Other_Null'` 문자열로 채움 | shape 불변 |
| 5 | `category_name` → `cat_dae`/`cat_jung`/`cat_so` 3단계 분할 (컬럼 3개 추가) | `(1481603, 10)` |
| 6 | `price` → `np.log1p` 변환 (`y_labels`도 동일하게 log1p) | shape 불변 |
| 7 | `item_description` 텍스트 클렌징 + 토큰화 (nltk) | shape 불변, 값이 문자열→토큰 리스트로 변경 |
| 8 | `null_penalty` 피처 생성 (Other_Null 개수 0~2) | 컬럼 1개 추가 |

### 단계별 핵심 코드

**1~3단계 — 이상치/식별자/중복 제거**
```python
mercari_df = mercari_df[mercari_df['price'] > 0]
mercari_df = mercari_df[mercari_df['price'] < 2000]
mercari_df.drop(columns='train_id', inplace=True)
mercari_df = mercari_df.drop_duplicates()
mercari_df = mercari_df.reset_index(drop=True)
```

**4단계 — 결측치를 삭제 대신 카테고리화**
```python
mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Other_Null')
mercari_df['category_name'] = mercari_df['category_name'].fillna('Other_Null')
mercari_df['item_description'] = mercari_df['item_description'].fillna('Other_Null')
```
단순 삭제 대신 `'Other_Null'`이라는 별도 카테고리로 채우는 이유: "브랜드/카테고리/설명이 없다"는 사실 자체가
가격과 상관관계가 있는 정보이기 때문 (실측: `null_penalty`가 클수록 평균 price가 낮아지는 경향 확인, 섹션 10-1 시각화 참고).
이 값은 이후 `OneHotEncoder`/`TfidfVectorizer`에서 하나의 카테고리/토큰처럼 취급됨.

**5단계 — 카테고리 3단계 분할**
```python
def split_cat(category_name):
    try:
        cat_dae, cat_jung, cat_so = category_name.split('/')[:3]
        return cat_dae, cat_jung, cat_so
    except ValueError:
        return 'Other_Null', 'Other_Null', 'Other_Null'

mercari_df['cat_dae'], mercari_df['cat_jung'], mercari_df['cat_so'] = zip(
    *mercari_df['category_name'].apply(lambda x: split_cat(x))
)
```
`/`가 없는 값(4단계에서 채운 `Other_Null` 포함)은 3개 컬럼 모두 `'Other_Null'`로 통일.

**6단계 — 타깃 로그 변환**
```python
y_labels = mercari_df['price']
mercari_df['price'] = np.log1p(y_labels)
y_labels = np.log1p(y_labels)
```
⚠️ 이 시점 이후 `mercari_df['price']`와 `y_labels`는 로그 스케일. `X_features`를 만들 때 `price` 계열 컬럼은 타깃이므로 반드시 제외.

**7단계 — item_description 텍스트 정제 (name/brand/category는 벡터화 단계의 기본 처리에 맡기고, 설명글만 별도 처리)**
```python
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # 영문자/공백 외 제거
    text = re.sub(r'\s+', ' ', text)        # 중복 공백 정리
    return text.strip()

stop_words = set(stopwords.words('english'))
negative_words = {'no', 'not', 'nor', 'never'}
stop_words = stop_words - negative_words     # 부정어는 의미가 커서 불용어에서 제외

def preprocess_text(text):
    text = text_cleaning(text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

mercari_df['item_description'] = mercari_df['item_description'].apply(preprocess_text)
```
결과: `item_description` 컬럼 값이 문자열이 아니라 **토큰 리스트**로 바뀜 (예: `['no', 'description', 'yet']`). 이는 이후
벡터화 단계에서 `analyzer=lambda x: x`를 쓰는 이유와 직결됨 (4장 참고).

**8단계 — Other_Null 개수를 수치형 피처로 명시화**
```python
NULL_PENALTY_COLS = ['brand_name', 'category_name', 'item_description']

def null_penalty(df, cols=NULL_PENALTY_COLS):
    return sum((df[col] == 'Other_Null').astype(int) for col in cols)

mercari_df['null_penalty'] = null_penalty(mercari_df)
```
결과 분포: `0`(결측 없음) 846,446건 / `1`(1개 결측) 631,706건 / `2`(2개 결측) 3,451건.
(원-핫/벡터에 이미 `Other_Null`이 반영되어 있지만, "결측 개수"라는 요약 신호를 별도 수치형으로 추가해 모델이 더 직접적으로 활용하도록 함)

---

## 3. Feature Engineering (벡터화 / 인코딩)

전처리가 끝난 `mercari_df`를 컬럼별 특성에 맞는 방식으로 각각 sparse matrix로 변환합니다.

| 컬럼 | 변환 방법 | 이유 | 결과 shape |
|---|---|---|---|
| `name` | `CountVectorizer(min_df=2, max_features=20000)` | 짧은 상품명은 단어의 등장 여부/빈도 자체가 중요 | `(1481603, 20000)` |
| `item_description` (토큰 리스트) | `TfidfVectorizer(min_df=2, max_features=30000, analyzer=lambda x: x)` | 문서 길이가 들쭉날쭉 → TF-IDF로 흔한 단어 가중치 완화. `analyzer=lambda x: x`로 이미 토큰화된 리스트를 그대로 사용(추가 문자열 파싱 불필요) | `(1481603, 30000)` |
| `brand_name` | `OneHotEncoder(handle_unknown='ignore', sparse_output=True)` | 고유값 4천 개 이상. `LabelBinarizer`는 이런 고카디널리티 컬럼에서 현저히 느림(수십 초 이상, 실측 ~82s) 반면 `OneHotEncoder`는 ~0.43s. `handle_unknown='ignore'`로 학습 때 못 본 값도 에러 없이 0벡터 처리 | `(1481603, 4808)` |
| `cat_dae` | 동일 방식 | 동일 | `(1481603, 11)` |
| `cat_jung` | 동일 방식 | 동일 | `(1481603, 114)` |
| `cat_so` | 동일 방식 | 동일 | `(1481603, 871)` |
| `item_condition_id`, `shipping`, `null_penalty` | `csr_matrix(df[cols].astype(float).values)` | 이미 수치형이므로 그대로 sparse 변환만 수행 | `(1481603, 3)` |

```python
name_vec = CountVectorizer(min_df=2, max_features=20000)
X_name = name_vec.fit_transform(mercari_df['name'])

desc_vec = TfidfVectorizer(min_df=2, max_features=30000, analyzer=lambda x: x)
X_descp = desc_vec.fit_transform(mercari_df['item_description'])

CAT_COLS = ['brand_name', 'cat_dae', 'cat_jung', 'cat_so']
X_cat = {}
for col in CAT_COLS:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat[col] = ohe.fit_transform(mercari_df[[col]])

NUM_COLS = ['item_condition_id', 'shipping', 'null_penalty']
X_num = csr_matrix(mercari_df[NUM_COLS].astype(float).values)
```

**차원 축소 관련 참고**: `max_features`/`min_df`는 hstack *이전에* 어휘 크기를 제한하는 방법(사전 캡)이고,
`TruncatedSVD`는 hstack *이후* 수학적으로 차원을 축소하는 방법(PCA는 sparse 미지원이라 대신 사용)입니다.
이 파이프라인은 전자(어휘 캡)만 적용했고, SVD 등 후처리 차원 축소는 적용하지 않았습니다.

---

## 4. hstack 결합 — 최종 `X_features`

```python
X_features = hstack([
    X_name,               # 20,000
    X_descp,              # 30,000
    X_cat['brand_name'],  #  4,808
    X_cat['cat_dae'],     #     11
    X_cat['cat_jung'],    #    114
    X_cat['cat_so'],      #    871
    X_num,                #      3
]).tocsr()
# X_features: (1481603, 55807)
```

블록을 개별 변수(`X_name`, `X_descp`, `X_cat[...]`, `X_num`)로 남겨뒀기 때문에, 이후 다른 조합(예: 설명글 제외 버전,
가중치 버전)을 만들 때 벡터화를 다시 하지 않고 재조합만 하면 됩니다 (5-B, 5-C 참고).

**Data leakage 트레이드오프**: 벡터화/인코딩을 train/test 분할 *이전*, 전체 데이터에 `fit`합니다. 엄밀히는 test
어휘가 학습에 살짝 새어 들어가는 leakage지만, 별도 벤치마크(Ridge 0.46774 vs 0.46806, LightGBM 0.49666 vs
0.49696)에서 실측 영향이 미미함을 확인했고, Kaggle 등에서도 흔히 쓰이는 실용적 단순화이므로 그대로 채택.

---

## 5. 모델 학습 & 평가

### 5-A. 공통 유틸리티

```python
def rmsle(y_test, preds):
    # y_labels가 이미 log1p(price)이므로, 이 스케일에서의 RMSE == 원본 스케일의 RMSLE
    return np.sqrt(mean_squared_error(y_test, preds))

def model_train_predict(model, X_features, y_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=156
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test
```
`train_test_split`을 항상 `random_state=156`으로 고정하기 때문에, 아래에서 비교하는 모든 모델이 **동일한 행 구성의
테스트셋**을 공유합니다 (다른 feature 조합이라도 행 수만 같으면 분할 인덱스가 동일) → RMSLE를 서로 공정하게 비교 가능.

### 5-B. 가중치 기반 `combine_features` (Ridge 전용 실험)

`category_name` > `shipping` > `null_penalty` > 나머지(`name`/`item_description`/`brand_name`/`item_condition_id`) 순으로
우선순위를 두고, 해당 블록에 배수를 곱해 결합:

```python
def combine_features(weights=None):
    weights = weights or {'category': 3.0, 'shipping': 2.0, 'null_penalty': 2.0, 'default': 1.0}
    num_weight_vec = np.array([weights['default'], weights['shipping'], weights['null_penalty']])
    X_num_weighted = csr_matrix(mercari_df[NUM_COLS].astype(float).values * num_weight_vec)
    return hstack([
        X_name * weights['default'],
        X_descp * weights['default'],
        X_cat['brand_name'] * weights['default'],
        X_cat['cat_dae'] * weights['category'],
        X_cat['cat_jung'] * weights['category'],
        X_cat['cat_so'] * weights['category'],
        X_num_weighted,
    ]).tocsr()

X_features_weighted = combine_features()  # (1481603, 55807) — 구조는 동일, 값의 스케일만 다름
```

**작동 원리**: sparse matrix에 스칼라를 곱하면 구조(행/열 수)는 그대로고 값의 크기만 커집니다. Ridge는
`‖y - Xw‖² + alpha·‖w‖²`를 최소화하는데, 어떤 블록의 값을 k배로 키우면 같은 기여도를 내는 데 필요한 계수가
`w/k`로 작아져도 되고, `‖w‖²` 페널티는 계수 크기에 걸리므로 작아진 계수는 덜 억제됩니다 → 결과적으로 Ridge가
그 블록을 더 적극적으로 사용하게 됨. **LightGBM에는 사실상 효과 없음** — 트리 분기는 값의 크기가 아니라
순서/분포로 결정되기 때문.

Ridge(원본)과 Ridge(가중치) 예측을 단순 평균한 **블렌드**도 함께 평가:
```python
blend_preds = (linear_preds + weighted_preds) / 2
```

### 5-C. 단계적(잔차) LightGBM 파이프라인 — "B안"

우선순위 피처로 먼저 설명하고, 남은 피처로 잔차(residual)를 보정하는 2단계 구조:

```python
# 1단계 피처: category(3개 원-핫) + shipping + null_penalty
X_priority = hstack([X_cat['cat_dae'], X_cat['cat_jung'], X_cat['cat_so'], X_num[:, [1, 2]]]).tocsr()
# (1481603, 998)

# 2단계 피처: name + item_description + brand_name + item_condition_id
X_rest = hstack([X_name, X_descp, X_cat['brand_name'], X_num[:, [0]]]).tocsr()
# (1481603, 54809)

# 11~12번과 동일한 random_state=156으로 분할 (인덱스 공유 목적)
idx_train, idx_test = train_test_split(np.arange(X_features.shape[0]), test_size=0.2, random_state=156)
y_arr = y_labels.to_numpy()
y_train_stage, y_test_stage = y_arr[idx_train], y_arr[idx_test]

# 1단계: 우선순위 피처만으로 LGBM 학습
stage1_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=156)
stage1_model.fit(X_priority[idx_train], y_train_stage)
stage1_train_pred = stage1_model.predict(X_priority[idx_train])
stage1_test_pred = stage1_model.predict(X_priority[idx_test])

# 2단계: 1단계 잔차를 나머지 피처로 보정
residual_train = y_train_stage - stage1_train_pred
stage2_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=125, random_state=156)
stage2_model.fit(X_rest[idx_train], residual_train)
stage2_test_pred = stage2_model.predict(X_rest[idx_test])

# 최종 예측 = 1단계 + 2단계(잔차) 예측
staged_preds = stage1_test_pred + stage2_test_pred
```

`train_test_split(np.arange(n), ...)`으로 분할하는 이유: `train_test_split`의 셔플은 `n_samples`와
`random_state`에만 의존하고 데이터 값과는 무관하므로, 배열 대신 인덱스를 분할해도 `X_features`/`y_labels`를
직접 분할했을 때와 동일한 행 구성이 나옵니다. → `X_priority`, `X_rest`, `y_labels` 세 가지를 동시에, 그리고
앞선 Ridge 실험과도 동일한 테스트셋으로 맞출 수 있음.

### 5-D. LightGBM 단일 모델 (베이스라인 비교용)

```python
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=125, random_state=156)
lgbm_preds, lgbm_y_test = model_train_predict(lgbm_model, X_features, y_labels)
```

---

## 6. 최종 결과 비교

| 모델 | 구성 | RMSLE | 비고 |
|---|---|---|---|
| 🥇 **Ridge(원본)** | `X_features` 그대로, `alpha=3` | **0.47067** | **최고 성능** |
| 2 | Ridge Blend | (Ridge 원본 + Ridge 가중치) 예측 평균 | 0.47084 |
| 3 | Ridge(가중치) | `combine_features()` 가중치 적용 | 0.47154 |
| 4 | LightGBM(단일) | `X_features`, `n_estimators=200` | 0.49220 |
| 5 | B안 단계적 LGBM | 우선순위→잔차 2단계 LGBM | 0.49735 |

### 인사이트

- **가장 단순한 원본 Ridge가 5개 중 최고 성능.** 가중치·블렌딩·단계적 LGBM 등 사람이 개입한 "우선순위" 가정이
  모두 오히려 성능을 깎아먹었음.
- Ridge는 정규화된 최소제곱법으로 **데이터에 최적인 계수 배분을 스스로 찾아냄**. 여기에 `category가 더
  중요할 것`이라는 수동 가중치(3배)를 강제로 얹으면, 데이터가 실제로 원했던 최적 균형에서 벗어나 성능이 소폭
  하락함 (0.47067 → 0.47154).
- LightGBM 계열(단일/2단계 모두)이 Ridge보다 낮은 성능을 보인 것은, 이 데이터의 피처가 5만+ 차원의 고차원
  sparse 텍스트 벡터 위주라 **선형모델이 트리 기반 모델보다 이런 구조를 더 잘 살린다**는 공통된 원인으로 보임.
  B안에서 문제를 두 단계로 나눠도 결국 두 단계 모두 LightGBM이라 이 한계를 벗어나지 못함.

---

## 7. 재현 방법

**필요 패키지**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk` (리소스: `punkt_tab`, `stopwords`), `lightgbm`

**데이터 경로**: `../data/mercari_train.tsv` (tab-separated, 노트북 기준 상대경로)

**실행 순서**: 노트북 셀을 위에서 아래로 순서대로 실행하면 본 문서의 흐름 그대로 재현됩니다.
- 섹션 1~2: 데이터 로드 및 기본 탐색
- 섹션 3~8-1: 전처리 (2장)
- 섹션 9~10-1: 벡터화/인코딩, hstack, 학습 전 데이터 시각화 (3~4장)
- 섹션 11: Ridge vs LightGBM 기본 비교, `model_train_predict`/`rmsle` 정의 (5-A, 5-D)
- 섹션 12: 가중치 `combine_features` + Ridge 블렌드 (5-B)
- 섹션 13~14: B안 단계적 LGBM + 전체 지표 비교 (5-C, 6장)

**주의사항**
- `mercari_df['price']`는 섹션 7(로그 변환) 이후 로그 스케일로 덮어써지므로, `X_features`에 절대 포함하지 않도록 주의.
- 전체 데이터(148만 행) 처리라 `item_description` 토큰화, LightGBM 학습 셀은 실행 시간이 오래 걸릴 수 있음 (LightGBM 1회 학습 기준 수 분 단위).
- `random_state=156`을 모든 분할/모델에서 고정해야 5장의 비교표와 동일한 수치가 재현됨.

---

## 8. 향후 개선 아이디어 (미적용)

- `TruncatedSVD`로 hstack 이후 차원 축소 시도 (현재는 `max_features` 캡만 적용)
- Ridge `alpha` 그리드/베이지안 탐색으로 정규화 강도 튜닝
- `max_features`(어휘 크기) 확대/축소에 따른 RMSLE 민감도 실험
- 수동 가중치 대신 실제 stacking(메타러너로 최적 결합 가중치를 학습)으로 5-B 재시도
