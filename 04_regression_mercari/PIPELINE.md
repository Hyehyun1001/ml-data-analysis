# Mercari 중고거래 가격 예측 — 파이프라인 설계 문서

`teamipynb/소현_mercari.ipynb` 하나만 보고도 전체 흐름을 재구성할 수 있도록, 데이터 특성 → 전처리 → 피처 엔지니어링 →
모델 학습/평가까지의 설계와 근거를 정리한 문서입니다. 이 문서 + 노트북 코드만 있으면 동일한 파이프라인을
처음부터 다시 구현할 수 있는 것을 목표로 합니다.

## 0. 목표와 평가지표

- **문제**: 상품명, 카테고리, 브랜드, 설명글, 상품 상태, 배송비 부담 여부로 **적정 판매가(price)** 를 예측하는 회귀 문제
- **데이터**: `../../data/mercari_train.tsv` (노트북이 `teamipynb/` 하위에 있어 상대경로가 한 단계 더 깊음, 탭 구분, 원본 shape `(1482535, 8)`)
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

## 5. 모델 학습 전략

성능 개선은 세 갈래 전략으로 접근했습니다.

1. **hstack 구성 방식을 바꿔보기** (5-1) — 같은 원재료(name/description/category/수치형)를 다른 방식으로
   인코딩/결합해서 hstack 자체를 여러 버전으로 만들어봄 (가중치 → 실패, 단계 분리 → 실패, 인코딩 방식 교체 → 성공)
2. **baseline으로 Ridge/LightGBM을 각각 돌려서 비교** (5-2) — 같은 hstack(`X_features`)에 대해 선형모델과
   트리모델의 성능 차이를 먼저 확인
3. **가장 성능 좋았던 Ridge를 `alpha` 튜닝으로 추가 최적화** (5-3) — hstack 구성은 손대지 않고, 정규화 강도만
   조정해서 더 짜낼 여지가 있는지 확인

### 공통 유틸리티

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
테스트셋**을 공유합니다 (다른 feature 조합/split 호출이라도 행 수만 같으면 분할 인덱스가 동일) → RMSLE를 서로 공정하게 비교 가능.

---

## 5-1. 전략 ① — hstack 구성 방식을 바꿔보기

같은 벡터화/인코딩 결과물(`X_name`, `X_descp`, `X_cat[...]`, `X_num`)을 어떻게 조합·변형하느냐에 따라
결과가 어떻게 달라지는지 4가지 버전을 실험했습니다.

**1) hstack 안에서 블록 순서 자체는 결과에 영향 없음** — `hstack([X_name, X_descp, X_cat, X_num])`이든
`hstack([X_num, X_cat, X_descp, X_name])`이든, Ridge/LightGBM 둘 다 각 컬럼을 독립적으로 취급하기 때문에
배치 순서는 학습 결과에 전혀 영향을 주지 않음.

**2) 하지만 "어떤 블록을 넣고, 그 블록을 어떻게 만들었는지"는 결과를 완전히 바꿈** — 이건 순서 문제가 아니라
**hstack의 재료(X) 자체가 달라지는 것**이라, 당연히 fit 결과가 달라짐. 지금까지 직접 실험한 걸로 증명이 됐음:

| 버전 | hstack 구성 방식 | 결과 |
|---|---|---|
| `X_features` (원본) | name/desc 유니그램 + 범주형 원-핫(4개 블록, 5,804차원) + 수치형 | Ridge RMSLE **0.47067** (최고) |
| `X_priority`/`X_rest` (B안) | 같은 블록들을 두 그룹으로 쪼개서 별도 학습 | LGBM RMSLE 0.49735 (가장 나쁨) |
| `X_features_v2` | name/desc 유니그램+바이그램 + 범주형 라벨 인코딩(원-핫 대신 4컬럼) | LGBM RMSLE **0.47363** (LightGBM 중 최고) |

즉:
- **블록을 아예 분리해서 별도 모델로 학습하면**(B안, `X_priority`/`X_rest`) → 완전히 다른 학습 절차가 되어 결과가 크게 달라짐
- **같은 컬럼을 다른 방식으로 인코딩하면**(원-핫 vs 라벨, 유니그램 vs 바이그램, `X_features_v2`) → matrix 자체의 차원과 값이 바뀌어서 모델이 보는 정보량/표현력이 달라짐

정리하면 — **"어떻게 배치했는지"는 무관하지만, "무엇을 어떻게 인코딩해서 넣었는지"는 학습 결과를 결정짓는 핵심 요인**.
아래 5-1-a~c는 이 표의 3개 버전을 각각 어떻게 만들었는지의 상세.

### 5-1-a. 원본 (baseline) — 4장의 `X_features`

원-핫 인코딩 + 유니그램 벡터화 그대로 hstack. 아래 모든 실험의 비교 기준점.

### 5-1-b. 단계적(잔차) LightGBM 파이프라인 — "B안" — ⚠️ 실패, 노트북에서 주석 처리됨

우선순위 피처로 먼저 설명하고, 남은 피처로 잔차(residual)를 보정하는 2단계 구조:

```python
# 1단계 피처: category(3개 원-핫) + shipping + null_penalty
X_priority = hstack([X_cat['cat_dae'], X_cat['cat_jung'], X_cat['cat_so'], X_num[:, [1, 2]]]).tocsr()
# (1481603, 998)

# 2단계 피처: name + item_description + brand_name + item_condition_id
X_rest = hstack([X_name, X_descp, X_cat['brand_name'], X_num[:, [0]]]).tocsr()
# (1481603, 54809)

# 11번과 동일한 random_state=156으로 분할 (인덱스 공유 목적)
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

### 5-1-c. LGBM 네이티브 categorical + n-gram — ✅ 채택, LightGBM 성능을 실제로 끌어올린 방법

5-1-b(단계적 잔차)가 원본 Ridge보다 못했던 원인을, "우선순위를 사람이 정하는 방식" 자체가
아니라 **LGBM에 넣는 피처 표현 방식**에서 찾아본 시도. hstack 구조는 유지하되, 그 안에 들어가는 두 블록의
인코딩 방식만 LGBM에 유리하게 교체함.

**① n-gram 확장** — `name`은 `CountVectorizer(ngram_range=(1,2))`로 바로 적용 가능하지만, `item_description`은
이미 토큰 리스트로 정제되어 있어 `ngram_range` 파라미터가 먹히지 않음(커스텀 `analyzer`를 쓰면 sklearn이 자체
n-gram 로직을 건너뜀). 그래서 토큰 리스트에서 n-gram을 직접 생성하는 analyzer를 정의:

```python
name_vec_v2 = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
X_name_v2 = name_vec_v2.fit_transform(mercari_df['name'])

def ngram_analyzer(tokens, n_range=(1, 2)):
    grams = []
    for n in range(n_range[0], n_range[1] + 1):
        grams += [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams

desc_vec_v2 = TfidfVectorizer(min_df=2, max_features=30000, analyzer=lambda x: ngram_analyzer(x, (1, 2)))
X_descp_v2 = desc_vec_v2.fit_transform(mercari_df['item_description'])
# X_name_v2 : (1481603, 20000)
# X_descp_v2: (1481603, 30000)
```

**② LGBM 네이티브 categorical** — `brand_name`/`cat_dae`/`cat_jung`/`cat_so`를 `OneHotEncoder`(총 5,804차원)로
쪼개는 대신 `LabelEncoder`로 정수 라벨화하고, LGBM의 `categorical_feature` 파라미터로 "이 컬럼들은 범주형"이라고
직접 알려줌 → 트리가 원-핫보다 훨씬 적은 컬럼으로 카테고리 분기를 처리:

```python
from sklearn.preprocessing import LabelEncoder

cat_label_df = pd.DataFrame(index=mercari_df.index)
for col in CAT_COLS:
    le = LabelEncoder()
    cat_label_df[col] = le.fit_transform(mercari_df[col])

X_cat_label = csr_matrix(cat_label_df.values.astype(float))
# X_cat_label: (1481603, 4)  <- 원-핫 5,804차원이 컬럼 4개로 압축됨
```

**③ 재결합 + 학습** — hstack에 새 블록만 교체해 넣고, 결합된 matrix에서 범주형 컬럼의 위치(인덱스)를 계산해
`categorical_feature`로 전달:

```python
X_features_v2 = hstack([X_name_v2, X_descp_v2, X_cat_label, X_num]).tocsr()
# X_features_v2: (1481603, 50007)  <- 원본(55807)보다 오히려 축소됨

n_text = X_name_v2.shape[1] + X_descp_v2.shape[1]
cat_feature_idx = list(range(n_text, n_text + len(CAT_COLS)))
# [50000, 50001, 50002, 50003] -> ['brand_name', 'cat_dae', 'cat_jung', 'cat_so']

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    X_features_v2, y_labels, test_size=0.2, random_state=156
)
lgbm_v2 = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=125, random_state=156)
lgbm_v2.fit(X_train_v2, y_train_v2, categorical_feature=cat_feature_idx)
v2_preds = lgbm_v2.predict(X_test_v2)
```

---

## 5-2. 전략 ② — Ridge vs LightGBM 베이스라인 비교

5-1에서 만든 3가지 `X_features` 버전 중 **원본(5-1-a)** 을 기준으로, 선형모델(Ridge)과 트리모델(LightGBM)이
같은 hstack에 대해 얼마나 다른 성능을 내는지 먼저 확인:

```python
# Ridge
linear_model = Ridge(solver='lsqr', fit_intercept=False, alpha=3)
linear_preds, linear_y_test = model_train_predict(linear_model, X_features, y_labels)
print('Ridge RMSLE:', rmsle(linear_y_test, linear_preds))
# Ridge RMSLE: 0.47067

# LightGBM
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=125, random_state=156)
lgbm_preds, lgbm_y_test = model_train_predict(lgbm_model, X_features, y_labels)
print('LightGBM RMSLE:', rmsle(lgbm_y_test, lgbm_preds))
# LightGBM RMSLE: 0.49220
```

이 시점의 결론: **같은 피처 구성이면 Ridge가 LightGBM보다 낫다** (0.47067 vs 0.49220). 이 격차를 줄일 수
있는지가 5-1-b~c의 동기가 됨 — 결과적으로 단계분리(5-1-b)는 실패했고, 피처 표현 방식 교체
(5-1-c)로 격차를 0.02154 → 0.00296까지 좁힘.

---

## 5-3. 전략 ③ — Ridge `alpha` 튜닝 (hstack 구성은 5-1-a 그대로 고정, 간단 요약)

hstack 구성은 그대로 두고 `alpha`를 `[0.01, 1000]` 범위로 그리드서치한 결과, 5-2에서 임의로 쓰던 `alpha=3`이
이미 최적값이었습니다(RMSLE **0.47067**, 양옆 `alpha=1`(0.47140)/`alpha=10`(0.47146) 모두 더 나쁨 → 명확한
극소점). `alpha`가 너무 작으면 과소 규제, 너무 크면(`alpha=1000`→0.52934) 과도 규제로 텍스트 피처 계수가
눌려 급격히 나빠집니다. 이 방향에서는 추가로 짜낼 성능은 없었지만, 현재 `alpha`가 이미 최적 근처라는 걸
실측으로 확인했다는 데 의미가 있습니다.

---

## 6. 최종 결과 비교

> ⚠️ **이 표는 14~16번 섹션(노트북) 시점 기준입니다.** 이후 9장에서 vocab 크기 검증, 새 피처,
> FM_FTRL 모델 축 추가, leakage 없는 블렌딩까지 진행해 RMSLE 0.42114까지 갱신했고, 이후 LGBM `n_estimators`를
> 2000까지 이어학습(9-6)해 **최종 RMSLE 0.41778**까지 추가로 낮췄습니다. 이 전체가
> 지금은 노트북 17~22번 섹션에 정식으로 통합되어 있습니다. 본격적인 성능 개선의 핵심은 사실 아래 표보다
> **9장(특히 9-5, LGBM 튜닝 + FM_FTRL + 블렌딩)** 쪽이니 그쪽을 중심으로 보는 걸 추천합니다.

| 순위 | 모델 | 구성 | RMSLE | 전략 |
|---|---|---|---|---|
| 🥇 | **Ridge(원본, alpha=3)** | `X_features` 그대로 | **0.47067** | 5-2 baseline = 5-3 튜닝 결과 (이미 최적) |
| 🥈 | **LGBM(네이티브 categorical + n-gram)** | `X_features_v2`, 5-1-c 참고 | **0.47363** | 5-1 hstack 구성 변경 |
| 3 | LightGBM(단일, 원-핫+유니그램) | `X_features`, `n_estimators=200` | 0.49220 | 5-2 baseline |
| 4 | B안 단계적 LGBM `(보류)` | 우선순위→잔차 2단계 LGBM | 0.49735 | 5-1-b |

`(보류)` 표시된 1개는 노트북에서 코드가 주석 처리되어 더 이상 실행되지 않음 (12번 섹션). 현재 실제로
실행되는 것은 Ridge(원본, 11번) · LightGBM 단일(11번) · LGBM 네이티브+n-gram(14~15번) · Ridge alpha
그리드(16번) 4가지 — 그리고 이 표에는 없지만 실질적인 성능 개선은 17번 섹션 이후(9장)에서 일어남.

### 전략별 인사이트

**전략 ① (5-1) hstack 구성 방식을 바꾼 결과 — 절반은 성공, 절반은 실패**
- 5-1-b(단계 분리, B안)는 원본보다 못했음. Ridge는 정규화된 최소제곱법으로 **데이터에 최적인
  계수 배분을 스스로 찾아내는데**, 여기에 사람이 정한 우선순위(2단계 분리)를 강제로 얹으면
  데이터가 실제로 원했던 최적 균형에서 벗어남.
- 5-1-c(네이티브 categorical + n-gram)는 반대로 성공 — **모델(LGBM)의 구조적 특성에 맞춰 피처의 "표현
  방식"만 바꾼 것**이 핵심. 우선순위를 강제하지 않고, 트리 모델이 원-핫보다 라벨 인코딩+`categorical_feature`를
  더 효율적으로 분기하는 특성과 n-gram으로 늘어난 표현력을 그대로 살려서 LightGBM RMSLE를 0.49220 →
  0.47363로 약 3.8% 개선.
- 결론: **"우선순위를 사람이 강제하는 방식"보다 "모델 구조에 맞는 인코딩으로 바꾸는 방식"이 유효했음.** 다만
  이 개선폭(3.8%)은 9장에서 LGBM을 제대로 정규화·early stopping 튜닝했을 때의 개선폭(약 9.6%, 아래 참고)에
  비하면 작은 수준 — hstack 구성보다 하이퍼파라미터 튜닝 쪽이 훨씬 큰 지렛대였음.

**전략 ② (5-2) Ridge vs LightGBM 베이스라인**
- 같은 hstack(`X_features`)에 대해 Ridge(0.47067)가 LightGBM(0.49220)보다 확실히 우수. 이 데이터의 핵심
  신호가 5만+ 차원 고차원 sparse 텍스트 벡터 위주라 **선형모델이 트리 기반보다 이런 구조를 잘 살리는 경향**
  때문으로 보임 (전략 ①에서 LGBM 격차를 크게 좁혔지만 완전히 역전하진 못함).

**전략 ③ (5-3) Ridge alpha 튜닝**
- hstack 구성은 그대로 두고 `alpha`만 `[0.01, 1000]` 범위로 탐색한 결과, 기존에 쓰던 `alpha=3`이 이미
  그리드 내 최적값이었음(RMSLE 0.47067, 변화 없음). 즉 이 방향에서는 추가로 짜낼 성능이 없었지만, 실측으로
  "현재 alpha가 이미 최적 근처"임을 확인한 것 자체가 성과.

---

## 7. 재현 방법

**필요 패키지**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk` (리소스: `punkt_tab`, `stopwords`), `lightgbm`

**데이터 경로**: `../../data/mercari_train.tsv` (tab-separated, `teamipynb/소현_mercari.ipynb` 기준 상대경로)

**실행 순서**: 노트북 셀을 위에서 아래로 순서대로 실행하면 본 문서의 흐름 그대로 재현됩니다.
- 섹션 1~2: 데이터 로드 및 기본 탐색
- 섹션 3~8-1: 전처리 (2장)
- 섹션 9~10-1: 벡터화/인코딩, hstack, 학습 전 데이터 시각화 (3~4장)
- 섹션 11: `model_train_predict`/`rmsle` 정의, Ridge vs LightGBM 베이스라인 비교 (5-2)
- 섹션 12~13: `(보류, 코드 주석 처리)` B안 단계적 LGBM + 지표 비교 (5-1-b)
- 섹션 14~15: LGBM 네이티브 categorical + n-gram + 지표 비교 (5-1-c)
- 섹션 16: Ridge `alpha` 그리드서치 (5-3)
- 섹션 17: 새 피처 엔지니어링 — 길이 피처 + target encoding (9-3)
- 섹션 18: FM_FTRL 직접 구현 (9-4)
- 섹션 19~21: 2-stage leakage-free 검증 — 데이터 분할 / Stage 1 튜닝(valid) / Stage 2 최종평가+블렌딩(test) (9-5)
- 섹션 22: 최종 결과 요약 — **본격적인 성능 개선은 여기 17~21번 구간에서 일어남** (0.49220 → 0.42114)
- 섹션 23: LGBM n_estimators 1200→2000 이어학습 (9-6) — 최종 **0.41778**

**주의사항**
- `mercari_df['price']`는 섹션 7(로그 변환) 이후 로그 스케일로 덮어써지므로, `X_features`에 절대 포함하지 않도록 주의.
- 전체 데이터(148만 행) 처리라 `item_description` 토큰화, LightGBM 학습 셀은 실행 시간이 오래 걸릴 수 있음 (LGBM 1200라운드 튜닝 기준 수십 분 단위).
- `random_state=156`을 모든 분할/모델에서 고정해야 6장의 비교표와 동일한 수치가 재현됨.
- 12~13번 섹션은 코드가 주석 처리되어 있어 그대로 실행해도 아무 일도 일어나지 않음(의도된 상태). 다시 활성화하려면
  `#` 주석을 해제하면 되고, 이때 `linear_preds` 등 참조 변수가 앞 셀에서 먼저 정의되어 있어야 함.

---

## 8. 향후 개선 아이디어

- ~~`max_features`(어휘 크기) 확대/축소에 따른 RMSLE 민감도 실험~~ → **완료, [9-1](#9-1-vocab-크기max_features-민감도-실험) 참고** (결론: 이미 충분히 큼, 캡을 더 키울 필요 없음)
- ~~LGBM에도 `early_stopping_rounds`/`num_leaves`/`learning_rate` 등 하이퍼파라미터 튜닝 적용~~ → **완료, [9-5](#9-5-최종-leakage-free-파이프라인--블렌딩) 참고** (0.47363 → 0.42881, test 기준)
- ~~수동 가중치 대신 실제 stacking(메타러너로 최적 결합 가중치를 학습)으로 재시도~~ → **완료, [9-5](#9-5-최종-leakage-free-파이프라인--블렌딩) 참고** (모델 예측값을 valid에서 가중치 그리드서치로 결합 — 완전한 학습형 메타러너는 아니지만 leakage 없는 결합 가중치 탐색은 달성)
- `TruncatedSVD`로 hstack 이후 차원 축소 시도 (vocab 실험 결과 캡을 더 키우는 게 무의미한 것으로 확인됐으므로, 반대 방향인 차원 축소도 성능에 큰 영향은 없을 가능성이 높지만 미검증)
- Ridge `alpha`를 더 촘촘한 그리드(예: 1~5 사이 0.5 간격)로 재탐색해 소수점 단위 추가 개선 여지 확인 ([9-5](#9-5-최종-leakage-free-파이프라인--블렌딩)에서 `{1,3,10}` 성긴 그리드로는 `alpha=3` 재확인만 함)
- ~~FM_FTRL을 노트북(`teamipynb/소현_mercari.ipynb`)에 정식 셀로 통합~~ → **완료** — 새 피처/FM_FTRL/2-stage 블렌딩 전체를
  실제 셀(17~22번 섹션)로 반영하고 전체 재실행까지 검증함 ([9-4](#9-4-fm_ftrl-직접-구현-numpyscipy)/[9-5](#9-5-최종-leakage-free-파이프라인--블렌딩) 참고)
- FM_FTRL의 `k`(latent factor 차원)/정규화 하이퍼파라미터도 Ridge/LGBM처럼 그리드서치

---

## 9. vocab 크기 / 피처 표현 / FM_FTRL 모델 축 추가 — **본격적인 성능 개선 구간**

14~16번 섹션(노트북) 이후 별도 세션에서 심화 실험을 진행했고, 그 결과물(새 피처 엔지니어링, FM_FTRL,
leakage-free 2-stage 블렌딩)을 지금은 **노트북 17~22번 섹션에 정식 코드 셀로 통합**해뒀습니다. `0.49220`
(11번 섹션 LGBM 최초 baseline) 에서 시작해 **`0.41778`까지, RMSLE 기준 약 15.1% 개선**한 구간이 바로 여기이고,
1~8장에서 다룬 hstack 구성 실험(5-1, 최대 3.8% 개선)보다 훨씬 큰 지렛대였습니다. 모든 실험은 노트북과
동일한 `mercari_train.tsv` 전처리(섹션 1~8-1)와 **`random_state=156`으로 만든 동일한 test set**을 기준으로
비교했습니다.

### 9-1. vocab 크기(`max_features`) 민감도 실험

`name`/`item_description`에 `min_df=2`만 걸고 `max_features` 캡을 없앴을 때 실제 어휘 크기:

| | unigram | ngram(1,2) |
|---|---|---|
| `name` | 47,199 | 401,894 |
| `item_description` | 76,925 | 1,566,190 |

기존 캡(`name` 20,000 / `desc` 30,000)은 unigram만으로도 이미 넘는 값이라 **확실히 binding**입니다.
하지만 LGBM(native categorical + ngram) 기준으로 캡을 실제로 풀어봤을 때:

| config | max_features | dims | RMSLE |
|---|---|---|---|
| D | 20k / 30k (노트북 14번과 동일) | 50,007 | 0.47320 |
| E | 40k / 60k | 100,007 | 0.47288 |
| F | uncapped (min_df=2만) | 1,968,091 | 0.47293 |

**결론**: dims를 5만 → 약 200만(40배)까지 늘려도 RMSLE는 사실상 평평합니다(D→E→F 변화폭 0.0004,
노이즈 수준). "vocab이 작다"는 직관은 사실이었지만(unigram 어휘가 이미 캡을 초과), **그 binding이
성능에 미치는 실질 영향은 거의 없음** — 앞으로 `max_features`를 더 키울 필요는 없습니다.

### 9-2. 개선의 원천 분리 — 카테고리 인코딩 vs n-gram 확장

14번 섹션의 `0.49220 → 0.47363` 개선은 (a) OHE→LGBM native categorical 인코딩, (b) unigram→ngram(1,2)
확장이 동시에 적용된 결과입니다. 둘을 분리해서 재측정(캡은 20k/30k 고정):

| config | 카테고리 인코딩 | 텍스트 | dims | RMSLE |
|---|---|---|---|---|
| A (원본 baseline 재현) | OHE | unigram | 55,807 | 0.49180 |
| B (카테고리만 교체) | **native** | unigram | 50,007 | **0.47409** |
| C (텍스트만 교체) | OHE | **ngram** | 55,807 | 0.49093 |
| D (노트북 14번 재현) | native | ngram | 50,007 | 0.47320 |

**B(0.47409)가 D(0.47320)에 거의 근접** → 개선의 대부분(~97%)이 **카테고리 인코딩 교체**에서 왔고,
ngram 확장의 순수 기여(C vs A: `0.49180→0.49093`, -0.00087)는 미미합니다. 캡을 그대로 둔 채 bigram만
추가하면 빈도 기준 top-K 캡에서 unigram이 슬롯을 거의 다 차지해 bigram이 들어갈 자리가 부족하기
때문으로 보입니다(9-1에서 실제 bigram 어휘가 40만+임을 확인).

### 9-3. 새 피처 — 길이 피처 + target encoding *(노트북 17번 섹션)*

브랜드/세부카테고리별 가격 통계, 텍스트 길이 등 파생 피처를 추가:

| 피처 | 내용 | leakage 방지 |
|---|---|---|
| `name_len_words`, `name_len_chars` | 상품명 단어수/글자수 | 타깃 미사용, 문제 없음 |
| `desc_len_tokens` | 설명 토큰 개수 | 동일 |
| `has_brand` | 브랜드 존재 여부(이진) | 동일 |
| `brand_te` | 브랜드별 평균 `log1p(price)` (target encoding) | **학습 데이터에서만** 통계 계산, count 기반 스무딩(`k=10`)으로 희귀 브랜드 과적합 방지, 미관측 값은 global mean으로 대체 |
| `cat_so_te` | 소분류(`cat_so`)별 평균 `log1p(price)` | 동일 방식 |

```python
def target_encode(train_col, train_y, full_col, k=10):
    global_mean = train_y.mean()
    stats = pd.DataFrame({"col": train_col.values, "y": train_y}).groupby("col")["y"].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + k * global_mean) / (stats["count"] + k)
    return full_col.map(smoothed).fillna(global_mean).values
```

새 피처(20k/30k 캡, 기존 하이퍼파라미터 고정) 적용 효과:

| 모델 | 새 피처 없음 | 새 피처 적용 | 개선폭 |
|---|---|---|---|
| Ridge | 0.46954 | 0.46853 | -0.00101 |
| LGBM | 0.47320 | 0.47022 | -0.00298 |

LGBM이 Ridge보다 약 3배 더 큰 이득을 봤습니다 — target encoding이 만드는 비선형 신호(가격대 threshold
등)를 트리 모델이 더 잘 활용하는 것으로 해석됩니다.

### 9-4. FM_FTRL 직접 구현 (numpy/scipy) *(노트북 18번 섹션)*

**배경**: Kaggle 공개 커널 중 `Wordbatch`(FTRL + FM_FTRL + LightGBM 블렌드)가 이 대회에서 LB
**0.42555**를 기록한 바 있습니다 ([참고](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555)).
참고로 이 대회 역대 1위(Sparse MLP 앙상블)는 RMSLE **0.3875**입니다
([참고](https://github.com/pjankiewicz/mercari-solution)) — "0.3까지 내려간다"는 속설은 근거가 없었습니다.

**환경 제약**: `wordbatch` 패키지(FM_FTRL 제공)는 Python 3.13용 사전빌드 wheel이 없고, 이 환경엔
C 컴파일러(`cl.exe`/`gcc`)도 없어 설치 자체가 불가능했습니다. 대신 같은 알고리즘을 numpy/scipy만으로
직접 구현:

- **선형항 `w`**: McMahan et al., *"Ad Click Prediction: a View from the Trenches"* (2013)의
  FTRL-proximal 업데이트를 그대로 사용 (좌표별 적응 학습률 + L1/L2).
- **2차 상호작용항 `V`(FM, latent factor `k=8`)**: 표준 FM 2차항 트릭
  `0.5 * sum_f[(sum_i v_if x_i)^2 - sum_i v_if^2 x_i^2]`을 미니배치 전체에 대해
  **sparse @ dense 행렬곱으로 벡터화**(행 단위 Python 루프 없음)하고, AdaGrad로 학습.
- 미니배치(5만 행)마다 `X_b.T @ (G*S) - (X_b^2)^T@G * V` 형태로 그래디언트를 계산 — nnz에 비례하는
  비용이라 148만 행 규모에서도 epoch당 수 초 수준으로 충분히 빠름.
- 합성 데이터(실제 FM 구조를 가진 랜덤 데이터)로 스모크 테스트 → train/test loss가 발산·NaN 없이
  단조 감소함을 확인 후 실데이터에 적용.

Wordbatch의 Cython/해싱 최적화 구현과 동일하지는 않지만(속도 최적화가 덜 됨), 수학적으로는 같은
모델(선형 + FM 상호작용, FTRL류 좌표별 적응 학습률)입니다.

### 9-5. 최종 leakage-free 파이프라인 + 블렌딩 *(노트북 19~22번 섹션 — 개선의 실질적 핵심 구간)*

**모델별 피처 표현** (모델마다 자기에게 유리한 인코딩을 그대로 사용 — 블렌딩은 예측값 레벨에서만 결합):

| 모델 | 텍스트 | 카테고리 | 새 피처 | dims |
|---|---|---|---|---|
| Ridge, FM_FTRL | unigram (20k/30k) | OneHotEncoder | 표준화(StandardScaler) | 55,813 |
| LGBM | ngram(1,2) (20k/30k) | LabelEncoder + `categorical_feature` | raw | 50,013 |

**2-stage 검증 절차** (test set은 마지막에 딱 한 번만 사용):

1. **Stage 1 (튜닝)**: `train`(68%)으로만 학습 → `valid`(12%)로 Ridge `alpha` 그리드, LGBM
   `early_stopping_rounds`, FM_FTRL epoch 수, 3모델 블렌드 가중치(0.05 단위 그리드)를 전부 결정.
   `target_encode`/`StandardScaler`도 이 stage에서는 `train`에만 fit.
2. **Stage 2 (최종 평가)**: Stage 1에서 정한 하이퍼파라미터/epoch 수 그대로 `train+valid`(80%) 전체로
   재학습(`target_encode`/`StandardScaler`도 `train+valid`로 재fit) → `test`(20%)에 **딱 한 번** 예측 →
   Stage 1에서 정한 블렌드 가중치를 그대로 적용(재탐색 없음).

**LGBM 튜닝 결과**: `n_estimators` 상한 1200(early stopping) · `learning_rate=0.05` ·
`num_leaves=180` · `min_child_samples=30` · `reg_alpha=reg_lambda=0.1` ·
`feature_fraction=0.7` · `bagging_fraction=0.8`, `bagging_freq=1` → valid에서 1200라운드까지 계속
개선되어 `best_iteration=1200`(상한 도달, 더 큰 `n_estimators`면 추가 개선 여지 있음).

**최종 결과 (test, leakage-free — 노트북 17~22번 섹션 실제 실행 결과)**

| 모델/조합 | test RMSLE |
|---|---|
| Ridge (alpha=3) | 0.46971 |
| FM_FTRL (60 epoch) | 0.44584 |
| LGBM (튜닝, 1200 라운드) | 0.42881 |
| 블렌드 (동일가중 1/3,1/3,1/3) | 0.42974 |
| **블렌드 (valid 최적가중치 ridge=0.0 / lgbm=0.65 / fm=0.35)** | **0.42114** |

`alpha`=1/3/10 그리드에서 Ridge는 여전히 3이 최적, 최적 블렌드는 **Ridge 가중치 0** — LGBM을 제대로
튜닝하고 FM_FTRL을 섞으면 Ridge는 더 이상 블렌드에 기여하지 않습니다(14~16번 섹션 시점엔 Ridge가
LGBM보다 우수했지만, LGBM을 튜닝한 뒤로는 역전됨). **`0.49220`(11번 섹션 LGBM 최초 baseline)에서
`0.42114`까지, RMSLE 기준 약 14.4% 개선.**

### 9-6. n_estimators 상한 확장 (1200 → 2000) *(노트북 23번 섹션)*

9-5의 LGBM이 1200라운드 상한에서 "Did not meet early stopping"으로 끝났고 직전 200라운드 구간에서도 계속
개선 중이었던 점(상한이 진짜 수렴점이 아니었을 가능성)을 확인하기 위해, 처음부터 재학습하지 않고 이미 학습된
1200그루 부스터를 `init_model`로 이어 학습(체크포인트 이어학습)해 총 2000그루까지 확장했습니다. 블렌드 가중치는
9-5에서 valid로 정한 값(ridge=0.0/lgbm=0.65/fm=0.35)을 재탐색 없이 그대로 재사용:

| 모델/조합 | 1200 (9-5) | 2000 (이어학습) | 개선폭 |
|---|---|---|---|
| LGBM 단독 | 0.42881 | **0.42388** | -0.00493 |
| 블렌드 (동일가중) | 0.42974 | 0.42764 | -0.00210 |
| **블렌드 (기존 가중치 재사용)** | 0.42114 | **0.41778** | **-0.00336** |

참고로 test에서 블렌드 가중치를 직접 재탐색하면 `ridge=0/lgbm=0.7/fm=0.3`, RMSLE `0.41772`로 나와 기존
가중치(0.41778)와 거의 차이가 없었습니다 — 9-5에서 valid로 정한 가중치가 라운드 수가 바뀌어도 잘 버티는
견고한 값이었다는 뜻입니다. `n_estimators`를 더 늘리면(예: 3000+) 추가 개선 여지가 있는지는 아직 미검증.
