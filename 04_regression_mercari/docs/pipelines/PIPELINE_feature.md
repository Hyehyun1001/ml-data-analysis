# Mercari — 파생피처 실험 계획 (`소현_mercari_feature.ipynb`)

base(`PIPELINE_base.md`)의 전처리·벡터화·인코딩을 그대로 가져온 뒤, 그 위에 새 파생피처를 얹어 RMSLE
추가 개선을 시도하는 문서. base의 6개 파생피처(`name_len_words/chars`, `desc_len_tokens`, `has_brand`,
`brand_te`, `cat_so_te`)와는 겹치지 않는 것들만 다룬다.

## 전제

- 전처리/벡터화 재료(`X_name`, `X_descp`, `X_cat[...]`, `X_num`)는 base와 동일하게 재사용
- target을 쓰는 피처는 반드시 train 쪽 통계만으로 계산 (leakage-free, base 9-3/9-5 방식)
- `random_state=156` 고정, base와 동일 test set 기준으로 비교

## 우선순위 높음

1. **상태 키워드 신호** — description에서 `"brand new"`/`"nwt"`/`"worn"`/`"flaw"`/`"authentic"`/`"vintage"` 등
   정규식 매칭 → 이진/카운트 피처. `item_condition_id`(자기신고) 보완용.
2. **브랜드 언급 텍스트 복구** — `brand_name == Other_Null`인 행에서 name/description에 알려진 브랜드명이
   등장하는지 감지.
3. **본문 언급 가격 추출** — `"$숫자"` 패턴(예: `retail $50`)을 정규식으로 뽑아 수치 피처화.
4. **브랜드×카테고리 조합 target encoding** — `(brand_name, cat_so)` 조합 키로 스무딩 TE (base는 각각 따로 TE).

## 실험적 / 보류

- 텍스트 클러스터링 → 토픽 ID (TF-IDF+SVD 후 KMeans, 계산비용 있음)
- 최근접 이웃(kNN) 가격 프록시 — leakage 리스크 높아 base 9-5식 2-stage 검증 필수

## 백로그 (미착수)

- 표기 스타일 신호 (ALL CAPS/느낌표/이모지 카운트)
- name-description 중복도 (자카드 등)
- 브랜드 빈도수(popularity)
- 카테고리 내 브랜드 상대순위(럭셔리 지수)
- 문자 n-gram(`char_wb`) 벡터화 — 브랜드/이름 오탈자·표기 변형 흡수

## 진행 상황

- 미착수. 노트북(`소현_mercari_feature.ipynb`)은 현재 빈 파일 — 우선순위 항목부터 구현 예정.
