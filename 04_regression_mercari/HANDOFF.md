# HANDOFF — Mercari 작업 인수인계

마지막 갱신: 2026-08-31 · 브랜치 `KSH` · 규칙은 `CLAUDE.md` 참고.

## 보고서 작업 현황 (report_초안_C, 2026-08-31)

- **활성 파일은 `report_초안_C.md`/`.pdf`뿐이다.** `report_초안_A.*`, `report_초안_B.*`는 동결됨 — 절대 수정 금지(`REPORT_GUIDE.md` 0절 참고). 매 수정 후 `git diff --stat -- report_초안_A.md report_초안_A.pdf report_초안_B.md report_초안_B.pdf`가 빈 출력이어야 한다.
- 오늘 작업한 내용: 팀원 균형(소현이 항상 먼저 나오는 문제 전면 수정), 5장(Feature Engineering)에서 혜현·유경·소현 구간 보강, 2.1절 보강, 9장(팀원별 실험 상세)을 코드 없이 핵심만 남기고 팀원 간 분량을 균등화(팀원당 시각화 1개, 소현도 FM_FTRL epoch 곡선을 노트북 실제 로그로 새로 생성해 추가), 8.4 표를 RMSLE 내림차순 단일 기준으로 재정렬, `report_style.css`의 `text-align: justify`→`left` 수정(한글+영문 혼합 텍스트에서 word-spacing 왜곡 버그).
- 재현 파이프라인: `pandoc report_초안_C.md -f markdown-implicit_figures -t html5 -s -c report_style.css -o report_초안_C.html` → Chrome headless `--print-to-pdf`(정확한 명령은 세션 히스토리 또는 `REPORT_GUIDE.md` 참고). PDF 재생성 후 PyMuPDF로 페이지별 fill% 감사 필수(`REPORT_GUIDE.md` 6절 체크리스트).
- 오늘 확정된 새 규칙은 전부 `REPORT_GUIDE.md`에 반영함(팀원 표 순서 원칙, 9장 코드-없음/이미지 1개/분량 균등 규칙, `text-align:left` 규칙, 이미지는 재사용보다 원본 노트북 데이터 기반 신규 생성 우선).
- **다음 세션에서 이어받을 것**: 사용자가 추가 피드백을 줄 가능성이 높음(이번 세션도 스크린샷 기반 미세 조정이 여러 차례 있었음) — `report_초안_C.pdf`를 페이지 단위로 다시 렌더링해 확인 후 시작. 아직 보고서를 최종본으로 확정(freeze)하라는 지시는 없었음.

## 노트북/파이프라인 작업 현황 (base/feature)

- **base 파이프라인: 완료.** `teamipynb/소현_mercari_base.ipynb`에 전처리~인코딩~모델링~블렌딩까지 전부
  구현되어 있고, 전 과정이 `docs/pipelines/PIPELINE_base.md`에 문서화됨. 최종 RMSLE **0.41778**.
- **feature 노트북: 미착수.** `teamipynb/소현_mercari_feature.ipynb`는 현재 빈 파일(0 byte).
  계획은 `docs/pipelines/PIPELINE_feature.md`에 우선순위별로 정리해둠 (아직 코드 없음, 계획 단계).

## 확정된 결정 (Do not reconsider)

아래 두 가지는 이미 확정된 결정입니다. 새로 세션을 이어받은 Claude는 이걸 다시 논의 대상으로 꺼내거나
다른 방식을 제안하지 마세요 — 그대로 따르세요.

1. **base/feature 노트북-문서는 1:1 매핑, 절대 섞지 않는다.**
   `소현_mercari_base.ipynb` ↔ `PIPELINE_base.md`, `소현_mercari_feature.ipynb` ↔ `PIPELINE_feature.md`.
2. **feature.ipynb는 base의 전처리/벡터화 재료를 그대로 재사용한다.**
   `X_name`/`X_descp`/`X_cat[...]`/`X_num`을 다시 만들지 않고, base 산출물 위에 새 피처만 얹는다.

이 외의 항목(우선순위 순서, 어떤 파생피처를 최종 채택할지 등)은 아직 확정이 아니며 논의/조정 가능합니다.

## Next Exact Action

`PIPELINE_feature.md` "우선순위 높음" 1번 — **상태 키워드 신호**부터 구현 착수:

1. `item_description` 정제 토큰(base 2-7단계 결과)에서 매칭할 키워드 목록 정의
   (`brand new`, `nwt`, `worn`, `flaw`, `defect`, `authentic`, `vintage` 등 — 확정 리스트는 아니므로
   구현하면서 EDA로 추가/삭제 가능).
2. 키워드별 이진 플래그 또는 총 매칭 카운트 컬럼 생성 (base의 `null_penalty` 만드는 방식과 동일 패턴).
3. base `X_features`에 이 컬럼(들)만 추가한 버전으로 Ridge 1회 학습 → RMSLE를 base baseline(0.46971,
   `PIPELINE_base.md` 9-5 표)과 비교해 순수 효과 측정.
4. 결과를 `PIPELINE_feature.md`에 표로 기록 (base 9-3 형식 참고) 후, 우선순위 2번(브랜드 언급 텍스트
   복구)으로 이동.

## 다음 할 일 (전체 순서 요약)

1. `소현_mercari_feature.ipynb`를 base 노트북 기반으로 시작 (전처리/벡터화 재료 재사용).
2. `PIPELINE_feature.md`의 "우선순위 높음" 4개(상태 키워드 신호 / 브랜드 언급 텍스트 복구 / 본문 언급 가격
   추출 / 브랜드×카테고리 조합 target encoding)부터 구현.
3. 구현하면서 `PIPELINE_feature.md`를 base와 같은 수준(코드+결과 RMSLE 표)으로 갱신.
4. **주의**: base 관련 내용은 `PIPELINE_base.md`에만, feature 관련 내용은 `PIPELINE_feature.md`에만 —
   두 문서를 섞지 않는다 (`CLAUDE.md` 참고).

## 참고사항

- `data/mercari_train.tsv`는 git에 없음(레포 공통 `.gitignore`) — 로컬에 직접 받아야 노트북 실행 가능.
- 이전에 있던 루트 `Mercari_validate.ipynb`는 삭제됨(커밋 `e78a229`에 스냅샷으로 남아있음, 필요시
  `git show e78a229:04_regression_mercari/Mercari_validate.ipynb`로 복구).
- `docs/reports_claude/report_draft_01.md`는 현재 빈 파일 — 미사용.
