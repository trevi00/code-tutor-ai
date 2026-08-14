---
origin: authored
authored_by: claude (onboarding-coverage evaluator)
source_commit: 370caeaf1141
---

# 온보딩 리포트 — code-tutor-ai (D-014 reverse 완주)

## 커버리지 (기계 판정)

- requirement→AC 커버리지: **PASS** (REQ 22건 전부 usecase 추적)
- AC→EP 대응: **PASS** (AC 17건 — EP 실존 인용 16 + EP-N/A 명시 1)
- usecase EP 참조 실존: **PASS** (유령 EP 0)
- 구조 역추출: EP 120 · 테이블 30 · 컴포넌트 엣지 32, round-trip 멱등 verify OK

## 정직 잔여 (숨기지 않음 — 알려진 결함·미검증)

1. **순환 의존 위반 (REQ-031)**: `execution → learning → execution` — 역추출이 검출한 실물 결함. P1 개발 루프의 첫 수리 대상 후보.
2. **스키마 드리프트 추정 (REQ-032)**: alembic 리비전 1개 vs 모델 30개 — 스키마 소스는 models.py 로 간주하고 마이그레이션 정합은 미검증.
3. **테스트 36파일 통과 여부 미검증 (REQ-030)**: 아직 한 번도 하네스 게이트로 실행되지 않았다.
4. **P2 유예 11건**: usecase 상세 복원 유예 (REQ-013~023) — coverage 는 등록 수준.
5. **행위 구획(R3, Gherkin 상세) 미착수**: D-014 의 llm 추정 구획은 후속.
6. pending 그래프 질의 5종(명사 개념·상태 엔티티 등)은 백엔드 미구축 — 해당 게이트는 이 프로젝트에서 미적용.

## RESULT

온보딩 완결 조건 충족 — 구조(extracted)·의도(authored·승인)·커버리지(기계 PASS).
ready_for_kickoff 승인 시 이 프로젝트는 하네스 관리 하에 들어가며,
다음 루프는 P1 핵심 축(가입→문제→제출→채점)의 프로덕션급 인상이다.
