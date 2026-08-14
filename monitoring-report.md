# monitoring 프로필 기동 리포트 (5차 루프)

- `docker compose --profile monitoring up -d` — prometheus(9090)·grafana(3001) 기동.
- prometheus: `/-/ready` 게이트 판정 · 수집 대상 = backend `/metrics`(prometheus.yml 프로비저닝).
- grafana: `/api/health` 게이트 판정 · 프로비저닝 대시보드 탑재(`monitoring/grafana/provisioning`).
- 기본 자격(admin/admin)은 하드닝 리포트에서 판정.

## RESULT
monitoring 프로필 기동·관측 게이트 통과 시 REQ-033 잔여("프로필 미기동") 해소.
