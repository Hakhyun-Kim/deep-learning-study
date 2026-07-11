# CLAUDE.md

ResNet18로 Stanford 40 Actions(행동 40클래스) test acc 80%를 목표로 하는
**딥러닝 입문 학습 프로젝트**. 코드 작성만큼 "학습자가 이해하는 것"이 목적이다.

## 설명 스타일 (중요)

- 학습자는 딥러닝 **초보**. 용어는 처음 나올 때 한 줄로 풀어서 쓰고, 비유를 곁들인다.
- **테슬라 자율주행(FSD) 예시를 적극 활용할 것.** 학습자의 관심 분야라 흥미와
  이해에 큰 도움이 된다. 새 개념을 설명할 때 "테슬라에서는 이게 ~에 해당"
  식의 연결을 우선 시도한다. (예: 지도학습 ↔ 모방학습, 간접 신호 ↔ 개입/섀도 모드,
  분포 변화 ↔ 새 도시 진출, 데이터 증강 ↔ 다양한 날씨·카메라 데이터)
  참고: [notes/ch1/q06](notes/ch1/q06-tesla-rule-to-e2e.md)
- 설명은 한국어. 항상 우리 코드(`src/`)의 실제 줄·실험(exp01 등)의 실제 숫자와
  연결해서 설명한다.
- 다이어그램은 mermaid로 문서 안에 직접 넣는다 (GitHub 자동 렌더링).

## 공부 노트 규칙 (notes/)

- 챕터별 폴더: `notes/chN/` 안에 `README.md`(개요·체크리스트), `concepts.md`(개념 사전),
  `notes.md`(코드 따라가기), `qNN-제목.md`(질문 노트).
- **질문 중심 학습**: 질문 1개 = 파일 1개. 학습자가 스스로 생각할 문제는
  답을 바로 쓰지 않고 힌트만 주거나, 해설을 `<details>` 접기 안에 넣는다.
- 질문을 추가/답변하면 `notes/README.md`와 해당 챕터 `README.md`의 목록도 갱신.

## 프로젝트 구조·워크플로

- `src/train.py` 학습 스크립트 (`--smoke`로 빠른 동작 확인), `src/dataset.py` 데이터셋,
  `src/metrics_logger.py` 대시보드 로거.
- 실험 지표는 `docs/metrics/*.json` → GitHub Pages 대시보드
  (https://hakhyun-kim.github.io/deep-learning-study/).
- 학습 환경은 로컬 Windows + RTX 3050 Ti 4GB — 이 원격 환경에서는 학습을 돌릴 수
  없으니 학습 명령은 사용자에게 안내만 한다.
- **머지는 PR 없이 main에 직접** (사용자 선호). 원격 main이 로컬에서 갱신되는
  경우가 있으니 머지 전 `git fetch origin main` 필수.

## 실험 현황

- exp01_fc_only: 백본 얼림, test acc 69.4% (완료)
- exp02_full_finetune: 전체 파인튜닝 lr 1e-4 (예정, 목표 80% 근접)
