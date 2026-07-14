# 공부 노트

챕터(프로젝트 단계)별로 폴더를 나누고, 각 챕터 안에 세 종류의 문서를 둔다:

| 종류 | 파일 | 역할 |
|---|---|---|
| 개념 설명 | `chN/concepts.md` | 그 챕터에 꼭 필요한 용어를 한 번에 정리한 사전. 강의 영상 대신/보완용 |
| 학습 노트 | `chN/notes.md` | 개념이 실제 코드·실험에서 어떻게 쓰였는지 흐름 따라가기 |
| 질문 노트 | `chN/qNN-제목.md` | 공부하다 생긴 질문 1개 = 파일 1개 |

## 질문 중심 학습

1. 막히거나 궁금한 게 생기면 → 질문을 파일로 등록 (`qNN-제목.md`)
2. 답을 찾으면 (직접 실험, 강의, Claude에게 질문) → 파일에 정리
3. 나중에 파일 제목(질문)만 보고 답을 떠올려보는 게 복습

## 챕터 목록

| 챕터 | 주제 | 상태 |
|---|---|---|
| [ch1](ch1/README.md) | 전이학습 베이스라인 — 딥러닝 기본 개념 + exp01 (69.4%) | 완료 (q02 내 답 채우기 남음) |
| [ch2](ch2/README.md) | 전체 파인튜닝 — exp02 75.7% / exp02b 55.0% (q03 검증) | 실험·질문 완료, 졸업 체크리스트 남음 |
| [ch3](ch3/README.md) | 분석과 개선 — Grad-CAM으로 exp02 진단(지름길 학습 발견), exp03 설계 | 진단 완료, exp03 실행 남음 |

## ch1 질문 목록

| # | 질문 | 상태 | 관련 |
|---|------|------|------|
| [q01](ch1/q01-backbone-freeze.md) | 백본 얼림(freeze)이 무슨 뜻? | ✅ 정리됨 | exp01, `train.py` |
| [q02](ch1/q02-why-test-acc-higher.md) | 왜 test_acc가 train_acc보다 높지? (보통은 반대인데) | 🤔 스스로 생각해보기 · 해설 있음 | exp01 학습 곡선 |
| [q03](ch1/q03-full-finetune-low-lr.md) | 전체 파인튜닝은 왜 학습률을 10배 낮추나? | ✅ exp02 vs exp02b로 확인 | exp02, exp02b |
| [q04](ch1/q04-eval-loss-labels.md) | 평가 때 손실은 어떻게 계산하지? 정답이 없는데? | ✅ 정리됨 | `train.py` `evaluate()` |
| [q05](ch1/q05-deploy-to-real-world.md) | 정답지 없는 현실 데이터에는 어떻게 적용하나? | ✅ 정리됨 | q04에서 이어짐 |
| [q06](ch1/q06-tesla-rule-to-e2e.md) | 테슬라는 어떻게 룰베이스에서 E2E로 갔나? | ✅ 정리됨 | q05에서 이어짐 |
| [q10](ch1/q10-learning-paradigms-and-llms.md) | 지도·자기지도·강화학습 차이와, Claude는 어떤 학습으로 만드나? | ✅ 정리됨 | q06에서 이어짐, exp02 전이학습 |
| [q11](ch1/q11-why-llms-improve-so-fast.md) | 왜 요즘 LLM은 3개월 단위로 빠르게 좋아지나? 질답 데이터 축적 때문? | ✅ 정리됨 | q10에서 이어짐 |

## ch2 질문 목록

| # | 질문 | 상태 | 관련 |
|---|------|------|------|
| [q03](ch1/q03-full-finetune-low-lr.md) | 전체 파인튜닝은 왜 학습률을 10배 낮추나? | ✅ exp02 vs exp02b로 확인 | exp02, exp02b |
| [q07](ch2/q07-why-slower-epochs.md) | 학습 파라미터가 546배인데 에폭 시간도 546배 걸릴까? | ✅ 실측 1.5배 — 이유 정리됨 | exp02 실측 |
| [q08](ch2/q08-lr-big-relative-to-what.md) | lr 1e-3이 "큰 발자국"이라는 건 어떻게 아나? 상대적인 값 아닌가? | ✅ 가중치 이동량 실측으로 확인 | exp02 vs exp02b |

## ch3 질문 목록

| # | 질문 | 상태 | 관련 |
|---|------|------|------|
| [q09](ch3/q09-shortcut-learning.md) | 배경(글씨·접시) 보고 행동 맞히면, 정확도는 높은데 왜 문제? | 🤔 스스로 생각해보기 · 해설 있음 | exp02 Grad-CAM, `gradcam.py` |
| [q10](ch3/q10-gradcam-mechanism.md) | Grad-CAM은 "모델이 본 이미지 부분"을 어떻게 찾아내나? | 🤔 스스로 생각해보기 · 해설 있음 | `gradcam.py` (특징 지도 × 기울기) |
| [q11](ch3/q11-pytorch-speed-and-stack.md) | PyTorch는 빠른가? 실무는 뭘 쓰나? 소스를 직접 고칠 수 있나? | ✅ 정리됨 | `train.py` `backward()`, ch4(배포) |
