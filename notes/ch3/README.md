# ch3 — 분석과 개선 (모델을 뜯어보기)

exp02로 75.7%까지 왔다. 남은 4.3%p(목표 80%)를 **아무 기법이나 시도해서가 아니라,
오답을 읽어서** 메꾸는 챕터. ch1·ch2가 "모델 만들고 돌리기"였다면 ch3는
**"돌린 모델을 진단하고, 진단에 맞는 처방을 하기"**다.

핵심 도구는 **Grad-CAM**(모델이 어디를 보고 판단했는지 열지도)과 **오답 분석**
(클래스별 정확도·혼동행렬). 진단 없이 처방하지 않는다 — 그게 ch3의 규율.

## 읽는 순서

| 순서 | 파일 | 내용 | 언제 읽나 |
|---|---|---|---|
| 1 | [concepts.md](concepts.md) | **개념 사전** — 혼동행렬·Grad-CAM·지름길 학습·증강 메뉴 | 진단 돌리기 전에 |
| 2 | [notes.md](notes.md) | **진단 결과 + exp03 설계** — 실측 숫자, 처방, 예상 적기 | 진단 전·중·후 |
| 3 | 아래 질문 목록 | **질문 노트** | 복습할 때 |

## 진단 도구 (this chapter)

| 스크립트 | 무엇 | 출력 |
|---|---|---|
| [`scripts/analyze_errors.py`](../../scripts/analyze_errors.py) | 클래스별 정확도·혼동 쌍·확신하며 틀린 예 | 콘솔 텍스트 |
| [`scripts/gradcam.py`](../../scripts/gradcam.py) | 모델이 어디를 보고 예측했나 열지도 | `runs/gradcam/*.png` (gitignore) |
| [`scripts/position_sensitivity.py`](../../scripts/position_sensitivity.py) | 위치별(모서리/중앙) 정확도 + TTA | 콘솔 텍스트 |

```powershell
python scripts/analyze_errors.py     # 숫자 진단
python scripts/gradcam.py            # 그림 진단 (한글 깨지면 $env:PYTHONUTF8=1; 앞에)
```

## 할 일 체크리스트

- [x] exp02 진단 도구 작성·실행 (`analyze_errors.py`, `gradcam.py`)
- [x] 진단 결과 [notes.md 2단계](notes.md#2단계-진단-결과-2026-07-13-실측-exp02)에 기록
      (약한 클래스·혼동 쌍·Grad-CAM 3대 사례)
- [ ] [q09](q09-shortcut-learning.md) "스스로 생각해보기"에 내 답 적어보기
- [x] exp03 후보 중 **후보 A(증강 강화)** 선택 (예상 미기입 — exp04에선 먼저 적기)
- [x] exp03 실행 → `analyze_errors.py`로 재진단, before/after 비교
      ([notes.md 결과 기록](notes.md#결과-기록-2026-07-15-실측-exp03_aug_stronger):
      75.05%, 과적합↓·약한 클래스 일부↑, 큰 배경 지름길은 생존)
- [x] 대시보드에 exp03 곡선 push
- [x] exp04: 증강 유지 + 에폭 20 → best **75.80%** (ep12), ep13부터 과적합
      ([notes.md exp04 결과](notes.md#exp04-결과-2026-07-15-실측-증강-유지--에폭-20))
- [x] 곁가지 진단: 위치 민감도(중앙 vs 모서리 **+7.1%p**) + **TTA 77.77%**
      ([notes.md 곁가지 진단](notes.md#곁가지-진단-위치-민감도-fivecrop--tta--2026-07-15), q12)
- [ ] 다음 갈림길: TTA를 최종 카드로 확정할지, person-crop(ch4급) 등 더 갈지 결정

## 질문 목록

| # | 질문 | 상태 | 관련 |
|---|------|------|------|
| [q09](q09-shortcut-learning.md) | 배경(글씨·접시) 보고 행동 맞히면, 정확도는 높은데 왜 문제? | 🤔 스스로 생각해보기 · 해설 있음 | exp02 Grad-CAM, `gradcam.py` |
| [q10](q10-gradcam-mechanism.md) | Grad-CAM은 "모델이 본 이미지 부분"을 어떻게 찾아내나? | 🤔 스스로 생각해보기 · 해설 있음 | `gradcam.py` (특징 지도 × 기울기) |
| [q12](q12-center-bias-spatial-prior.md) | 모델은 가운데를 "먼저" 보나? 중앙 편향은 어떤 형태인가? | ✅ 실측(+7.1%p)으로 확인 | `position_sensitivity.py`, TTA 77.77% |
| [q11](q11-pytorch-speed-and-stack.md) | PyTorch는 빠른가? 실무는 뭘 쓰나? 소스를 직접 고칠 수 있나? | ✅ 정리됨 | `train.py` `backward()`, ch4(배포) |

## ch3 졸업 체크리스트

- [ ] 전체 정확도 한 숫자와 클래스별 정확도가 왜 다른 이야기인지 설명할 수 있다
- [ ] Grad-CAM이 무엇을 재는지, 정확도와 뭐가 다른지 안다 (맞혔나 vs 맞는 이유로 맞혔나)
- [ ] 우리 모델의 오답을 3종류(배경 지름길·덜 수렴·미세분류)로 분해할 수 있다
- [ ] 지름길 학습이 왜 배포에서 위험한지 테슬라 예시로 설명할 수 있다
- [ ] "진단 → 한 변수 처방 → 재진단" 루프를 exp03에서 한 바퀴 돌려봤다

## 다음 챕터 (예정)

- **ch4 — 마무리**: 목표 80% 도달 여부 정리, 최종 모델 선택 근거, 배포 시 한계
  (미세분류·지름길이 남았다면 그 정직한 기록)
