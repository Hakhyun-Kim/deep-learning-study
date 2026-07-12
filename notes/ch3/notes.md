# ch3 실험 가이드 — exp02 진단하고, exp03로 처방하기

> ch2까지는 "모델을 더 잘 학습시키는" 이야기였다. ch3는 **이미 학습된 exp02(75.7%)를
> 뜯어보고**, 오답이 말해주는 방향으로 다음 한 걸음(exp03)을 고른다.
> 이 문서 = 진단 결과 기록 + 처방(실험) 설계.

## 1단계: 진단 도구 돌리기

두 스크립트 모두 exp02 체크포인트(`checkpoints/exp02_full_finetune_best.pt`)를
읽어 test set을 추론한다. 학습이 아니라 추론이라 **1~2분**이면 끝난다.

```powershell
# (1) 숫자로: 약한 클래스 · 혼동 쌍 · 확신하며 틀린 예
python scripts/analyze_errors.py

# (2) 그림으로: 모델이 어디를 보고 틀렸나 (runs/gradcam/*.png 로 저장)
python scripts/gradcam.py
```

> Windows 콘솔에서 한글/기호가 깨지면 앞에 `$env:PYTHONUTF8=1;` 를 붙인다.
> Grad-CAM 결과 이미지는 `data/`처럼 gitignore 대상이라(원본 사진 비재배포)
> 저장소엔 안 올라간다 — 로컬에서 열어 본다.

## 2단계: 진단 결과 (2026-07-13 실측, exp02)

### 클래스별 정확도 — 격차가 56%p

전체 75.65% 뒤에 숨어 있던 지형:

| 약한 클래스 (하위) | 정확도 | | 강한 클래스 (상위) | 정확도 |
|---|---|---|---|---|
| phoning | **39.0%** (62/159) | | riding_a_horse | 95.4% |
| waving_hands | 40.0% (44/110) | | walking_the_dog | 93.3% |
| texting_message | 40.9% (38/93) | | riding_a_bike | 92.7% |
| pouring_liquid | 48.0% | | climbing | 92.3% |
| taking_photos | 48.5% | | rowing_a_boat | 91.8% |
| smoking | 51.8% | | | |

한눈에 보이는 규칙: **강한 클래스 = 큰 물체 + 독특한 전신 포즈 + 특징적 배경**
(말·자전거·보트). **약한 클래스 = 작은 물건 + 비슷한 상반신 포즈**
(전화·문자·담배는 다 "손이 얼굴 근처").

### 혼동 쌍 — 헷갈리는 짝이 죄다 "의미상 이웃"

| 정답 → 오답 | 횟수 | 왜 헷갈릴까 |
|---|---|---|
| writing_on_a_book → reading | 22 | 둘 다 "책을 내려다보는 사람" |
| cooking → washing_dishes | 19 | 둘 다 "싱크대/조리대 앞 손동작" |
| phoning → smoking | 18 | 둘 다 "손이 얼굴 근처" |
| drinking → brushing_teeth | 17 | 둘 다 "손을 입으로" |
| feeding_a_horse → riding_a_horse | 12 | **같은 물체**(말), 다른 동작 |

무작위로 틀리는 게 아니라 **비슷한 것끼리** 틀린다. 이게 "모델이 대충은
이해했지만 마지막 미세한 구분에서 진다"는 증거.

### Grad-CAM — 확신하며 틀린 3장이 같은 병을 가리킴

`analyze_errors.py`의 "가장 확신하며 틀린 예"를 Grad-CAM으로 열어보니
(전부 확신 98~100%), 셋 다 **사람의 행동이 아니라 배경/맥락 물체**를 보고 있었다:

| 파일 | 정답 → 예측(확신) | 열지도가 켜진 곳 |
|---|---|---|
| `phoning_253.jpg` | phoning → writing_on_a_board (99%) | 뒤 **벽에 적힌 글씨** — 전화기가 아니라 |
| `reading_076.jpg` | reading → writing_on_a_board (100%) | 뒤 **칠판의 글씨** — 손의 책이 아니라 |
| `watching_TV_017.jpg` | watching_TV → cooking (98%) | 앞 **식탁의 접시들** — TV가 아니라 |

세 장의 공통 진단 한 줄:

> **exp02는 "행동 인식기"가 아니라 "장면 인식기"에 가깝다.**
> "글씨 있는 판 → writing_on_a_board", "접시 많음 → cooking" 같은
> **지름길(shortcut)**을 학습했다. → 개념: [shortcut learning](concepts.md#지름길-학습--거짓-상관-shortcut-learning--spurious-correlation),
> 질문: [q09](q09-shortcut-learning.md)

이게 왜 중요한가는 [q09](q09-shortcut-learning.md)에서: **정확도는 높은데
근거가 틀린** 모델은, 배경이 바뀌는 순간(=배포) 무너진다.

## 3단계: 진단 → 처방 (오답을 3종류로 나누면 지렛대가 보인다)

| 오답 유형 | 대표 사례 | 우리 지렛대로 고칠 수 있나 |
|---|---|---|
| **배경 지름길** | phoning/reading → writing_on_a_board | ○ 배경을 흔드는 증강(RandomErasing·ColorJitter) |
| **덜 수렴** | test_loss가 10에폭까지 하락 지속(ch2) | ○ cosine 스케줄러 + 에폭 증가 |
| **미세분류** | phoning↔smoking↔drinking (손-작은물건) | △ 해상도/데이터 문제 — 우리 지렛대론 한계 |

**정직한 결론**: 4.3%p를 한 방에 메꾸는 마법은 없다. 배경 지름길과 덜 수렴은
공략 가능, 미세분류는 인정하고 기록한다. ch3의 교훈은 점수보다 이 **분해 능력**이다.

## 4단계: exp03 설계 — 한 번에 한 변수만

진단이 가장 크게 가리킨 건 **배경 지름길**이다(확신 100% 오답 3장 전부 배경).
그래서 exp03 1순위 후보는 **배경을 못 믿게 만드는 증강**:

```powershell
# 후보 A (1순위): exp02 + RandomErasing + ColorJitter, 나머지 전부 동일
#   → dataset.build_transform(train=True)에 두 줄 추가하고 새 이름으로 실행
python src/train.py --name exp03_aug_stronger --lr 1e-4

# 후보 B (2순위): exp02 + cosine 스케줄러 20에폭 (덜 수렴 가설)
#   → train.py에 scheduler 추가하고 실행
python src/train.py --name exp03_cosine20 --lr 1e-4 --epochs 20
```

> **주의 — 한 번에 한 변수**: 후보 A는 "증강만", 후보 B는 "스케줄+에폭만"
> 바꾼다. 둘을 섞으면 어느 게 효과였는지 못 가린다(exp02b 교훈).
> 코드 수정이 필요하므로, 고른 뒤에 같이 두 줄만 손대면 된다.

### 실행 전 예상 (돌리기 전에 먼저 적기!)

ch2에서 하던 예상→실측 습관 그대로. 후보 A(증강 강화)를 고른다면:

- 배경 지름길 3인방(phoning/reading의 → writing_on_a_board 오답)이 줄까? _____
- 전체 정확도: exp02 75.7% → 예상 ____% (증강이 세지면 **초반 몇 에폭은 오히려
  낮을** 수 있다 — 힌트: 학습이 어려워지니까)
- writing_on_a_board가 다른 클래스를 잡아먹던 게 줄어들까, 아니면 새로운
  혼동이 생길까? _____
- 미세분류(phoning↔smoking)는 증강으로 **안** 좋아질 것 같은데, 실제로 그런가? _____

### 결과 기록 (실측 후 채우기)

| 항목 | 예상 | 실측 | 해석 |
|---|---|---|---|
| 전체 test_acc | | | |
| phoning 정확도 (39.0%→?) | | | |
| 배경 지름길 오답 (재진단) | | | |
| 새로 생긴 혼동 | | | |

> 실행 후 `python scripts/analyze_errors.py --name exp03_aug_stronger`로
> **같은 진단을 다시** 돌려 before/after를 비교하는 게 핵심. 점수 한 줄이 아니라
> "약한 클래스가 실제로 나아졌나"로 판단한다.

## 문제가 생기면

- 증강이 너무 세서 학습이 안 오르면 → 강도를 낮추거나(예: ColorJitter 폭 축소)
  에폭을 늘린다. 한 번에 하나씩.
- `CUDA out of memory` → `--batch-size 16` (ch2와 동일).
- 곡선이 지저분해지면 → 새 이름으로 다시, `python src/metrics_logger.py remove <이름>`.

## 끝나면

- [ ] 위 "실행 전 예상" 채우고 exp03 실행
- [ ] `analyze_errors.py`로 재진단, before/after 표 채우기
- [ ] 대시보드 push (exp01·exp02·exp02b·exp03 곡선 겹쳐 보기)
- [ ] 80% 도달 여부와 무관하게, **무엇이 왜 좋아졌나/안 좋아졌나**를 한 단락으로
