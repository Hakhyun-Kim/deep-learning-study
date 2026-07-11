# ch2 실험 가이드 — exp02 전체 파인튜닝

> exp01과 코드는 동일, 옵션만 다르다. 이 문서는 실험 **전에** 예상을 적고,
> **후에** 결과와 비교하는 실험 노트다. (예상 → 실측 → 해석 습관 들이기 —
> 테슬라도 새 모델을 내보내기 전에 예상 지표를 먼저 걸어두고 실측과 비교한다.)

## 실행

```powershell
# 본 실험 (10에폭, lr 1e-4)
python src/train.py --name exp02_full_finetune --lr 1e-4

# 끝나면 대시보드 반영
git add docs/metrics && git commit -m "exp02 full finetune results" && git push
```

`--freeze-backbone`을 빼는 것과 lr을 1e-4로 낮추는 것, 차이는 이 둘뿐이다.

## 실행 전 예상 (돌리기 전에 먼저 적기!)

아래를 자기 말로 채우고 나서 실행할 것. 틀려도 좋다 — 예상과 실측의 차이가
가장 많이 가르쳐준다.

- 에폭당 시간: exp01은 ~18초였다. exp02는 __?5-10배?_초일 것 같다 (이유: ___)
  → [q07](q07-why-slower-epochs.md)
- 에폭 1의 test_acc: exp01은 54.6%였다. exp02는 더 높을까 낮을까? _낮다__
  (힌트: 백본이 움직이기 시작하면 초반엔 오히려 흔들릴 수도)
- 최종 test_acc: _59__% (exp01 69.4% 대비)
- 과적합 신호(test_loss 상승)가 나타날까? 몇 에폭쯤? _?__

## 결과 기록 (실측 후 채우기)

| 항목 | 예상 | 실측 | 해석 |
|---|---|---|---|
| 에폭당 시간 | | | |
| 에폭 1 test_acc | | | |
| 최고 test_acc (에폭) | | | |
| test_loss 최저점 에폭 | | | |

- 학습 곡선에서 눈에 띈 것:
- exp01 곡선과 겹쳐 보고 알게 된 것:

## 비교 실험 — q03을 곡선으로 확인

본 실험이 끝나면, lr만 10배 올린 쌍둥이 실험으로
[q03](../ch1/q03-full-finetune-low-lr.md)의 해설을 검증한다:

```powershell
python src/train.py --name exp02b_full_lr1e-3 --lr 1e-3
```

예상 (q03 해설이 맞다면): 초반 에폭에서 test_acc가 exp02보다 크게 낮거나
심하면 exp01 베이스라인(69.4%) 아래로 꺼지고, 끝까지 회복 못 할 것.
대시보드에서 세 곡선(exp01, exp02, exp02b)을 겹쳐 보면 "학습률이 사전학습
가중치를 지키는 안전장치"라는 게 그림 한 장으로 보인다.

## 문제가 생기면

- `CUDA out of memory` → `--batch-size 16` 추가
  ([concepts.md의 GPU 메모리 항목](concepts.md#gpu-메모리와-배치-크기))
- 에폭당 시간이 너무 길면 → `--epochs 5`로 먼저 경향만 확인
- 곡선이 이상하면 → 실험 이름 새로 지어 다시 (대시보드에 겹쳐 보며 원인 추적,
  지저분해진 실험은 `python src/metrics_logger.py remove <실험명>`)

## 끝나면

- [ ] 위 결과 기록 채우기
- [ ] [q03](../ch1/q03-full-finetune-low-lr.md) "내 답" 채우고 상태 ✅로
- [ ] [q07](q07-why-slower-epochs.md) 실측으로 답 채우기
- [ ] `README.md`(루트) 실험 기록 표에 exp02 줄 추가
- [ ] 80%에 못 미치면: ch3에서 증강 강화·스케줄러·층별 lr 중 **하나씩** 실험
