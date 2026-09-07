# LG Aimers 9기 × LG스포츠
## 투수 제구 성공 확률 예측

투구 직전까지 확인할 수 있는 경기 상황과 선수의 과거 기록으로 2025시즌의
`control_success` 확률을 예측하였습니다. 학습 데이터는 2019~2024시즌의 투구 단위
기록으로 구성되어 있으며, 한 행은 하나의 투구를 의미합니다.

이 문제에서는 정답 class를 맞히는 것보다 실제 성공 빈도에 가까운 확률을 출력하는
것이 중요하다고 판단하여 Brier Score를 중심 지표로 사용하였습니다. 실험을 진행할수록
모델의 복잡도보다 선수 기록의 생성 시점, 표본 수에 따른 신뢰도, 다음 시즌으로의
일반화 여부가 더 큰 영향을 줄 수 있다는 점을 확인하였고, 최종 접근에도 이 판단을
반영하였습니다.

## 프로젝트 개요

초기에는 경기 상황과 `asof_*` 누적 기록을 조합한 CatBoost만으로도 안정적인 기준
성능을 만들 수 있다고 생각하였습니다. 이후 feature를 추가하고 HCCN 구조를 확장하며
성능을 비교하였지만, local validation과 leaderboard의 개선 폭이 항상 같은 방향으로
나타나지는 않았습니다.

이 차이를 모델 capacity만의 문제로 보지 않고, 과거 기록이 실제 예측 시점에 맞게
생성되었는지와 validation이 다음 시즌의 변화를 충분히 모사하는지를 먼저 확인하는
방향으로 접근을 수정하였습니다. 그 결과 strict-past profile, shrinkage, temporal
validation, 최근 시즌 가중치, 여러 tree model의 방향을 결합하는 흐름을 최종 후보로
발전시켰습니다. HCCN은 CatBoost가 놓치는 부분을 residual로 보정할 수 있는지 확인하기
위한 challenger로 남겼습니다.

## 문제 정의 및 데이터

예측 대상은 투구 단위 이진 target인 `control_success`입니다. 출력은 0 또는 1의
class가 아니라 해당 투구가 제구 성공일 확률이며, 제출 시점에는 2025시즌을 예측합니다.

| 항목 | 값 |
|---|---:|
| train | 1,475,092행 × 49열 |
| test | 5행 × 48열 |
| 학습 시즌 | 2019~2024 |
| 예측 시즌 | 2025 |
| target | `control_success` |
| train target mean | 0.523766 |
| 정규시즌(`R`) 행 | 1,314,088 |
| `F` game type 행 | 161,004 |

기본 입력은 `balls_before`, `strikes_before`, `outs_before`, 이닝, 점수 차, 주자
상태, `li`, 투수·타자 손잡이, 팀 정보, 선수별 as-of 누적 기록으로 구성하였습니다.
2025년에 존재하지 않는 값이 생기거나 예측 시점 이후의 정보가 섞일 수 있는 컬럼은
제외하여, 학습과 추론에서 사용할 수 있는 정보의 범위를 먼저 고정하였습니다.

## 모델링 접근

### Baseline Model

여러 모델을 처음부터 동시에 비교하기보다 tabular 데이터에서 안정적인 기준 성능을
확보할 필요가 있다고 생각하여 CatBoost를 baseline으로 설정하였습니다. 이후 LightGBM과
TabM을 같은 feature 흐름에서 비교하면서, 모델 하나의 점수보다 서로 다른 모델이
같은 행에서 어떤 방향의 오차를 내는지를 확인하였습니다.

초기 실험에서는 feature 수와 모델 표현력을 늘려도 성능이 일관되게 좋아지지 않는
경우가 있었습니다. random split에서 좋아 보인 결과가 다음 시즌을 모사한 validation이나
leaderboard에서 같은 폭으로 재현되지도 않았습니다. 따라서 이후에는 모델을 확장하기
전에 시간 순서와 과거 기록의 기준 시점을 고정하는 것을 우선 과제로 두었습니다.

### Validation Strategy

2025시즌을 예측하는 문제에서 validation 행의 미래 정보가 feature에 포함되면 실제
일반화 성능을 과대평가할 수 있다고 판단하였습니다. 이에 시즌 단위 temporal fold를
사용하고, validation season보다 이전 데이터만 profile과 모델 학습에 사용하였습니다.

현재 저장된 V14 OOF는 다음 세 fold로 구성되어 있습니다. `temporal_train.py`의 현재
설정은 validation 직전 최대 4개 시즌을 사용하므로 저장 OOF의 학습 window는 2020년부터
시작합니다.

| validation season | training seasons | validation rows |
|---:|---|---:|
| 2022 | 2020~2021 | 247,472 |
| 2023 | 2020~2022 | 245,525 |
| 2024 | 2020~2023 | 253,507 |

2024년을 primary validation, 2023년을 regime stress test, 2022년을 reference fold로
사용하였습니다. Hyperparameter와 calibration은 outer validation label을 직접 보고
고정하지 않도록 inner 또는 pooled held-out 결과와 nested 진단을 구분하였습니다.

## Feature Engineering

### 경기 상황 Feature

제구 성공은 선수의 고정된 능력만으로 결정되지 않고 투구 직전의 count, 주자, 이닝,
점수 차, leverage 상황에 따라 달라질 수 있다고 보았습니다. 이에 `game_month`,
`game_dayofweek`, 이닝, 볼·스트라이크·아웃 count, `count_state`, 점수 차, 주자 상태,
`home_win_expectancy`, `li`, 손잡이와 팀 matchup을 함께 사용하였습니다.

이 feature들은 한 행의 투구 직전 상태를 설명하도록 두고, test 전체의 통계나 다른
test 행의 상태로 다시 계산하지 않았습니다. 제출 runtime은 각 입력 행과 사전에
저장된 lookup만 사용하여 예측합니다.

### Recent Form

누적 성적만 사용하면 최근 경기에서 나타난 컨디션 변화를 구분하기 어렵다고 판단하였습니다.
이를 보완하기 위해 투수의 최근 1/3/5경기 성공률과 중간 결과율을 구성하고, 최근
기록과 통산 기록의 차이도 별도 feature로 추가하였습니다. 이 차이를 통해 모델이
장기 평균뿐 아니라 최근 성적의 변화도 구분하도록 하였습니다.

### Strict-past Player Profile

2025시즌 예측에 같은 시즌의 기록이 profile에 포함되면 실제 추론 시점에서 사용할 수
없는 정보가 섞일 수 있습니다. 따라서 `profiles.py`에서는 target season보다 이전
시즌만 조회하여 투수·타자·투수팀·타자팀 profile을 생성하도록 제한하였습니다.

투수 profile에는 전체 성공률과 표본 수, 시즌 수, 좌·우타자 상대 성적, 앞선·뒤진
count, 2스트라이크·3볼, high-LI, 초반·후반 이닝, `R/F` split, 평균 이닝과 평균 LI를
포함하였습니다. 타자 profile에는 좌·우 투수 상대 성적, platoon 차이, `R/F`, count,
LI, 이닝 관련 기록을 사용하였고, 팀 profile은 과거 성공률을 별도 lookup으로
제공하였습니다.

예측 행은 ID로 이미 만들어진 lookup을 조회하므로 test 행끼리 다시 집계하지 않습니다.
이렇게 하면 한 행만 입력했을 때와 여러 행을 함께 입력했을 때 feature가 달라지지
않으며, 실제 제출 패키지에서도 single-row와 batch 예측의 일치 여부를 테스트할 수
있습니다. 저장된 패키지 검증에서는 행 순서를 섞었을 때 최대 예측 차이가 `0.0`으로
확인되었습니다.

### Shrinkage & Reliability

10번 중 7번 성공한 선수와 1,000번 중 700번 성공한 선수는 관측 성공률은 같지만
동일한 신뢰도로 보기 어렵습니다. 적은 표본의 극단적인 비율이 예측을 과도하게
움직이지 않도록 선수별 표본 수와 전체 prior를 함께 사용하는 shrinkage를 적용하였습니다.

```text
shrunk_rate = (observed_rate × n + prior × k) / (n + k)
confidence  = n / (n + 500)
```

일반 feature의 `shrink_k`는 `150`이며, profile 내부 entity effect에는 `k=250`,
직전 시즌 effect에는 `k=150`, 팀 profile에는 `k=400`을 사용합니다. prior는 각
training history에서 계산하여 validation과 test에 동일한 방식으로 적용하였습니다.

표본이 부족하거나 새로운 선수여서 lookup이 없는 경우에는 prior, 표본 수, confidence를
함께 제공하고 모델별 결측 처리 규칙을 적용하였습니다. CatBoost와 LightGBM은 수치
결측을 처리하고, TabM은 training fold의 중앙값과 표준화 통계를 사용합니다. 범주형
입력은 training vocabulary에 없는 값을 unknown bucket으로 보냅니다. 새 ID를 단순히
암기하거나 2025를 연속형 season 값으로 외삽하는 위험을 줄이기 위해 V14 기본 feature에서는
raw `pitcher_id`, `batter_id`, `season`을 제외하였습니다.

### Season Recency

2019년 기록과 2024년 기록을 동일하게 취급하면 최근 경기 환경의 변화를 충분히 반영하지
못할 수 있다고 보았습니다. 반대로 오래된 기록을 모두 버리면 안정적인 선수·팀 경향을
놓칠 수 있으므로 최근 시즌에 더 큰 weight를 주었습니다.

| validation season | season weight |
|---:|---|
| 2022 | 2020: 0.75, 2021: 1.00 |
| 2023 | 2020: 0.50, 2021: 0.75, 2022: 1.00 |
| 2024 | 2020: 0.25, 2021: 0.50, 2022: 0.75, 2023: 1.00 |

이 설정은 현재 V14 temporal training 코드와 저장 artifact에 반영된 값입니다. 별도
`mh_branch`의 4년 weight와 다른 branch의 5년 decay를 참고하였지만, 기록된 모든
과거 실험을 최종 모델에 그대로 포함하지는 않았습니다.

### Missing / Unknown Handling

새로운 선수나 과거 기록이 적은 선수는 모델이 불안정하게 반응할 수 있는 구간이라고
판단하였습니다. 그래서 lookup miss를 단일 평균으로 덮기보다 prior, 표본 수,
confidence를 함께 제공하고, 범주형 unknown과 수치 결측을 모델별 전처리 규칙에 따라
처리하였습니다. 이를 통해 모델이 값 자체와 함께 해당 기록을 얼마나 신뢰할 수 있는지도
판단하도록 하였습니다.

### Season-to-date Reconstruction

시즌 중 누적 기록은 career as-of 값만으로는 바로 분리되지 않기 때문에, 각 행 자신의
as-of count와 rate에서 target season 이전 누적값을 차감하여 복원하였습니다.

```python
career_success = asof_n * asof_rate
season_n = asof_n - prior_cumulative_n
season_success = career_success - prior_cumulative_success
season_rate = season_success / season_n
```

먼저 별도 branch에서 실제로 구현된 pitcher/batter 성공률 기반 8개 feature만 사용하였고,
`season_n < 20`이면 season-to-date rate를 안정적인 feature로 사용하지 않았습니다.
reverse, ball, strike, middle, pitchmix season-to-date feature는 분모와 누적 성공
횟수가 동일한 방식으로 정의되는지 확인할 수 없어 추가하지 않았습니다.

공식 train 기준 재구성 audit에서는 음수 count나 표본 수를 초과하는 성공 횟수가
발견되지 않았습니다. 반올림된 as-of rate로 인한 최대 rate error는 pitcher `0.00025657`,
batter `0.00026560`이었습니다. 다만 저장 OOF가 profile과 season-to-date를 함께
사용한 composite prediction이므로 season-to-date 단독 lift는 주장하지 않았습니다.

## Residual HCCN

### 설계 배경

CatBoost가 이미 상당한 기준 확률을 만들고 있다면 neural network가 전체 확률을 처음부터
다시 예측하게 하는 것보다, 기준 모델이 놓치는 오차만 보정하는 편이 안정적일 수 있다고
판단하였습니다. 이 가설을 확인하기 위해 CatBoost의 예측을 버리지 않고 그 logit 위에
작은 residual을 더하는 HCCN을 설계하였습니다.

### Architecture

서로 성격이 다른 feature를 한 벡터로 단순히 합치기보다 Entity, History, Context로
나누어 representation을 구성하였습니다.

- **Entity**: 투수, 타자, 팀, 경기 상태의 categorical embedding
- **History**: as-of 성적, 최근 form, 표본 수와 결측 관련 수치
- **Context**: count, base, inning, score, LI, pressure

```text
Entity Tower       History Tower       Context Tower
embeddings         numeric MLP         cat embeddings + numeric MLP
      \                 |                    /
       \                |                   /
        +--------- shared representation --+
                         |
             +-----------+-----------+
             |                       |
      low-rank Cross Network       Deep MLP
       2 layers, rank 32        256 -> 128 -> 64
             +-----------+-----------+
                         |
          4 residual experts, hidden 64
             pitcher / matchup / pressure / prior
                         |
           Reliability Gate -> softmax(4)
                         |
            optional Magnitude Gate (V2/V3)
                         |
       bounded residual -> base logit + residual
```

Cross branch는 low-rank projection과 residual connection으로 interaction을 학습하고,
Deep branch는 LayerNorm, SiLU, dropout을 사용하는 MLP로 비선형 관계를 추가합니다.
입력을 역할별로 나눈 이유는 장기 성적, 경기 압박도, categorical entity 정보가
residual에 서로 다른 방식으로 영향을 줄 수 있다고 보았기 때문입니다.

V1의 학습 loss는 `BCE + 0.25 × Brier + 0.001 × residual L2`로 구성하였고,
V2/V3에서는 residual penalty와 anchor loss 설정도 함께 바뀌었습니다. 따라서 버전별
결과 차이를 특정 tower나 gate 하나의 효과로 분리할 수 없습니다. 또한 HCCN의 History는
sequence encoder가 아니라 집계된 snapshot vector로 구현되어 있었습니다.

### Residual Correction

출력은 다음과 같이 구성하였습니다.

```text
CatBoost OOF probability
          ↓ logit
      base_logit
          +
      HCCN residual
          ↓ sigmoid
   final probability
```

```text
final_logit = base_logit + delta_logit
prediction   = sigmoid(final_logit)
```

기준 모델이 이미 맞힌 행까지 neural network가 크게 바꾸면 전체 Brier가 악화될 수
있다고 판단하였습니다. 이에 `max_delta × tanh(delta_raw)`로 residual 범위를 제한하고
residual L2 penalty를 추가하여 필요한 경우에만 기준 예측을 수정하도록 구성하였습니다.

### Reliability & Magnitude Gate

residual을 모든 행에 동일하게 적용하기보다 pitcher, matchup, pressure, prior expert의
신뢰도를 행별로 다르게 반영하는 reliability gate를 사용하였습니다. V2와 V3에서는
residual의 방향뿐 아니라 보정 크기 자체도 조절할 수 있도록 magnitude gate를 추가하여,
표본이 부족하거나 기준 예측이 불안정한 행에서 수정 폭을 줄이는 방향을 검토하였습니다.

| 버전 | 구조 및 학습 기록 |
|---|---|
| V1 | history raw 52개 → 104개, context raw 27개 → 54개, gate raw 10개 → 20개, entity categorical 11개, 271,461 parameters, final refit 2 epochs |
| V2 | magnitude gate 사용, model-state tower 미사용, 282,278 parameters, final refit 12 epochs |
| V3 | V2에 model-state tower 추가, 308,230 parameters, final refit 15 epochs |

V1에서 residual correction의 가능성을 확인한 뒤 V2와 V3에서 gate와 tower를 확장하였습니다.
하지만 구조와 함께 epoch와 penalty도 바뀌어 단일 구성요소의 효과를 분리하기 어려웠고,
복잡도가 커질수록 leaderboard 성능이 일관되게 증가하지도 않았습니다. 그래서 HCCN을
더 크게 만드는 것보다 feature reliability와 temporal generalization을 먼저 확인하는
방향으로 실험 우선순위를 변경하였습니다.

참고로 HCCN branch와 제출 artifact에 남아 있던 기록은 다음과 같습니다. 각 값은 동일한
OOF 생성 방식과 동일한 평가 protocol로 다시 계산된 표가 아니므로 현재 V14 결과표와
직접 합산하지 않았습니다.

| 기록 | Brier | 추가 기록 | 확인 방식 |
|---|---:|---|---|
| HCCN V1 | 0.247123 | branch 표기 BSS 1141.43, 2024 Brier 0.248198, AUROC 0.545125 | branch README와 ZIP policy |
| HCCN V4 best preselect | 0.247036 | 최종 모델 전체가 아닌 preselect 후보 | ZIP policy |
| HCCN V11 feature rebuild | 0.246938 | manifest 표기 official-like 1215.46 | ZIP manifest |

이 기록은 HCCN을 어디까지 확장했는지를 보여주는 참고 자료로 남겼으며, 같은 fold의 raw
OOF가 없는 상태에서 HCCN이 최종적으로 더 좋았다고 해석하지 않았습니다. HCCN source
branch의 `artifacts/`에도 실제 OOF parquet와 metric CSV가 남아 있지 않아, HCCN과 기준
CatBoost의 동일 fold·동일 seed 개선폭을 현재 clone에서 다시 계산할 수 없습니다.

## 모델 개선 과정

### Temporal Generalization

random split에서 보이는 성능만으로는 2025시즌의 시간 이동을 설명하기 어렵다고
판단하였습니다. 과거 branch에는 random 5-fold AUC `0.55931`과 temporal AUC `0.55033`이
기록되어 있었고, 두 평가 방식의 차이는 데이터가 섞일 때 성능이 낙관적으로 보일 수
있다는 점을 보여주었습니다. 이후 시즌 순서를 유지한 fold에서 profile과 feature를
생성하고, 2023년을 stress test로 별도 확인하였습니다.

시즌별 game type 변화도 이 판단에 영향을 주었습니다. 정규시즌 성공률은 2022년
`0.503691`, 2023년 `0.503118`, 2024년 `0.489707`이었고, `F`는 같은 기간
`0.708749`, `0.472904`, `0.459280`으로 변했습니다. 이런 변화가 관찰된 상황에서
network 구조만 확장하는 것은 다음 시즌 일반화를 확인하는 방법이 아니라고 판단하였습니다.

### Tree Ensemble

strict-past profile과 shrinkage를 적용한 뒤에는 CatBoost 세 설정과 LightGBM을 여러
seed로 학습하여 서로 다른 예측 방향을 비교하였습니다.

| 모델 | 주요 설정 |
|---|---|
| CatBoost A | 600 iterations, learning rate 0.02, depth 7, L2 25, min leaf 100 |
| CatBoost B | 600 iterations, learning rate 0.035, depth 5, L2 3, min leaf 100 |
| CatBoost C | 1,000 iterations, learning rate 0.02, depth 4, L2 8, min leaf 500 |
| LightGBM | 600 rounds, learning rate 0.01, 31 leaves, min leaf 1,500, feature/bagging fraction 0.7, L2 20 |

`mh_branch` 구현에서는 117개 feature를 사용하고 CatBoost A/B/C와 LightGBM을 합친
12-model ensemble을 구성하였습니다. 각 모델의 확률을 바로 평균하지 않고 logit으로
변환한 뒤 평균하고 다시 sigmoid를 적용하여 모델별 예측 방향을 안정적으로 결합하였습니다.

두 branch에서 얻은 아이디어를 비교하면서 다음과 같이 방향을 정리하였습니다.

| 항목 | 별도 HCCN branch | `mh_branch` / `jw_branch` 기록 | 최종 반영 |
|---|---|---|---|
| 선수 기록 | embedding과 집계 history를 network에 입력 | strict-past profile, shrinkage, raw ID 제외 | profile lookup과 표본 신뢰도 |
| 시간 사용 | HCCN config의 2년 window, decay 0.7 | `mh`: 4년 weight, `jw`: 5년 season decay와 월별 recency | 최대 4년 temporal training |
| 모델 | residual expert 계열 | CatBoost·LightGBM seed ensemble | 12-model tree ensemble |
| feature 수 | tower별 history/context/gate 분리 | `mh` 117개, `jw` 문서 112개 기록 | 현재 MH schema 117개 |
| 평가 | temporal fold, Brier, cluster bootstrap | BSS, logit 평균, 다년 temporal AUC | Brier 중심 temporal OOF |
| 기록된 결과 | HCCN V1 Brier 0.247123 | `mh` 문서 LB 1025, `jw` 문서 LB 961·다년 평균 AUC 0.55376 | 서로 다른 지표라 순위 비교하지 않음 |

`mh_branch`는 row-independent feature와 12개 모델의 logit 평균을 사용하였고,
`jw_branch`는 여러 CatBoost 설정과 LightGBM, seed ensemble, regularization을
사용하였습니다. 이 비교를 통해 모델 이름이나 구조 자체보다, 과거 기록을 예측 시점에
맞게 만들고 서로 다른 tree model의 방향을 검증하는 것이 다음 실험으로 이어지기 쉽다고
판단하였습니다.

### TabM

tree ensemble과 다른 inductive bias를 가진 보조 모델의 방향이 있는지 확인하기 위해
TabM을 같은 MH feature에 학습하였습니다. 현재 코드의 설정은 `n_blocks=3`, `d_block=256`,
`k=16`, `arch_type="tabm-mini"`, dropout `0.1`이며, seed `1234`와 `2345`를 따로
학습하였습니다.

TabM을 최종 모델 전체로 대체하기보다 V13과의 차이를 direction으로 사용하는 이유는,
absolute probability를 단순 평균할 때 calibration이 흔들릴 수 있기 때문이었습니다.
저장 OOF에서는 TabM 단독 방향 평균보다 MH와 TabM 방향을 함께 사용한 V14가 더 낮은
Brier를 보였습니다.

### Calibration

확률 예측에서는 순위가 좋아도 예측 평균과 실제 빈도가 어긋나면 Brier가 개선되지 않을
수 있다고 보았습니다. 그래서 V13을 중심으로 MH와 TabM의 absolute probability를
평균하지 않고, 정규시즌 행에서 각 후보와 V13의 차이만 결합하였습니다.

```text
p_v14_raw = clip(
    p_v13
    + R × [
        0.290352 × (p_mh - p_v13)
      + 0.268029 × (p_tabm_seed1234 - p_v13)
      + 0.245370 × (p_tabm_seed2345 - p_v13)
    ]
)
```

이후 `game_type × game_month`별 affine calibration을 적용하였습니다. pooled OOF에서는
calibration이 Brier를 개선했지만, `2023←2022`, `2024←2022+2023`으로 다시 계산한
nested diagnostic에서는 V13보다 좋아지지 않았습니다. 따라서 local OOF의 개선을
공식 test 일반화 성능으로 해석하지 않고 calibration의 장점과 한계를 분리하였습니다.

## 최종 모델

저장된 동일 기준 OOF에서 측정 가능한 후보를 비교한 결과, `E_v14_month_calibrated`를
선택하였습니다. 2024 Brier를 primary objective로 두고 2023 stress fold에서 큰
악화가 없는지, 2022까지 방향성이 유지되는지를 함께 확인하였습니다.

```text
MH profile direction       0.290352
TabM seed 1234 direction   0.268029
TabM seed 2345 direction   0.245370
```

제출 runtime에는 기존 Track A → HCCN V1/V2/V3 → V4/V5 router → V13/V14 단계가
상속되어 있습니다. 다만 같은 fold의 raw HCCN OOF가 보존되어 있지 않아 HCCN의
독립적인 incremental lift를 측정했다고 주장하지 않았습니다. 최종 선택의 근거는
HCCN 단일 모델의 성능이 아니라 저장된 V13 OOF를 기준으로 한 MH·TabM direction blend와
calibration의 비교 결과입니다.

## 실험 결과

### Pooled Temporal OOF

아래 표는 `final/v14_results.json`의 2022~2024 temporal OOF 결과입니다. Standard BSS는
`1 - Brier / climatology_Brier`이며 Official-like score는 이를 `100000`배한 local
표기입니다. 공식 leaderboard 점수로 해석하지 않았습니다.

평가 기준은 Brier `mean((y - p)^2)`(낮을수록 우수), Standard BSS
`1 - Brier / climatology_Brier`, Official-like score `100000 × Standard BSS`의
local 표기입니다. AUROC는 초기 branch 기록에는 존재하지만 V14 결과 JSON의 최종 지표에는
포함되어 있지 않아 HCCN branch 참고표에서만 제시하였습니다.

| 모델 | Brier ↓ | Standard BSS ↑ | Official-like |
|---|---:|---:|---:|
| V13 baseline | 0.246641 | 0.013345 | 1334.49 |
| V14 uncalibrated | 0.246279 | 0.014791 | 1479.12 |
| **V14 month-calibrated final** | **0.246189** | **0.015150** | **1515.04** |

V14 final은 V13 대비 pooled Brier를 `0.000451` 낮추었고, pitcher cluster bootstrap
3,000회에서 95% CI는 `[-0.000567, -0.000352]`였습니다. 같은 투수의 여러 투구가
독립 표본처럼 보이는 문제를 줄이기 위해 cluster bootstrap을 사용하였지만, 공식 test
정답이나 leaderboard confidence interval을 의미하지는 않습니다.

### Fold별 결과

| validation season | V13 Brier | V14 final Brier | delta |
|---:|---:|---:|---:|
| 2022 | 0.243444 | 0.242933 | -0.000510 |
| 2023 | 0.248728 | 0.248079 | -0.000649 |
| 2024 | 0.247739 | 0.247537 | -0.000202 |

세 fold에서 같은 방향의 개선이 나타났지만 nested calibration에서는 다음 시즌
일반화가 개선되지 않았습니다. 따라서 미래 test에서 반드시 더 좋다고 단정하지 않고,
저장된 OOF 범위 안에서 V14가 더 안정적인 후보였다고 정리하였습니다.

### R/F Subgroup

R/F별 Brier, BSS proxy, AUROC와 전체 후보 비교표는 다음 artifact에 분리해 저장하였습니다.

- [전체 후보 비교표](artifacts/lgaimers_final/experiments/results.csv)
- [연도별 fold 결과](artifacts/lgaimers_final/experiments/fold_results.csv)
- [R/F subgroup 결과](artifacts/lgaimers_final/experiments/subgroup_results.csv)
- [상세 실험 기록](artifacts/lgaimers_final/experiments/results.md)

pooled 평균만으로는 정규시즌과 `F` game type의 distribution shift를 확인하기 어렵기
때문에 subgroup 지표를 별도로 계산하였습니다. 실제 test target은 존재하지 않으므로
test의 성공률을 BSS 기준값으로 사용하지 않았습니다.

## 시행착오 및 한계

### HCCN 구조 확장만으로 해결하려 한 접근

HCCN V1에서 residual correction의 가능성을 확인한 뒤 V2와 V3에서 magnitude gate와
model-state 정보를 확장하였습니다. 하지만 구조와 함께 epoch와 penalty도 바뀌어
단일 구성요소의 효과를 분리할 수 없었고, 복잡도가 커질수록 leaderboard 성능이
일관되게 상승하지도 않았습니다. 이 결과를 통해 모델을 더 크게 만드는 것보다
feature reliability와 temporal generalization을 먼저 확인해야 한다고 판단하였습니다.

### Pooled calibration을 미래 성능으로 해석한 위험

세 시즌 held-out prediction에 fit한 month calibration은 local OOF에서 개선을 보였지만,
다음 시즌만을 남겨 둔 nested diagnostic에서는 V13보다 악화되었습니다. 따라서 두 결과를
함께 제시하고 pooled score만으로 calibration의 일반화를 주장하지 않았습니다.

### Profile과 season-to-date ablation의 미분리

현재 저장 OOF는 profile과 season-to-date를 함께 포함한 composite prediction입니다.
재구성 자체는 통과했지만 profile만 사용한 모델과 season-to-date를 제거한 모델의
same-fold pair가 보존되어 있지 않아 각 feature의 단독 lift는 보고하지 않았습니다.

### Contextual history aggregate

초기 HCCN branch에는 pitcher×count, pitcher×hand matchup과 같은 aggregate spec도
정의되어 있었습니다. 그러나 해당 utility가 초기 HCCN training path에서 실제 입력으로
호출되었다고 확인할 수 있는 artifact는 남아 있지 않으므로, 설계된 feature를 사용했다고
성과 설명에 포함하지 않았습니다.

### TrackMan

TrackMan 데이터를 최종 feature로 포함하는 방법도 검토하였지만, test 추론에서 직접
사용할 수 있는 pre-pitch 입력으로 안정적으로 연결하고 incremental Brier 개선을
확인한 artifact가 없어 최종 모델에는 포함하지 않았습니다. 제출 runtime은 TrackMan
파일을 요구하지 않습니다.

### 재현 범위

현재 workspace에는 HCCN same-fold raw OOF, 모든 year weighting 대안의 prediction file,
독립적인 season-to-date ablation, TrackMan student 결과가 없습니다. 이 항목은 점수를
추정하지 않고 [`reports/BLOCKED.md`](artifacts/lgaimers_final/reports/BLOCKED.md)에
필요한 파일과 함께 기록하였습니다. 저장된 V14 OOF의 fold는 2022~2024이며, 2019년부터
동일 protocol로 생성된 전체 비교 결과는 보존되어 있지 않습니다.

## 최종 추론 흐름

```text
official train.csv
   │
   ├─ validation season 이전 데이터로 pitcher / batter / team profile 생성
   ├─ prior 기반 shrinkage와 recent form 계산
   ├─ season-to-date 성공률 복원 및 표본 수 검사
   ├─ CatBoost A/B/C + LightGBM temporal training
   │       └─ 12-model MH logit ensemble
   ├─ regular-season TabM seed 1234 / 2345
   ├─ V13을 기준으로 R-only direction blend
   ├─ game_type × game_month affine calibration
   └─ submission.csv
```

제출 artifact에는 앞단의 Track A와 HCCN/V4/V5/V11~V13 모델도 포함되어 있어 실제
script는 이 단계를 순서대로 실행합니다. 반면 동일한 학습 artifact가 모두 Git에
보존되어 있는 것은 아니므로 깨끗한 clone만으로 최종 ZIP의 전체 학습 과정을 재현할
수 있다고 설명하지 않았습니다.

## Repository Structure

```text
.
├── challengers/
│   └── v14_mh_profile_ensemble/
│       ├── features.py          # 행 단위 feature 생성
│       ├── profiles.py          # strict-past profile
│       ├── temporal_train.py    # temporal OOF CatBoost/LightGBM
│       ├── regular_train.py     # 정규시즌 실험
│       ├── tabm_screen.py       # TabM 후보 확인
│       ├── tabm_full_refit.py   # TabM seed별 refit
│       ├── residual_stack.py    # residual 비교 실험
│       └── select_v14.py        # direction blend/calibration/bootstrap
├── final/
│   ├── v14_results.json         # V13/V14 결과와 nested 진단
│   ├── v14_policy.json          # direction weight와 calibration
│   ├── package_verification.json # 기존 제출 패키지 검증
│   └── SHA256SUMS.txt           # local ZIP checksum 기록
├── artifacts/lgaimers_final/    # 재평가표·감사 보고서·제출 패키지
├── data/raw/                    # 원본 CSV, Git 제외
├── pyproject.toml
├── uv.lock
└── README.md
```

## 실행 방법

Python 3.11 이상과 `uv`를 사용합니다.

```bash
uv sync --dev
```

원본 파일은 다음 위치에 둡니다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

V14 feature cache와 temporal prediction을 생성합니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train --cache-only
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train \
  --models cat_a,cat_b,cat_c,lgb --seeds 0
```

TabM 후보와 refit은 다음과 같이 실행합니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.tabm_screen \
  --valid-season 2024 --seed 1234
uv run python -m challengers.v14_mh_profile_ensemble.tabm_full_refit
```

`select_v14`는 MH/TabM output과 과거 V13 OOF
`challengers/v11_feature_rebuild/artifacts/main_screen_oof.parquet`가 필요합니다.
이 artifact는 현재 Git에 없으므로 깨끗한 clone에서는 마지막 selection 명령이 바로
실행되지 않을 수 있습니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.select_v14
```

재평가와 제출 패키지 검증은 다음과 같이 실행합니다.

```bash
uv run python artifacts/lgaimers_final/experiments/feature_reconstruction.py
uv run python artifacts/lgaimers_final/experiments/evaluate_oof.py
uv run pytest -q artifacts/lgaimers_final/tests
python artifacts/lgaimers_final/submit_package/script.py
```

최종 제출 결과는 [`artifacts/lgaimers_final/output/submission.csv`](artifacts/lgaimers_final/output/submission.csv),
압축 패키지는 [`artifacts/lgaimers_final/submit.zip`](artifacts/lgaimers_final/submit.zip)에
저장됩니다. 패키지에는 `script.py`, `requirements.txt`, `model/`만 포함하고
`data/`와 `output/`은 포함하지 않습니다.

## Implementation과 확인 범위

현재 저장소에서 재현 가능한 후속 모델 코드는 다음과 같습니다.

- [`features.py`](challengers/v14_mh_profile_ensemble/features.py)
- [`profiles.py`](challengers/v14_mh_profile_ensemble/profiles.py)
- [`temporal_train.py`](challengers/v14_mh_profile_ensemble/temporal_train.py)
- [`select_v14.py`](challengers/v14_mh_profile_ensemble/select_v14.py)
- [`v14_results.json`](final/v14_results.json)

초기 HCCN source는 별도 작업 branch의
[`challengers/residual_hccn/`](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch/challengers/residual_hccn)에
남아 있습니다. HCCN branch의 기록과 현재 V14 JSON을 하나의 성능 순위로 합치지
않은 이유는 OOF 원본과 실행 조건이 동일하게 보존되어 있지 않기 때문입니다.

이 프로젝트에서 가장 중요하게 남은 결정은 모델을 계속 복잡하게 만드는 것보다
예측 시점에 맞는 기록을 만들고, 시간 순서를 지킨 validation에서 개선이 유지되는지를
확인하는 것이었습니다. 그 판단이 strict-past profile, shrinkage, temporal OOF,
HCCN residual 실험, 최종 direction blend를 하나의 실험 흐름으로 연결하였습니다.
