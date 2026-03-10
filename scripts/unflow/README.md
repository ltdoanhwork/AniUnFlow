# UnFlow x AnimeRun Runner

## 1) Setup environment

```bash
scripts/unflow/setup_unflow_animerun_env.sh
```

## 2) Train baseline (medium budget)

```bash
scripts/unflow/train_unflow_animerun.sh
```

## 3) Evaluate and export results

```bash
scripts/unflow/eval_unflow_animerun.sh
```

Default output:

- `workspaces/unflow_animerun/results_unflow_animerun_c_medium.json`

## 4) Smoke test (100 train iters + eval 50 pairs)

```bash
scripts/unflow/smoke_unflow_animerun.sh
```
