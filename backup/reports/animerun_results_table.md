# AnimeRun Results

| run_name | research_branch | model_family | source_kind | epe | 1px | 3px | 5px | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_01_baseline | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| animerun_sam_v2 | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| animerun_unsup_lite | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| animerun_with_sam | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| flow_collapse_fix | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| global_matching_v1 | Other | AniFlowFormerT | missing_metrics |  |  |  |  |  |
| global_matching_v3 | Other | AniFlowFormerTV3 | missing_metrics |  |  |  |  |  |
| v5_object_memory_sam_parallel | V5 Object Memory | AniFlowFormerTV5 | tensorboard | 10.028397 | 0.154935 | 0.414258 | 0.534390 | workspaces/v5_object_memory_sam_parallel/tb_v5_object_memory_sam_parallel/events.out.tfevents.1773334444.ldtan.2985161.0 |
| v5_1_object_memory_dense_parallel_ft | V5.1 Object Memory Dense | AniFlowFormerTV5.1 | tensorboard_partial | 9.218757 | 0.230305 | 0.466289 | 0.583906 | workspaces/v5_1_object_memory_dense_parallel_ft/tb_v5_1_object_memory_dense_finetune/events.out.tfevents.1773670255.ldtan.3872171.0 |
| v5_1_object_memory_dense_parallel_v4 | V5.1 Object Memory Dense | AniFlowFormerTV5.1 | tensorboard_partial | 9.261245 | 0.224687 | 0.461006 | 0.576181 | workspaces/v5_1_object_memory_dense_parallel_v4/tb_v5_1_object_memory_dense_parallel/events.out.tfevents.1773639834.ldtan.3778696.0 |
| v5_1_object_memory_dense_parallel_v3 | V5.1 Object Memory Dense | AniFlowFormerTV5.1 | tensorboard_partial | 9.606187 | 0.153561 | 0.420488 | 0.546844 | workspaces/v5_1_object_memory_dense_parallel_v3/tb_v5_1_object_memory_dense_parallel/events.out.tfevents.1773509783.ldtan.3471747.0 |
| v5_1_object_memory_dense_parallel | V5.1 Object Memory Dense | AniFlowFormerTV5.1 | tensorboard | 10.310184 | 0.158991 | 0.405397 | 0.527462 | workspaces/v5_1_object_memory_dense_parallel/tb_v5_1_object_memory_dense_parallel/events.out.tfevents.1773478598.ldtan.3348649.0 |
| v5_1_object_memory_dense_parallel_v2 | V5.1 Object Memory Dense | AniFlowFormerTV5.1 | tensorboard | 10.603038 | 0.160332 | 0.395434 | 0.517397 | workspaces/v5_1_object_memory_dense_parallel_v2/tb_v5_1_object_memory_dense_parallel/events.out.tfevents.1773499676.ldtan.3432667.0 |
| ddflow_unsup_animerun | Other | DDFlow | tensorboard | 11.534436 | 0.208027 | 0.407469 |  | work_dirs/ddflow_unsup_animerun/tb/events.out.tfevents.1770371009.ldtan.2671741.0 |
| animerun_unsup_mdflow | Other | MDFlow | tensorboard | 22.931963 | 0.008029 | 0.066460 | 0.154657 | outputs/animerun_unsup_mdflow/tb_animerun_mdflow/events.out.tfevents.1770537688.ldtan.3842051.0 |
| segment_aware_unsup | Other | SegmentAware | missing_metrics |  |  |  |  |  |
| segment_aware_unsup_v2 | Other | SegmentAware | missing_metrics |  |  |  |  |  |
| upflow_animerun_smoke2 | Other | UPFlow | metrics_json | 2.601047 | 0.215820 | 0.855591 | 0.891968 | workspaces/upflow_animerun_smoke2/metrics_eval_script_smoke.json |
| upflow_animerun_smoke | Other | UPFlow | metrics_json | 2.601529 | 0.216797 | 0.855957 | 0.891602 | workspaces/upflow_animerun_smoke/metrics_smoke.json |
| upflow_animerun_full | Other | UPFlow | metrics_json | 9.398730 | 0.215508 | 0.437963 | 0.552496 | workspaces/upflow_animerun_full/metrics_final.json |
| upflow_animerun_smoke_v4shape | Other | UPFlow | metrics_json | 24.382416 | 0.000000 | 0.011658 | 0.041313 | workspaces/upflow_animerun_smoke_v4shape/metrics_eval_script_smoke.json |
| unflow_animerun_quick10 | Other | UnFlow | results_json | 9.360522 |  |  |  | workspaces/unflow_animerun/results_unflow_animerun_quick10_num1.json |
| unflow_animerun_quick10 | Other | UnFlow | missing_metrics |  |  |  |  |  |
| unflow_animerun_smoke | Other | UnFlow | missing_metrics |  |  |  |  |  |
| unflow_animerun_smoke50 | Other | UnFlow | missing_metrics |  |  |  |  |  |
| seed1337 | UnSAMFlow | UnSAMFlow | train_summary | 7.328736 |  |  |  | workspaces/unsamflow_animerun_repro/seed1337/train_summary.json |
| unsamflow_animerun | UnSAMFlow | UnSAMFlow | tensorboard_partial | 7.801400 |  |  |  | workspaces/unsamflow_animerun/tb/events.out.tfevents.1770538697.ldtan.3854287.0 |
| v4_unsamflow_strategy | UnSAMFlow | UnSAMFlow | tensorboard_partial | 9.438497 | 0.190604 | 0.439953 | 0.551599 | workspaces/v4_unsamflow_strategy/tb_v4_unsamflow_strategy/events.out.tfevents.1772988752.ldtan.1890633.0 |
| v4_unsamflow_strategy_v2 | UnSAMFlow | UnSAMFlow | missing_metrics |  |  |  |  |  |
| v4_6_hybrid_sam_sub7_v2 | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard | 9.430534 | 0.199020 | 0.439368 | 0.552037 | workspaces/v4_6_hybrid_sam_sub7_v2/tb_v4_6_hybrid_sam_sub7_v2/events.out.tfevents.1773200736.ldtan.2649812.0 |
| v4_6_hybrid_sam_sub7_v3_fast | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard | 9.449843 | 0.183211 | 0.439436 | 0.551918 | workspaces/v4_6_hybrid_sam_sub7_v3_fast/tb_v4_6_hybrid_sam_sub7_v3_fast/events.out.tfevents.1773222589.ldtan.2751859.0 |
| v4_6_hybrid_sam_sub7 | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard_partial | 9.513989 | 0.185461 | 0.431476 | 0.550214 | workspaces/v4_6_hybrid_sam_sub7/tb_v4_6_hybrid_sam_sub7/events.out.tfevents.1773124558.ldtan.2379722.0 |
| v4_5_hybrid_sam | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard | 9.543705 | 0.171791 | 0.429397 | 0.549901 | workspaces/v4_5_hybrid_sam/tb_v4_5_hybrid_sam/events.out.tfevents.1773031206.ldtan.2057501.0 |
| v4_5_hybrid_sam_stable | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard_partial | 9.586208 | 0.166180 | 0.428500 | 0.544451 | workspaces/v4_5_hybrid_sam_stable/tb_v4_5_hybrid_sam_stable/events.out.tfevents.1773070870.ldtan.2261884.0 |
| v4_6_hybrid_sam_sub7_v2_v5 | V4 Hybrid SAM | v4_5_hybrid_sam | tensorboard_partial | 13.610682 | 0.145136 | 0.320099 | 0.410930 | workspaces/v4_6_hybrid_sam_sub7_v2_v5/tb_v4_6_hybrid_sam_sub7_v2/events.out.tfevents.1773373344.ldtan.3055905.0 |
| v4_5_ablation_d_hybrid_maskcorr | V4 Hybrid SAM | v4_5_hybrid_sam | missing_metrics |  |  |  |  |  |
| v4_6_hybrid_sam_longgap | V4 Hybrid SAM | v4_5_hybrid_sam | missing_metrics |  |  |  |  |  |
| v4_5_ablation_b_matcher_lcm | Other | v4_5_matcher_lcm | missing_metrics |  |  |  |  |  |
| v4_epe_optimized | Other | v4_epe_optimized | tensorboard_partial | 13.690188 | 0.003785 | 0.054596 | 0.170267 | workspaces/v4_epe_optimized/tb_v4_epe_optimized/events.out.tfevents.1772894026.ldtan.1575089.0 |
| v4_epe_optimized_v2 | Other | v4_epe_optimized_v2 | missing_metrics |  |  |  |  |  |
| v4_full | Other | v4_full | missing_metrics |  |  |  |  |  |
| v4_full_lite | Other | v4_full_lite | missing_metrics |  |  |  |  |  |
| v4_full_v2 | Other | v4_full_v2 | missing_metrics |  |  |  |  |  |
| _v5_dry_run | V5 Object Memory | v5_object_memory_sam | missing_metrics |  |  |  |  |  |
