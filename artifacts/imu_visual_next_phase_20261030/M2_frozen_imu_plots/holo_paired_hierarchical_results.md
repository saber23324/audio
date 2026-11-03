|Variant|Correct/QA|Accuracy [cluster 95% CI]|Gold NLL|Macro F1|Visual tokens|
|---|---:|---:|---:|---:|---:|
|H0_RGB|24/64|0.3750 [0.2969, 0.4375]|1.864270|0.4396|[2920]|
|H0_V1|22/64|0.3438 [0.1875, 0.5004]|1.498790|0.3408|[256]|
|H1_plot|25/64|0.3906 [0.3438, 0.4375]|1.713900|0.3350|[730]|
|H2_real|24/64|0.3750 [0.3281, 0.4219]|1.965642|0.4856|[3650]|
|H3_duplicate|23/64|0.3594 [0.2812, 0.4375]|1.936488|0.4314|[3650]|
|H4_zero|24/64|0.3750 [0.3281, 0.4219]|1.945589|0.4856|[3650]|
|H5_wrong|24/64|0.3750 [0.3281, 0.4219]|1.964939|0.4856|[3650]|
|H6_past_shift|24/64|0.3750 [0.3281, 0.4219]|1.954738|0.4856|[3650]|
|H7_numeric_text|23/64|0.3594 [0.2500, 0.4375]|1.437789|0.3832|[0]|
|H1_zero_plot|24/64|0.3750 [0.2969, 0.4375]|1.704421|0.3387|[730]|
|H1_wrong_plot|25/64|0.3906 [0.2500, 0.4844]|1.683888|0.3224|[730]|
|H1_prior_no_image|23/64|0.3594 [0.2812, 0.4219]|1.567146|0.3694|[0]|
|Hs_P2_sensor_instructions_only|25/64|0.3906 [0.3438, 0.4375]|1.719350|0.3350|[730]|
|Hs_P2_sensor_instructions_real|25/64|0.3906 [0.3438, 0.4375]|1.938839|0.4959|[3650]|
|Hs_P2_sensor_instructions_zero|24/64|0.3750 [0.3281, 0.4219]|1.938481|0.4891|[3650]|
|Hs_P2_sensor_instructions_wrong|24/64|0.3750 [0.3281, 0.4219]|1.941415|0.4856|[3650]|
|Hs_P1_neutral_only|21/64|0.3281 [0.2344, 0.4062]|1.853250|0.2788|[730]|
|Hs_P1_neutral_real|25/64|0.3906 [0.3438, 0.4375]|2.255096|0.4851|[3650]|
|Hs_P1_neutral_zero|22/64|0.3438 [0.3125, 0.3750]|2.234198|0.4282|[3650]|
|Hs_P1_neutral_wrong|23/64|0.3594 [0.3125, 0.4066]|2.249568|0.4545|[3650]|

|Paired comparison|Delta accuracy [cluster 95% CI]|Rescue / Harm|Delta NLL|
|---|---:|---:|---:|
|H2_real minus H0_RGB|0.0000 [-0.0625, 0.0781]|3 / 3|0.101372|
|H2_real minus H3_duplicate|0.0156 [-0.0469, 0.0938]|4 / 3|0.029154|
|H2_real minus H4_zero|0.0000 [0.0000, 0.0000]|0 / 0|0.020053|
|H2_real minus H5_wrong|0.0000 [0.0000, 0.0000]|0 / 0|0.000703|
|H2_real minus H6_past_shift|0.0000 [0.0000, 0.0000]|0 / 0|0.010904|
|H1_plot minus H1_prior_no_image|0.0312 [-0.0312, 0.0938]|6 / 4|0.146754|
|H1_plot minus H1_zero_plot|0.0156 [-0.0312, 0.0625]|3 / 2|0.009479|
|H1_plot minus H1_wrong_plot|0.0000 [-0.0938, 0.0938]|2 / 2|0.030012|

Development only. Few independent clusters: intervals are exploratory. Zero flips and degenerate intervals do not establish equivalence/noninferiority. Macro-F1 uses observed gold classes; four-choice QA is not official multiclass benchmark accuracy.
