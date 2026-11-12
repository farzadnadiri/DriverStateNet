# Hypotheses and analysis plan (IDD session-order study)

*Written 27 Oct 2026, before any pre-hazard-window results were computed (not externally preregistered). Results computed on the end-of-scenario (hazard) window in the April 2026 course report were already known; everything else below is confirmatory unless marked exploratory. Changes after this commit must be logged at the bottom with a reason.*

Code: `src/study/` and `scripts/study/`. Aggregate results: `artifacts/study/RESULTS.md`.

## Data and units
- Toyota Research Institute Impaired Driving Dataset (IDD), CC BY-NC-SA 4.0.
- 52 participants. v1 (alcohol arm, n = 20, TRI employees) and v2 (control arm, n = 32, external).
- Round R1 = before the break. Round R2 = after the break; v1 drinks to ~0.10% BAC during the break, v2 does not.
- Unit of analysis: trip (one scenario drive). Unit of inference: participant.
- Trip exclusion (primary): DBM defaults (8 removed participants; scenarios with annotated simulator anomalies removed).

## Windows (fixed before results)
| Name | Definition (relative to scenario end) | Role |
|---|---|---|
| `pre90` | [−120 s, −30 s] | **Primary** (routine driving, hazard excluded; same as TRI) |
| `pre30` | [−60 s, −30 s] | Duration-matched comparison with `haz30` |
| `haz30` | [−30 s, 0 s] | Hazard-proximal window (used in the course report) |

## Measures
**Primary behavioural measures** (pre90 window):
- Gaze: horizontal gaze dispersion (gaze yaw SD), percent road centre (PRC; share of valid gaze samples within 8° of the window's median gaze direction), fixation and saccade change rates, pupil diameter.
- Vehicle control: steering-wheel angle SD, steering reversal rates (0.5° and 2.5°), lateral acceleration SD, speed mean and SD.

Lane position is not available in IDD, so SDLP cannot be computed. Steering-wheel angle SD and lateral acceleration SD are used as lateral-control proxies.

**Hazard response** (exploratory until hazard onset estimation is validated): brake reaction time, peak deceleration, hazard collision.

## Hypotheses
- **H1 (cognitive load: gaze concentration).** CD reduces gaze yaw SD and increases PRC, without degrading lateral control (steering-wheel SD, lateral acceleration SD not increased). Basis: cognitive control hypothesis (Engström et al., 2017).
- **H2 (alcohol: control degradation).** Alcohol, estimated as the v1-specific R1→R2 change beyond the v2 R1→R2 change, increases steering-wheel SD and steering reversals. Basis: alcohol myopia (Steele & Josephs, 1990); Ahlstrom et al. (2023).
- **H3 (over-additivity).** The alcohol × CD interaction is in the direction of greater impairment than the sum of the separate effects. Basis: Van Dyke & Fillmore (2015).
- **H4 (observability).** Classification of impairment from TRI's 10 features is better in `haz30` than in `pre30` (duration-matched).
- **H5 (order confound).** Session order alone (R1 vs R2 in the control arm) is detectable above chance from the same features. If so, alcohol estimates that do not correct for order are inflated.

## Models
- **Behaviour (H1–H3, H5):** linear mixed model per measure, on the pre90 window, fitted on both arms:
  `y ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario) + (1 | participant)`
  where `alcohol = cohort_v1 & round2`. `round2` estimates the order effect from both arms; `alcohol` is the extra v1 change. Effects are reported as standardised estimates with 95% CIs. Holm correction within each hypothesis family.
- **Order and cohort diagnostics (H5):** logistic regression (nested CV) for R2 vs R1 within v2, and for v1 vs v2 within R1. Report UAR with participant-bootstrap CIs. Exploratory: apply the v2-trained order model to v1 trips.
- **Classification (H4):** logistic regression on TRI's 10 features, nested CV (outer = TRI fixed folds; inner = leave-one-fold-out over the remaining folds; C tuned on inner UAR), class-balanced. Tasks: 4-class, binary CD, binary intoxication. UAR with participant-bootstrap CIs; paired participant bootstrap for window comparisons.

## Deviations log
- **27 Oct 2026: lateral-control measures added after first results.** Steering-wheel angle SD
  had a between-participant ICC of 0.00 and very wide intervals, consistent with turns at
  intersections dominating it. Two straight-road measures were added: high-pass steering SD and
  lateral deviation SD from the smoothed path (|yaw rate| < 3°/s). These are **exploratory**,
  because they were defined after seeing results. The original measures are still reported.
- **27 Oct 2026: Holm family size.** Holm correction is applied per term across all behavioural
  measures (15 after the addition above, previously 13). This is stricter than before; the CD effect
  on gaze yaw SD moved from Holm p = 0.043 to just above 0.05.
- **27 Oct 2026: hazard-response definitions fixed.**
  - Onset is the first time the hazard actor exceeds 0.5 m/s within the last 40 s.
  - Brake RT is the first brake > 0.02 within 6 s, excluding trips already braking at onset.
  - Throttle release is throttle < 0.02.
  - Scenario 2b (door-open) is excluded because its actor never moves.
  - All hazard-response analyses remain exploratory.
- **27 Oct 2026: TRI replication added (not in the original plan; supports the novelty comparison).**
  - TRI's Wilcoxon tests were re-run on the pre90 window, plus the same test in the control arm.
  - Pairs are formed on participants present in both conditions.
  - TRI's published values match only when anomaly trips are *kept*, so the replication is reported on all trips as well as on the primary set.
- **27 Oct 2026: break length differs by cohort.** Median R1→R2 break is 62 min in the alcohol arm vs
  21 min in the control arm. The order correction assumes equal R1→R2 change in both arms; the longer
  break weakens that assumption and is reported as a limitation. It is not modelled, because break
  length is collinear with cohort.
- **27 Oct 2026: matched model comparison added (secondary).**
  - Linear SVM and gradient boosting (small declared grids) were added to logistic regression, all nested on pre90.
  - The course project's temporal transformer was run on 84-channel pre90 sequences, with two candidate configurations chosen in the inner folds. Configuration A was originally tuned on the same folds; nested selection limits but does not remove that optimism.
  - Result: no model differed significantly from logistic regression.
- **27 Oct 2026: reviewer-requested robustness checks (post hoc).**
  - A random slope for CD left all conclusions unchanged; the CD effect on speed SD fell below Holm significance.
  - PRC around a per-driver baseline centre kept the CD effect (+0.39 SD) but not the alcohol effect (+0.42 SD, CI includes 0). The alcohol PRC result is therefore reported as definition-dependent.
- **27 Oct 2026: subjective measures not testable.** KSS, NASA-TLX
  and PANAS are not part of the public IDD release; a data request to TRI is needed.
- **29 Oct 2026: revision (all choices made after earlier results were known).**
  - **Main model.** The full factorial model `cohort × R2 × CD + route` replaces the constrained model as the main specification. Estimands are defined as contrasts (alcohol without CD, alcohol during CD, alcohol × CD, CD averaged over sober cells, session change in controls).
  - **Exclusions.** The exclusion imbalance (control R1 28/128 excluded) was audited. Added sensitivity analyses:
    - window-specific inclusion;
    - all 416 trips;
    - complete-case drivers (23 drivers, 184 trips).
  - **Further sensitivity analyses:**
    - gaze validity ≥ .75 and gaze validity as a covariate;
    - log scale for the seven dispersion outcomes;
    - participant random slopes for CD and session part;
    - MGC around each driver's baseline centre.
    - Participant-cluster bootstrap (500 resamples) for every estimand.
  - **Renaming.** PRC is renamed median-centred gaze concentration (MGC). Fixation and saccade "rates" are renamed transition rates, because the measure counts onsets and offsets.
  - **Changed result.** With the full model, the driver-centred MGC alcohol contrast is +0.73 SD [0.18, 1.27] (constrained model: +0.42, CI includes 0). The association is reported as specification-sensitive.
  - **False alarms.** Trip- and driver-weighted rates, task- and route-standardised gaps, and Holm across the four models were added. The driver-weighted gaps survive for all models; the trip-weighted gaps survive for 2 of 4.
- **3 Nov 2026: reporting clarifications (no new models).**
  - **Constrained vs full interaction.** The constrained model's interaction estimates are smaller than the full model's (horizontal dispersion +0.15 vs +0.76 SD; speed SD +0.24 vs +0.80 SD), with Holm p = 1.00 for every outcome. Combined-condition estimates therefore depend on that constraint.
  - **Pupil.** The alcohol-associated pupil contrast without CD survives Holm adjustment in the constrained model (p = .039) and in no other specification. Descriptive participant-mean changes (no-task trips, primary set): +0.003 mm in the alcohol cohort, −0.070 mm in controls.
  - **Order of the session-change figure.** `fig_order_did` now shows horizontal dispersion, fixation-transition rate, then pupil.
