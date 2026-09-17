# Analysis of q-BLMS and reproducible experiments

## Conclusion

There is no universal pair of q and eta that dominates BLMS. Under the white input and low noise conditions of the original examples, the regularization introduces a bias that harms the final error. In narrow band and frequent changes, it can reduce the average tracking error.

The main experiment uses **q=2.5, eta=0.00055** and improves the clean-output prediction error by **2.027 dB** compared to BLMS with separately tuned eta. Paired 95% bootstrap CI: **[2.005, 2.047] dB**. This claim is limited to the described synthetic scenario; it does not demonstrate universal superiority, nor superiority over APA or other regularizations.

## 1. Equation and mechanism

Equation (5) is implemented literally:
w[n+1] = w[n] + 2 eta X[n] e[n] - eta (q-1) diag(X[n] X[n]^T) w[n].

The block has K samples and shifts ONE sample per update. It is not non-overlapping block BLMS. q=1 exactly matches the BLMS used in the files. There is no X^T X inversion: the limit is not APA.

For scalar q > 1, the recursion is an ordinary gradient of the instantaneous cost:
||d-X^T w||^2 + (q-1)/2 sum_i(sum_k X_ik^2) w_i^2.
Therefore, it is a data-dependent diagonal regularization, equivalent to variable leakage per coefficient. Its origin in q-calculus does not eliminate the bias-variance tradeoff.

Under the independence approximation of the paper:
R=E[XX^T], D=(q-1)Diag(diag R),
E[w_infinity] = (2R+D)^(-1) 2R w_o.
The general IN-MEAN stability bound is 0 < eta < 2/lambda_max(2R+D), provided 2R+D is positive definite. It is not a guarantee of mean-square stability. With overlapping blocks, independence between the state and X is an approximation.

For a stationary scalar input of variance sigma_x^2 and delay regressors:
R=K R_x, D=(q-1) K sigma_x^2 I.
The regularization lifts weak modes but biases the solution. For white input, E[w_infinity] = 2/(q+1) w_o. For example, q=1.1 introduces a relative bias error of 0.1/2.1; q=1.001 is practically at the BLMS limit.

For the AR(2) of exp_especial (normalized), M=120, K=5:
lambda_min(R)=0.01264475, lambda_max(R)=138.11550; condition number approx 10923.
q < 0.994942 causes the mean approximation to have a non-positive mode. Therefore, transferring q=0.8 from another example to this scenario is not advisable. The diagonal bound of equations (12)-(13) of the paper should not be presented as the exact general bound for correlated inputs.

## 2. Findings in the files

- **I_identificador_2.m:** K=4, eta=mu_blms=1e-3, q=[1.15,1.1,1.01,1.001,0.8], runs=1. The system has 256 coefficients and the filter 128. With the MATs present, the 128:255 tail contains only 1.285e-5 of the energy of the first 256 coefficients; undermodeling exists, but it doesn't solely explain a large difference. [0.35,1,0.35] colors a white input; the comment calling it whitening is inaccurate.
- **II_iden_benoulli_2.m:** mu_blms=1e-5 and eta_q=1e-4: a factor of TEN. A faster q curve with this setup may be due to the step size. A sweep with both families was repeated; BLMS at 1e-4 eliminated that advantage on the final floor. Impulses are added to the input and then pass through the system; they are not independent measurement outliers.
- **III_iden_sign_2.m:** the LMS reference uses sign(e), while BLMS and q-BLMS keep the squared error. Comparing q-BLMS to that curve does not isolate the effect of q.
- **IV_iden_mnr_2.m / _3.m:** contain both the squared and sign versions. The latter are different algorithms and their correctness is not derived from the paper's squared cost. In _2, MNR is calculated for Sign-LMS; _3 adds MNR for the other families, allowing this metric to be compared among them. Some blocks start at n=2 and omit the first sample of the buffer.
- **V_iden_alphastable.m:** uses alpha=1.65 and sign(e), not equation (5). The alpha-stable distribution has no finite second moment for alpha < 2; invoking the paper's covariance/MSE analysis directly to justify it is inappropriate. It would require another formulation or declared truncation.
- **exp_especial.m:** is the best existing baseline for tracking: M=120, K=5, eta=5e-4, noise variance 1e-3, and AR(2) input. q > 1 improves part of the readaptation after inverting h, but can worsen the preceding stretch and the final error. The change rewrites d from past samples, and the system wo changes sign between realizations without re-initializing; in the replica, historical samples are preserved and the system is reset.
- **Identificador_sistemas_2.m:** white input and low noise. q=1.001 almost matches BLMS; q=1.01 or 1.05 adds bias. It's not a strong baseline to prove an advantage. q values are printed with two decimals, making 1.001 look like 1.00.
- **q_BLMS_AEC.m / frase.mat / wo.mat:** 74000 speech samples, 1024 coefficients, K=5, eta=1e-3. Approximate input power 0.01348, echo path energy 1, approximate measurement SNR 41.27 dB. The pilot test with q=[1,1.001,1.01,1.05,1.1] and eta=[1e-4,3e-4,1e-3,2e-3] favored BLMS at 1e-3; q=1.1 worsened the final NMSD from approx -7.60 to -6.31 dB. This is a single voice, not a statistical validation across speakers.

The previous pilots guided the design. They are not confirmatory tests of superiority. In AEC, there are low-energy segments: a small residual MSE does not inherently equal a high ERLE. Evaluate ERLE with echo energy in the SAME window and report voice activity.

Metric corrections: use mean(e.^2) on a linear scale, then 10*log10. Avoid abs(resample(...)) as an MSE estimator. Do not count initialization zeros. Differentiate 10*log10(E||w-h||^2/||h||^2) from 20*log10(E||w-h||/||h||), which are not the same statistic. Represent lack of reconvergence as censoring/NaN, not as a time reached at the end of the record.

## 3. Parameter alternatives in the existing code

For **exp_especial.m**, keep eta_q=eta_blms=5e-4 and test q=1.01,1.05,1.1. q=1.3 is useful to highlight the tradeoff, not as a universal winner.
200 independent realizations were validated, seeds 1510000..1510199, with unchanged equations and correct historical buffers.

| q | Prev NMSD 2501-3000 (dB) | Track NMSD 3001-5500 (dB) | Final NMSD 5501-6000 (dB) | Track EMSE (dB) |
|---|---:|---:|---:|---:|
| 1 | -11.976 | -0.835 | -8.452 | -8.856 |
| 1.01 | -11.739 | -0.951 | -8.573 | -8.948 |
| 1.05 | -10.744 | -1.355 | -8.759 | -9.274 |
| 1.1 | -9.582 | -1.741 | -8.526 | -9.590 |
| 1.3 | -6.554 | -2.468 | -6.513 | -10.202 |

This table compares the same step size, not a global optimum for each algorithm. It's an alternative to report transient improvement with its stated stationary cost. For white input, high-SNR speech, or Bernoulli, no convincing final improvement over well-tuned BLMS was found in the sweeps performed. q close to 1 reduces deterioration, but does not constitute an experimental contribution on its own.

## 4. Proposed main experiment

**Tracking a FIR with repeated inversions under highly correlated input.**

- M=120, K=5; T=12000; initial state w=0.
- h0 is a 119th-order low-pass FIR, cutoff 0.4 relative to Nyquist, symmetric Hamming window, and unit DC gain.
- h[n]=(-1)^floor(n/1000) h0, with indices n=0..11999. First change at sample 1001.
- x_raw=filter(1,[1,-1.6,0.95],v), white Gaussian v. 2000 samples are discarded and divided by the population standard deviation of the record; sample mean is not subtracted.
- y[n]=h[n]^T x_vec[n], d[n]=y[n]+independent Gaussian noise of variance 0.001.
- No algorithm receives the change indicator, resets weights, or changes parameters. The true state is only used to generate data and measure.
- The truly observed d[n-1],...,d[n-K+1] are stored; changing h does not rewrite the past.
- 64 realizations for calibration, seeds 310000..310063; 200 INDEPENDENT realizations for testing, 910000..910199. Noise seed is input seed +10000.
- q grid: [1,1.01,1.05,1.1,1.3,1.5,2,2.5,3,4]. eta grid: [1e-5,2.5e-5,5e-5,1e-4,2e-4:5e-5:8e-4,1e-3,1.5e-3,2e-3]. Divergent candidates are kept and marked; no realizations are removed from averages.
- Primary endpoint, fixed before calibration: clean-output prediction squared error, averaged over samples 3001..12000. The first 3000 are shown but excluded from the cyclic metric. BLMS and q-BLMS are optimized separately, using the same calibration data.
- Metrics: EMSE = mean((y-yhat)^2), NMSD = mean(||w-h||^2/||h||^2). Linear average over samples and realizations before converting to dB. EMSE here is clean prediction error; in a non-stationary scenario, a stationary Wiener excess is not assumed.
- Confidence interval: paired bootstrap of the 200 realizations, 5000 resamples, 2.5 and 97.5 percentiles of the mean ratio in dB.
- Controls: BLMS with the q-BLMS step size, BLMS tuned by NMSD, and separately tuned constant-leak BLMS. Constant leak uses D=(q_equiv-1) K I because input power is unitary.

| Algorithm | q / equivalent q | eta | Test EMSE (dB) | Test NMSD (dB) |
|---|---:|---:|---:|---:|
| BLMS tuned EMSE | 1 | 0.00055 | -6.473 | 0.487 |
| q-BLMS tuned EMSE | 2.5 | 0.00055 | -8.500 | -1.278 |
| BLMS same eta as q | 1 | 0.00055 | -6.473 | 0.487 |
| BLMS tuned NMSD | 1 | 0.0001 | -4.918 | -0.196 |
| Constant-leak BLMS tuned EMSE | 2.5 | 0.00055 | -8.546 | -1.297 |

The relatively near 0 dB NMSD indicates this is a difficult tracking problem, not precise identification of all coefficients. The prediction advantage should not be reinterpreted as stationary convergence to h.

**Secondary sensitivity with frozen parameters** (100 realizations per interval, distinct seeds):

| Interval between changes | EMSE gain q-BLMS vs BLMS (dB) | Paired 95% CI (dB) |
|---|---:|---:|
| 500 | 1.966 | [1.939, 1.992] |
| 1500 | 1.586 | [1.548, 1.622] |
| 3000 | 0.002 | [-0.058, 0.063] |

The scenario was selected after exploration; independent validation confirms its result conditional on this design, not the frequency of advantages across all possible systems. The repeated inversion experiment is synthetic and does not prove that these inversions are a validated model of a real acoustic channel.

## 5. Files and reproduction

- qblms_experiment.py: calibration, independent test, controls, and sensitivity.
- qblms_tracking_experiment.m: MATLAB replica with frozen parameters from results.json; requires Signal Processing Toolbox for fir1.
- matlab_verification.mat: same two realizations and expected results, to verify implementations without random generator differences.
- results.json, test_results.csv: parameters, statistics, and results per realization.
- calibration_sweep.csv, constant_leak_sweep.csv: all candidates, including divergences.
- special_exp_200.csv: alternative in the existing structure.
- test_curves.csv and tracking_comparison.png: curves and independent figure.

Python: run qblms_experiment.py (numpy, scipy, numba, matplotlib).
MATLAB: run qblms_tracking_experiment('verify') to check the same data; then qblms_tracking_experiment for an independent replica of 200 realizations.

External context reference: Kamenetsky and Widrow, A Variable Leaky LMS Adaptive Algorithm (2004), https://isl.stanford.edu/~widrow/papers/c2004avariable.pdf. Regularization reduces effective eigenvalue spread but introduces bias; the paper's own local analysis allows deriving this same tradeoff.
