# q-BLMS Adaptive Filtering

Python and MATLAB implementations supporting the paper **“A q-Jackson Block LMS Algorithm for Adaptive Filtering.”**

The repository includes an ECG artifact-cancellation example and a controlled system-identification benchmark with correlated inputs and periodic system changes.

## Algorithm

The q-BLMS update is obtained by applying the componentwise Jackson derivative to a quadratic block cost:

$$
\mathbf{w}_{n+1}
=
\mathbf{w}_n+2\eta\mathbf{X}_n\mathbf{e}_n
-\eta(q-1)\operatorname{Diag}
\left(\operatorname{diag}(\mathbf{X}_n\mathbf{X}_n^T)\right)\mathbf{w}_n.
$$

The implementation uses overlapping blocks and updates the coefficients at every sample.

For $q=1$, the diagonal correction vanishes and the algorithm reduces to the BLMS baseline used in these experiments. The update does not include affine projection normalization.

The file `q_japa_algorithm.py` and the function `q_japa_stream` retain their original names for compatibility with the ECG script.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── python/
│   ├── experiment_qblms.py
│   ├── exp1_ecg_cancellation.py
│   └── q_japa_algorithm.py
├── matlab/
│   ├── tracking_experiment_qblms.m
│   ├── exp_especial.m
│   ├── results.json
│   └── matlab_verification.mat
├── results/
│   ├── python/
│   └── matlab/
└── figures/
```

- `python/`: calibration, independent evaluation, and ECG demonstration.
- `matlab/`: independent tracking replication and an additional single-change example.
- `results/`: stored simulation outputs.
- `figures/`: figures prepared for inspection and reporting.

The current scripts save new outputs beside the script being executed. They do not automatically update the stored copies in `results/` or `figures/`.

## Requirements

### Python

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The tracking experiment uses NumPy, SciPy, Numba, and Matplotlib. The ECG example also uses WFDB and requires internet access to obtain the ECG record from PhysioNet.

### MATLAB

The tracking experiment requires Signal Processing Toolbox for `fir1`.

## 1. Python Calibration and Evaluation

From the repository root, run:

```bash
python python/experiment_qblms.py
```

The script:

1. Generates 64 calibration realizations.
2. Searches the step size for BLMS and the combination of q and step size for q-BLMS.
3. Tunes a constant-leak BLMS control separately.
4. Freezes the selected parameters.
5. Evaluates the configurations on 200 new realizations.
6. Runs additional sensitivity experiments.

The main outputs are written to `python/`:

| File | Content |
|---|---|
| `results.json` | Selected parameters, scenario, and evaluation summaries |
| `calibration_sweep.csv` | q-BLMS and BLMS calibration results |
| `constant_leak_sweep.csv` | Constant-leak calibration results |
| `results_test.csv` | Metrics for each evaluation realization |
| `curves_test.csv` | Averaged learning curves |
| `verification_matlab.mat` | Shared data for checking MATLAB against Python |
| `tracking_comparison.png` | Tracking comparison |
| `REPORT.md` | Detailed experimental report |

The first execution includes Numba compilation. The full calibration and evaluation may take substantial time.

## 2. MATLAB Replication

The MATLAB program reads the selected q and step-size values from `matlab/results.json`. It generates new random signals and computes its own metrics.

To use the stored configuration, copy `results/python/results.json` to `matlab/results.json`.

To use a newly generated Python configuration, copy:

```text
python/results.json
    → matlab/results.json

python/verification_matlab.mat
    → matlab/matlab_verification.mat
```

In MATLAB, open the `matlab` folder and run:

```matlab
results = tracking_experiment_qblms;
```

The program writes:

```text
matlab_replica.csv
matlab_replica.mat
matlab_replica.png
```

These files are saved in `matlab/`.

The MAT file includes metrics for each realization and the averaged curves. The current plot displays BLMS and q-BLMS; the output table also includes the additional controls.

### Numerical Verification

To compare both implementations using the same two realizations, run:

```matlab
tracking_experiment_qblms('verify');
```

This mode loads `matlab_verification.mat` and compares the MATLAB metrics with the values computed in Python. The implemented acceptance threshold is a maximum absolute difference below \(10^{-9}\).

Verification uses identical input data. The normal MATLAB experiment uses independent realizations, so its aggregate results can differ from the Python results.

## 3. ECG Demonstration

From the repository root, run:

```bash
python python/exp1_ecg_cancellation.py
```

The script downloads the first 3,000 samples of MIT-BIH record 100 through WFDB. A synthetic artifact generated from a correlated reference is added to the ECG.

The filter estimates the artifact, and the script displays the contaminated ECG, recovered signal, and clean reference.

This is an illustrative simulation. The animated display does not establish hardware real-time performance or clinical validity. The current script displays the figure interactively and does not automatically export it.

## Tracking Protocol

| Setting | Value |
|---|---|
| Filter length | 120 coefficients |
| Block length | 5 samples |
| Samples per realization | 12,000 |
| System changes | Polarity reversal every 1,000 samples |
| Observation-noise variance | 0.001 |
| Calibration realizations | 64 |
| Evaluation realizations | 200 |
| Evaluation window | Samples 3,001–12,000 |

The input is Gaussian noise filtered by:

\[
H(z)=\frac{1}{1-1.6z^{-1}+0.95z^{-2}}.
\]

The first 2,000 generated samples are discarded. Each retained input record is divided by its population standard deviation.

All algorithms receive the same data within each realization. They do not reset their coefficients or receive change notifications.

Python calibration uses input seeds `310000`–`310063`. Python evaluation uses `910000`–`910199`, with separate noise seeds obtained by adding 10,000. MATLAB evaluation uses `rng(1910000+r,'twister')`, for `r=1,...,200`.

## Metrics and Results

**Clean-output prediction MSE** compares the filter prediction before the update with the known system output before observation noise is added. The algorithms still adapt using noisy observations.

**NMSD** measures the coefficient error after the update, normalized by the squared norm of the true system.

Squared errors are averaged in linear scale before conversion to dB. The field `EMSE_dB` denotes clean-output prediction MSE in these files.

Representative evaluation results are:

| Evaluation | BLMS prediction MSE | q-BLMS prediction MSE | Reduction |
|---|---:|---:|---:|
| Python | −6.4732 dB | −8.4998 dB | 2.0266 dB |
| MATLAB | −6.4772 dB | −8.5336 dB | 2.0564 dB |

Both configurations use a step size of 0.00055; q-BLMS uses q = 2.5.

The Python confidence intervals apply to the Python evaluation realizations. They should not be assigned to the independent MATLAB results.

## Interpretation and Limitations

In the evaluated tracking scenario, q-BLMS produces smaller post-change error peaks, while BLMS reaches lower errors near the end of each constant-system interval. Constant-leak BLMS obtains similar average performance to q-BLMS.

The reported benefit concerns average tracking error, including transients. It does not demonstrate universal superiority, a lower stationary error floor, or superiority over APA or leakage-based alternatives.

The additional script `matlab/exp_especial.m` concerns a different single-change experiment. Its results should be distinguished from the periodic tracking benchmark.

## Citation

If you use this code, please cite the associated paper and the Zenodo record for the specific software version used in your experiments.

## License

The code is distributed under the MIT License. See `LICENSE`.

The ECG record is obtained from PhysioNet and is subject to the terms and citation requirements of its original source.
