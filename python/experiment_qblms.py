import numpy as np, json, time
from scipy.signal import lfilter, firwin
from scipy.io import loadmat
from numba import njit, prange, set_num_threads
set_num_threads(4)
@njit(parallel=True)
def simulate(xs, ds, hs, qs, mus, K, cuts, states, leakmode):
    runs,T=xs.shape; M=hs.shape[2]; C=len(qs); Wn=len(cuts)-1
    metrics=np.zeros((runs,C,Wn,2))
    for r in prange(runs):
        w=np.zeros((C,M)); X=np.zeros((K,M)); dn=np.zeros(K); alive=np.ones(C,np.bool_)
        for n in range(T):
            for k in range(K-1,0,-1):
                dn[k]=dn[k-1]
                for i in range(M): X[k,i]=X[k-1,i]
            dn[0]=ds[r,n]
            for i in range(M):
                if n>=i: X[0,i]=xs[r,n-i]
                else: X[0,i]=0.
            if n<M+K-2: continue
            diag=np.zeros(M)
            for i in range(M):
                for k in range(K): diag[i]+=X[k,i]**2
            phase=states[n]
            normh=0.
            for i in range(M): normh+=hs[r,phase,i]**2
            wi=-1
            wi=np.searchsorted(cuts,n,side='right')-1
            if wi>=Wn: wi=-1
            for c in range(C):
                if not alive[c]: continue
                pred=0.; target=0.
                for i in range(M):
                    pred+=w[c,i]*X[0,i]; target+=hs[r,phase,i]*X[0,i]
                er=(target-pred)**2
                grad=np.zeros(M)
                for k in range(K):
                    ek=dn[k]
                    for i in range(M): ek-=w[c,i]*X[k,i]
                    for i in range(M): grad[i]+=X[k,i]*ek
                msd=0.
                for i in range(M):
                    energy=diag[i] if leakmode[c]==0 else float(K)
                    w[c,i]+=2*mus[c]*grad[i]-mus[c]*(qs[c]-1)*energy*w[c,i]
                    msd+=(w[c,i]-hs[r,phase,i])**2
                if not np.isfinite(msd) or msd>1e12:
                    alive[c]=False
                    for z in range(Wn):
                        metrics[r,c,z,0]=np.inf; metrics[r,c,z,1]=np.inf
                    continue
                if wi>=0:
                    metrics[r,c,wi,0]+=msd/normh/(cuts[wi+1]-max(cuts[wi],M+K-2))
                    metrics[r,c,wi,1]+=er/(cuts[wi+1]-max(cuts[wi],M+K-2))
    return metrics
def dataset(kind,runs=16,T=6000,M=120,seed=1000,noise=.001):
    xs=np.zeros((runs,T)); ds=xs.copy(); hs=np.zeros((runs,2,M))
    for r in range(runs):
        rng=np.random.default_rng(seed+r)
        if kind=='white':
            x=rng.normal(size=T)
        elif kind=='narrow':
            x=lfilter([1],[1,-1.6,.95],rng.normal(size=T+2000))[2000:]; x=x/np.std(x)
        elif kind=='ar':
            x=lfilter([np.sqrt(1-.95**2)],[1,-.95],rng.normal(size=T+2000))[2000:]
        h=firwin(M,.4)
        h2=-h
        d=lfilter(h,[1],x); d[T//2:]=lfilter(h2,[1],x)[T//2:]
        xs[r]=x; ds[r]=d+np.sqrt(noise)*rng.normal(size=T); hs[r,0]=h;hs[r,1]=h2
    return xs,ds,hs

from pathlib import Path
from scipy.io import savemat
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
def periodic_data(runs,seed,interval=1000,T=12000):
    xs,ds,hs=dataset('narrow',runs=runs,T=T,seed=seed,noise=.001)
    states=(np.arange(T)//interval)%2
    for r in range(runs):
        rng=np.random.default_rng(seed+10000+r)
        ds[r]=lfilter(hs[r,0],[1],xs[r])*(1-2*states)+np.sqrt(.001)*rng.normal(size=T)
    return xs,ds,hs,states
def table(path,header,rows):
    with open(OUT/path,'w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)
def bootstrap(b,q,seed=777,B=5000):
    rng=np.random.default_rng(seed)
    ids=rng.integers(len(b),size=(B,len(b)))
    samples=10*np.log10(b[ids].mean(axis=1)/q[ids].mean(axis=1))
    return [float(v) for v in np.quantile(samples,[.025,.975])]
def run():
    # Scenario and primary endpoint fixed before this calibration/test split.
    # Calibration and test seeds have not appeared in the exploratory sweeps.
    steps=np.r_[1e-5,2.5e-5,5e-5,1e-4,np.arange(2e-4,8.01e-4,5e-5),.001,.0015,.002]
    qgrid=np.array([1.,1.01,1.05,1.1,1.3,1.5,2.,2.5,3.,4.])
    qs=np.repeat(qgrid,len(steps));mus=np.tile(steps,len(qgrid));mode=np.zeros(len(qs),np.int64)
    xs,ds,hs,states=periodic_data(64,310000)
    cal=simulate(xs,ds,hs,qs,mus,5,np.array([3000,12000]),states,mode)[:,:,0,:]
    avg=cal.mean(axis=0);avg[~np.isfinite(avg)]=np.inf
    valid=np.all(np.isfinite(cal),axis=(0,2))
    ids_b=np.flatnonzero((qs==1)&valid); ids_q=np.flatnonzero((qs>1)&valid)
    ib=int(ids_b[np.argmin(avg[ids_b,1])]);iq=int(ids_q[np.argmin(avg[ids_q,1])])
    ib_n=int(ids_b[np.argmin(avg[ids_b,0])])
    table('calibration_sweep.csv',['q','eta','NMSD_linear','EMSE_linear','finite_all_runs'],
          [[qs[j],mus[j],avg[j,0],avg[j,1],bool(valid[j])] for j in range(len(qs))])
    # Constant leakage comparator tuned independently using the same q parameterization.
    leak=simulate(xs,ds,hs,qs,mus,5,np.array([3000,12000]),states,np.ones(len(qs),np.int64))[:,:,0,:]
    la=leak.mean(axis=0);la[~np.isfinite(la)]=np.inf
    il=int(np.argmin(la[:,1]))
    table('constant_leak_sweep.csv',['q_equivalent','eta','NMSD_linear','EMSE_linear'],
          [[qs[j],mus[j],la[j,0],la[j,1]] for j in range(len(qs))])
    # Freeze all choices before drawing the independent 200-run test.
    labels=['BLMS tuned EMSE','q-BLMS tuned EMSE','BLMS same eta as q','BLMS tuned NMSD','Constant-leak BLMS tuned EMSE']
    qtest=np.array([1.,qs[iq],1.,1.,qs[il]])
    mutest=np.array([mus[ib],mus[iq],mus[iq],mus[ib_n],mus[il]])
    ltest=np.array([0,0,0,0,1],np.int64)
    print('Frozen parameters',list(zip(labels,qtest.tolist(),mutest.tolist())),flush=True)
    xs,ds,hs,states=periodic_data(200,910000)
    cuts=np.arange(0,12001,100)
    res=simulate(xs,ds,hs,qtest,mutest,5,cuts,states,ltest)
    # Last 9000 samples: bins 30..119, averaging linear errors first.
    per_run=res[:,:,30:,:].mean(axis=2)
    summary=[]
    for c,label in enumerate(labels):
        em=per_run[:,c,1]; nm=per_run[:,c,0]
        gain=float(10*np.log10(per_run[:,0,1].mean()/em.mean()))
        item=dict(label=label,q=float(qtest[c]),eta=float(mutest[c]),NMSD_dB=float(10*np.log10(nm.mean())),
                  EMSE_dB=float(10*np.log10(em.mean())),gain_EMSE_vs_BLMS_dB=gain,
                  gain_CI95=bootstrap(per_run[:,0,1],em),finite_all=bool(np.isfinite(per_run[:,c]).all()))
        summary.append(item)
    table('results_test.csv',['seed','algorithm','q','eta','NMSD_linear','EMSE_linear'],
          [[910000+r,labels[c],qtest[c],mutest[c],*per_run[r,c]] for r in range(200) for c in range(len(labels))])
    curve=res.mean(axis=0)
    table('curves_test.csv',['sample_end','algorithm','NMSD_linear','EMSE_linear'],
          [[int(cuts[t+1]),labels[c],*curve[c,t]] for c in range(len(labels)) for t in range(len(cuts)-1)])
    payload=dict(calibration_runs=64,test_runs=200,calibration_seed_start=310000,test_seed_start=910000,
                 scenario=dict(M=120,K=5,T=12000,change_interval=1000,noise_variance=.001,
                               input_denominator=[1,-1.6,.95],burn_in=2000,normalization='population std per realization',
                               h='firwin(120,0.4), equivalent Hamming-window FIR, unit DC gain',
                               primary_endpoint='mean clean-output squared prediction error, samples 3001..12000'),
                 results=summary)
    # Secondary sensitivity analyses reuse fixed parameters; no retuning.
    sensitivity=[]
    for interval in [500,1500,3000]:
        xx,dd,hh,st=periodic_data(100,1100000+interval,interval)
        mm=simulate(xx,dd,hh,qtest[:2],mutest[:2],5,np.array([3000,12000]),st,ltest[:2])[:,:,0,:]
        sensitivity.append(dict(interval=interval,EMSE_dB=(10*np.log10(mm.mean(axis=0)[:,1])).tolist(),
                                gain_dB=float(10*np.log10(mm[:,0,1].mean()/mm[:,1,1].mean())),
                                CI95=bootstrap(mm[:,0,1],mm[:,1,1])))
    payload['sensitivity_fixed_parameters']=sensitivity
    # Original exp_especial: same equation, honest buffers, reset h per run,
    # fixed parameters rather than tuning on these test observations.
    xx,dd,hh=dataset('narrow',runs=200,T=6000,seed=1510000,noise=.001)
    st=(np.arange(6000)>=3000).astype(np.int64)
    qq=np.array([1.,1.01,1.05,1.1,1.3]);ee=np.full(5,.0005)
    rr=simulate(xx,dd,hh,qq,ee,5,np.array([0,2500,3000,5500,6000]),st,np.zeros(5,np.int64))
    ar=rr.mean(axis=0)
    original=[dict(q=float(qq[c]),eta=.0005,
                   pre_NMSD_dB=float(10*np.log10(ar[c,1,0])),
                   post_transient_NMSD_dB=float(10*np.log10(ar[c,2,0])),
                   final_NMSD_dB=float(10*np.log10(ar[c,3,0])),
                   pre_EMSE_dB=float(10*np.log10(ar[c,1,1])),
                   post_transient_EMSE_dB=float(10*np.log10(ar[c,2,1])),
                   final_EMSE_dB=float(10*np.log10(ar[c,3,1]))) for c in range(5)]
    payload['original_narrow_single_change']=original
    table('exp_especial_200.csv',list(original[0]),[list(v.values()) for v in original])
    with open(OUT/'results.json','w',encoding='utf-8') as f:json.dump(payload,f,indent=2,allow_nan=False)
    # Export same exact realizations for MATLAB validation without RNG differences.
    savemat(OUT/'verification_matlab.mat',dict(xs=xs[:2],ds=ds[:2],hs=hs[:2],states=states,
             qs=qtest,mus=mutest,leakmode=ltest,cuts=cuts,expected=res[:2]),do_compression=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    colors=['#62666e','#006f9e','#cf7842','#9365a1','#298959']
    t=(cuts[1:]+cuts[:-1])/2
    for c in [0,1,4]:
        axs[0].plot(t[2:],10*np.log10(curve[c,2:,1]),label=labels[c],color=colors[c],lw=1.6)
    for jump in range(1000,12000,1000):axs[0].axvline(jump,color='#cccccc',lw=.6,zorder=0)
    axs[0].set(xlabel='Sample',ylabel='Clean-output prediction MSE (dB)',title='A. Tracking curve (200 independent runs)')
    axs[0].grid(alpha=.15);axs[0].legend(fontsize=8,loc='lower right')
    vals=[summary[c]['EMSE_dB'] for c in [0,1,4]]
    axs[1].bar(['BLMS','q-BLMS','Constant leak'],vals,color=[colors[c] for c in [0,1,4]])
    axs[1].set(ylabel='Mean prediction MSE (dB)',title='B. Samples 3001-12000',ylim=(min(vals)-1,0))
    for j,v in enumerate(vals):axs[1].text(j,v-.17,f'{v:.2f}',ha='center',va='top')
    fig.savefig(OUT/'tracking_comparison.png',dpi=220)
    plt.close(fig)
    write_report(payload)
    print(json.dumps(payload,indent=2),flush=True)

def write_report(payload):
    ss=payload['results']; orig=payload['original_narrow_single_change']
    q=ss[1]; b=ss[0]; leak=ss[4]
    rows='\n'.join(f"| {v['label']} | {v['q']:g} | {v['eta']:.6g} | {v['EMSE_dB']:.3f} | {v['NMSD_dB']:.3f} |" for v in ss)
    orows='\n'.join(f"| {v['q']:g} | {v['pre_NMSD_dB']:.3f} | {v['post_transient_NMSD_dB']:.3f} | {v['final_NMSD_dB']:.3f} | {v['post_transient_EMSE_dB']:.3f} |" for v in orig)
    sens='\n'.join(f"| {v['interval']} | {v['gain_dB']:.3f} | [{v['CI95'][0]:.3f}, {v['CI95'][1]:.3f}] |" for v in payload['sensitivity_fixed_parameters'])
    text=f"""# Analysis of q-BLMS and reproducible experiments

## Conclusion

There is no universal pair of q and eta that dominates BLMS. Under the white input and low noise conditions of the original examples, the regularization introduces a bias that harms the final error. In narrow band and frequent changes, it can reduce the average tracking error.

The main experiment uses **q={q['q']:g}, eta={q['eta']:.6g}** and improves the clean-output prediction error by **{q['gain_EMSE_vs_BLMS_dB']:.3f} dB** compared to BLMS with separately tuned eta. Paired 95% bootstrap CI: **[{q['gain_CI95'][0]:.3f}, {q['gain_CI95'][1]:.3f}] dB**. This claim is limited to the described synthetic scenario; it does not demonstrate universal superiority, nor superiority over APA or other regularizations.

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
{orows}

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
{rows}

The relatively near 0 dB NMSD indicates this is a difficult tracking problem, not precise identification of all coefficients. The prediction advantage should not be reinterpreted as stationary convergence to h.

**Secondary sensitivity with frozen parameters** (100 realizations per interval, distinct seeds):

| Interval between changes | EMSE gain q-BLMS vs BLMS (dB) | Paired 95% CI (dB) |
|---|---:|---:|
{sens}

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
"""
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')

if __name__=='__main__':run()
