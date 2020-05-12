# Cross-correlation of instantaneous amplitudes (under construction)

## Description

This is a small set of Python functions for computing the cross-correlation of instantaneous amplitudes, a method for estimating lead-lag directionality between two electroencephalogram (EEG) or local field potential (LFP) signals originally developed by Adhikari et al. [1]. This code was adapted directly from their original Matlab function, **amp_crosscorr.m**. In addition, I have included a simple function that creates a bootstrapped distribution in order to allow for non-parametric significance testing in line with the original paper.

## Usage

Simulate lead and lags signals with theta rhythm:
```Python
Signal simulation here
```

Compute cross-correlation and lag:
```Python
# amplitude cross-correlation
max_xcorr, max_xcorr_lag = amp_xcorr(lead_sig, lag_sig, fs, [4, 12])

print('Max correlation: ', max_xcorr)
print('Lag: ', max_xcorr_lag)
```

Bootstrap for significance:
```Python
# bootstrap resampling
bs_dist = bootstrap_xcorr(lead_sig, lag_sig, fs, [4, 12])

# plot distribution with orginial cross-correlation
plt.hist(bs_dist)
plt.xlim([-.2, 1.1])
plt.axvline(x=max_xcorr, color= 'r', linestyle='-')
plt.xlabel('Crosscorrelation')
plt.ylabel('Number of Resamples')
plt.legend(['Original xcorr', 'Bootstrap dist'])
```

## Future

Ideas for future work that I want to eventually get to.

* Improved bootstrap speed (parallel looping)
* Simple tutorial
* Scale up trials/subjects
* Stacked cross-correlation heatmaps over trials
* Output descriptives (significance/lag)
* Amplitude cross-correlations over time

## Credits

1. Adhikari, A., Sigurdsson, T., Topiwala, M.A. and Gordon, J.A. 2010. Journal of Neuroscience Methods 191(2), pp. 191–200. DIO: 10.1016/j.jneumeth.2010.06.019

2. xcorr() python function: https://github.com/colizoli/xcorr_python

3. Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for neural digital signal processing. Journal of Open Source Software, 4(36), 1272. DOI: 10.21105/joss.01272
