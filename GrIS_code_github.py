import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelmax
from statsmodels.tsa.arima_model import ARMA
import statsmodels.regression.linear_model as sm
from scipy.optimize import curve_fit, fmin, minimize, leastsq
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.ndimage import gaussian_filter1d
import scipy.stats as st

def fourrier_surrogates(ts, ns):
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def kendall_tau_test(ts, ns, tau, mode1 = 'fourier', mode2 = 'linear'):
    tlen = ts.shape[0]

    if mode1 == 'fourier':
        tsf = ts - ts.mean()
        nts = fourrier_surrogates(tsf, ns)
    elif mode1 == 'shuffle':
        nts = shuffle_surrogates(ts, ns)
    stat = np.zeros(ns)
    tlen = nts.shape[1]
    if mode2 == 'linear':
        for i in range(ns):
            stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
    elif mode2 == 'kt':
        for i in range(ns):
            stat[i] = st.kendalltau(np.arange(tlen), nts[i])[0]

    p = 1 - st.percentileofscore(stat, tau) / 100.
    return p

def runstd(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xw = x[: i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1


          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan
   for i in range(n - w // 2, n):
      xw = x[i - w // 2 + 1:]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1


          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   for i in range(w // 2, n - w // 2):
      xw = x[i - w // 2 : i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   return xs

def runac(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xw = x[: i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1


          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(n - w // 2, n):
      xw = x[i - w // 2 + 1:]
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(w // 2, n - w // 2):
      xw = x[i - w // 2 : i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   return xs


def LF(t, a, b):
    return a * t + b

def dhdt1(T, a, b, c):
    if b <= 0:
        return 1e8
    else:
        return a * (T + b) + c

def dhdt1_jac(T, a, b, c):
    return np.array([(T + b), a, 1]).T

def dhdt2(T, a, b, c, d):
    if c <= 0:
        return 1e8
    else:
        return a * (T + b)**2 + c * (T + b) + d

def dhdt4(T, a, b, c, d):
    if c <= 0:
        return 1e8
    else:
        return a * (T + b)**4 + c * (T + b) + d

def dhdt8(T, a, b, c, d):
    if c <= 0:
        return 1e8
    else:
        return a * (T + b)**8 + c * (T + b) + d


def dhdt8_jac(T, a, b, c):
    return np.array([(T + b)**8, 8 * a * (T + b)**7, T + b]).T


def sim1(h0, t_range, T_range, params):
    a, b, c = params
    dt = .01
    t_range_sim = np.arange(t_range.min(), t_range.max() + 1, dt)
    T_range_sim = LF(t_range_sim, *popt_t_T)
    height = np.zeros_like(t_range_sim)
    for i in range(t_range_sim.shape[0] - 1):
        height[i + 1] = height[i] + 100 * (a * (height[i] + b) + c - (T_range_sim[i])) * dt  + 5 * np.sqrt(dt) * np.random.randn(1)
    return height

def sim8(h0, t_range, T_range, params):
    a, b, c, d = params
    dt = .01
    t_range_sim = np.arange(t_range.min(), t_range.max() + 1, dt)
    T_range_sim = LF(t_range_sim, *popt_t_T)
    height = np.zeros_like(t_range_sim)
    for i in range(t_range_sim.shape[0] - 1):
        height[i + 1] = height[i] + 100 * (a * (height[i] + b)**8 + c * (height[i] + b) + d - (T_range_sim[i])) * dt  + 5 * np.sqrt(dt) * np.random.randn(1)
    return height

def potential(h, T, params):
    a, b, c, d = params
    return -a / 9. * (h + b)**9 - c / 2. * (h + b)**2 - d  * h + (T) * h


def runac2(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)
    xs[:w//2] = np.nan
    xs[n - w//2:] = np.nan
    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean()
        p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
        xw = xw - p0 * np.arange(xw.shape[0]) - p1
        xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
    return xs



start_year = 1855

data = np.loadtxt('data/GrIS_R_T.txt')
age = data[:, 0][::-1]
temp = data[:, 1][::-1]
runoff = data[:, 2][::-1]

racmo_data = np.loadtxt('data/GrIS_M_T_racmo.txt')
racmo_temp = racmo_data[:, 0]
racmo_melt = racmo_data[:, 1]

GW_data = np.loadtxt('data/GrIS_M_T_GW_core.txt')
GW_age = GW_data[:, 0]
GW_racmo_temp = GW_data[:, 2]
GW_racmo_melt = GW_data[:, 1]
GW_mar_temp = GW_data[:, 4]
GW_mar_melt = GW_data[:, 3]

GC_data = np.loadtxt('data/GrIS_M_T_GC_core.txt')
GC_age = GC_data[:, 0]
GC_racmo_temp = GC_data[:, 2]
GC_racmo_melt = GC_data[:, 1]
GC_mar_temp = GC_data[:, 4]
GC_mar_melt = GC_data[:, 3]

D5_data = np.loadtxt('data/GrIS_M_T_D5_core.txt')
D5_age = D5_data[:, 0]
D5_racmo_temp = D5_data[:, 2]
D5_racmo_melt = D5_data[:, 1]
D5_mar_temp = D5_data[:, 4]
D5_mar_melt = D5_data[:, 3]

NU_data = np.loadtxt('data/GrIS_M_T_NU_core.txt')
NU_age = NU_data[:, 0]
NU_racmo_temp = NU_data[:, 2]
NU_racmo_melt = NU_data[:, 1]
NU_mar_temp = NU_data[:, 4]
NU_mar_melt = NU_data[:, 3]

stack_data = np.loadtxt('data/CWG_NU_melt_jja_temp.txt')
stack_age = stack_data[:, 0][::-1]
stack_jja_temp = stack_data[:, 1][::-1]
stack_cwg_melt = stack_data[:, 2][::-1]
stack_nu_melt = stack_data[:, 3][::-1]

runoff = runoff[np.where(age == 1675)[0][0]:]
temp = temp[np.where(age == 1675)[0][0]:]
age = age[np.where(age == 1675)[0][0]:]

stack_cwg_melt = stack_cwg_melt[np.where(stack_age == 1675)[0][0]:]
stack_nu_melt = stack_nu_melt[np.where(stack_age == 1675)[0][0]:]
stack_jja_temp = stack_jja_temp[np.where(stack_age == 1675)[0][0]:]
stack_age = stack_age[np.where(stack_age == 1675)[0][0]:]

swg_temps = np.loadtxt('data/swg_monthly_1785_2019.txt')
swg_age = swg_temps[:, 0]
swg_jja = np.mean(swg_temps[:, 6:9], axis = 1)
swg_jja = swg_jja[np.where(swg_age == start_year)[0][0]:-1]
swg_age = swg_age[np.where(swg_age == start_year)[0][0]:-1]

ilu_temps = np.loadtxt('data/ilullisat_monthly_1785_2019.txt')
ilu_age = ilu_temps[:, 0]
ilu_jja = np.mean(ilu_temps[:, 6:9], axis = 1)
ilu_jja = ilu_jja[np.where(ilu_temps == start_year)[0][0]:-1]
ilu_age = ilu_age[np.where(ilu_age == start_year)[0][0]:-1]



idx_1855 = int(np.where(age == 1855)[0])
idx_2000 = int(np.where(age == 2000)[0])
idx2_1855 = int(np.where(stack_age == 1855)[0])

runoff_log = np.log(runoff)
stack_nu_melt_log = np.log(stack_nu_melt + np.abs(stack_nu_melt.min()) + 1)
stack_cwg_melt_log = np.log(stack_cwg_melt + np.abs(stack_cwg_melt.min()) + 1)
stack_nu_melt_log = np.log(stack_nu_melt + np.abs(stack_nu_melt.min()) + 1)

age_dt = age[idx2_1855:idx_2000]
stack_age_dt = stack_age[idx2_1855:]


t_fit = stack_age[stack_age > start_year]
T_fit = stack_jja_temp[stack_age > start_year]

popt_t_T, cov = curve_fit(LF, t_fit, T_fit)
T_L_fit = LF(t_fit, *popt_t_T)

slopes_var = np.zeros((5, 9))
pvs_var = np.zeros((5, 9))
slopes_ac = np.zeros((5, 9))
pvs_ac = np.zeros((5, 9))

slopes_var_nu = np.zeros((5, 9))
pvs_var_nu = np.zeros((5, 9))
slopes_ac_nu = np.zeros((5, 9))
pvs_ac_nu = np.zeros((5, 9))


kt_var = np.zeros((5, 9))
ktpvs_var = np.zeros((5, 9))
kt_ac = np.zeros((5, 9))
ktpvs_ac = np.zeros((5, 9))

kt_var_nu = np.zeros((5, 9))
ktpvs_var_nu = np.zeros((5, 9))
kt_ac_nu = np.zeros((5, 9))
ktpvs_ac_nu = np.zeros((5, 9))


ssamps = 100000
c1 = 0

gsigmas = [20, 30, 40, 50, 60]
wss = [40, 50, 60, 70, 80, 90, 100, 110, 120]

for gsigma in gsigmas:
    c2 = 0
    for ws in wss:
        bound = ws // 2
        tlen = int(stack_age[idx2_1855 + ws:].shape[0])

        stack_jja_temp_dt = stack_jja_temp[idx2_1855:] - gaussian_filter1d(stack_jja_temp[idx2_1855:], gsigma)
        temp_dt = temp[idx2_1855:idx_2000] - gaussian_filter1d(temp[idx2_1855:idx_2000], gsigma)
        runoff_dt = runoff_log - gaussian_filter1d(runoff_log, gsigma)
        stack_cwg_melt_dt = stack_cwg_melt_log - gaussian_filter1d(stack_cwg_melt_log, gsigma)
        stack_nu_melt_dt = stack_nu_melt_log - gaussian_filter1d(stack_nu_melt_log, gsigma)


        fig = plt.figure(figsize = (8,14))
        ax = fig.add_subplot(511)
        ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(age, temp, color = 'r', label = 'Arctic temperature anomalies [${}^{\circ}$C]')
        ax.set_ylabel('Arctic temperature anomalies [${}^{\circ}$C]', color = 'r')
        ax.tick_params(axis='y', colors='red')

        plt.legend(loc = 2)
        ax2 = ax.twinx()
        ax2.plot(stack_age, stack_jja_temp, color = 'b', label = 'CWG JJA temperature [${}^{\circ}$C]')
        ax2.plot(t_fit, T_L_fit, color = 'b', ls = '--')
        ax2.set_ylabel('CWG JJA temperature [${}^{\circ}$C]', color = 'b')
        ax2.tick_params(axis='y', colors='blue')
        plt.legend(loc = 1)
        ax.set_xlim(start_year, 2020)

        ax = fig.add_subplot(512)
        ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        ax.plot(stack_age, stack_cwg_melt, color = 'b', label = 'CWG Stack melt')
        ax.plot(stack_age, stack_nu_melt, color = 'r', label = 'NU melt')
        ax.set_ylabel('Surface melt [z-scores]')
        ax.set_xlim(start_year, 2020)
        plt.legend(loc = 2)

        ax = fig.add_subplot(513)
        ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(stack_age, stack_cwg_melt_dt, color = 'b', ls = '-', label = 'CWG Stack melt')
        ax.plot(stack_age, stack_nu_melt_dt, color = 'r', ls = '-', label = 'NU melt')
        ax.set_ylabel('Detrended log(Surface melt)')
        plt.legend(loc = 2)
        ax.set_xlim(start_year, 2020)
        start_idx = int(np.where(age == 1870)[0])
        start_idx2 = int(np.where(stack_age == 1870)[0])

        ax = fig.add_subplot(514)
        ax.text(-.1, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')


        ax.plot(stack_age[idx2_1855 + ws:], runstd(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound]**2, color = 'b', label = r'CWG Stack melt')
        p0, p1 = np.polyfit(stack_age[idx2_1855 + ws:], runstd(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound]**2, 1)
        kt = st.kendalltau(np.arange(tlen), runstd(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound]**2)[0]
        pv1 = kendall_tau_test(runstd(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound]**2, ssamps, kt, mode2 = 'kt')
        pv = kendall_tau_test(runstd(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound]**2, ssamps, p0)
        if pv >= .001:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'b', label = r'$p = %.3f$'%pv)
        else:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'b', label = r'$p < 10^{-3}$')
        slopes_var[c1, c2] = p0
        pvs_var[c1, c2] = pv
        kt_var[c1, c2] = kt
        ktpvs_var[c1, c2] = pv1
        print("p value cwg melt std", pv)

        ax.plot(stack_age[idx2_1855 + ws:], runstd(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound]**2, color = 'r', label = r'NU melt')
        p0, p1 = np.polyfit(stack_age[idx2_1855 + ws:], runstd(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound]**2, 1)
        kt = st.kendalltau(np.arange(tlen), runstd(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound]**2)[0]
        pv1 = kendall_tau_test(runstd(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound]**2, ssamps, kt, mode2 = 'kt')
        pv = kendall_tau_test(runstd(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound]**2, ssamps, p0)
        if pv >= 0.001:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'r', label = r'$p = %.3f$'%pv)
        else:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'r', label = r'$p < 10^{-3}$')
        slopes_var_nu[c1, c2] = p0
        pvs_var_nu[c1, c2] = pv
        kt_var_nu[c1, c2] = kt
        ktpvs_var_nu[c1, c2] = pv1

        print("p value nu melt std", pv)

        ax.set_ylabel('Variance (w = %d yr)'%ws)
        plt.legend(loc = 2)
        ax.set_xlim(start_year, 2020)

        ax = fig.add_subplot(515)
        ax.text(-.1, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')


        ax.plot(stack_age[idx2_1855 + ws:], runac(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound], color = 'b', label = r'CWG Stack melt')
        p0, p1 = np.polyfit(stack_age[idx2_1855 + ws:], runac(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound], 1)
        kt = st.kendalltau(np.arange(tlen), runac(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound])[0]
        pv1 = kendall_tau_test(runac(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound], ssamps, kt, mode2 = 'kt')
        pv = kendall_tau_test(runac(stack_cwg_melt_dt, ws)[idx2_1855 + bound :-bound], ssamps, p0)
        if pv >= 0.001:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'b', label = r'$p = %.3f$'%pv)
        else:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'b', label = r'$p < 10^{-3}$')
        slopes_ac[c1, c2] = p0
        pvs_ac[c1, c2] = pv
        kt_ac[c1, c2] = kt
        ktpvs_ac[c1, c2] = pv1

        print("p value cwg melt ac", pv)

        ax.plot(stack_age[idx2_1855 + ws:], runac(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound], color = 'r', label = r'NU melt')
        p0, p1 = np.polyfit(stack_age[idx2_1855 + ws:], runac(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound], 1)
        kt = st.kendalltau(np.arange(tlen), runac(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound])[0]
        pv1 = kendall_tau_test(runac(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound], ssamps, kt, mode2 = 'kt')
        pv = kendall_tau_test(runac(stack_nu_melt_dt, ws)[idx2_1855 + bound :-bound], ssamps, p0)
        if pv >= 0.001:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'r', label = r'$p = %.3f$'%pv)
        else:
            ax.plot(stack_age[idx2_1855 + ws:], p0 * stack_age[idx2_1855 + ws:] + p1, ls = '--', color = 'r', label = r'$p < 10^{-3}$')
        slopes_ac_nu[c1, c2] = p0
        pvs_ac_nu[c1, c2] = pv
        kt_ac_nu[c1, c2] = kt
        ktpvs_ac_nu[c1, c2] = pv1

        print("p value nu melt ac", pv)

        ax.set_ylabel('AC1 (w = %d yr)'%ws)
        plt.legend(loc = 2)
        ax.set_xlim(start_year, 2020)
        ax.set_xlabel('Time [yr AD]')
        fig.savefig('pics/Fig1_melt_gs%d_ws%d.pdf'%(gsigma, ws), bbox_inches = 'tight')


        fig = plt.figure(figsize = (8,14))
        ax = fig.add_subplot(311)
        ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(age_dt, temp_dt, color = 'r', label = 'Arctic temperature anomalies [${}^{\circ}$C]')
        ax.set_ylabel('Arctic temperature anomalies [${}^{\circ}$C]', color = 'r')
        ax.tick_params(axis='y', colors='red')

        plt.legend(loc = 2)
        ax2 = ax.twinx()
        ax2.plot(stack_age_dt, stack_jja_temp_dt, color = 'b', label = 'CWG JJA temperature [${}^{\circ}$C]')
        ax2.set_ylabel('CWG JJA temperature [${}^{\circ}$C]', color = 'b')
        ax2.tick_params(axis='y', colors='blue')
        plt.legend(loc = 1)
        ax.set_xlim(start_year, 2020)



        ax = fig.add_subplot(312)
        ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        ax.plot(age_dt[ws:], runstd(temp_dt, ws)[bound :-bound]**2, color = 'r', label = 'Arctic temperature anomalies [${}^{\circ}$C]')
        p0, p1 = np.polyfit(age_dt[ws:], runstd(temp_dt, ws)[bound :-bound]**2, 1)
        pv = kendall_tau_test(runstd(temp_dt, ws)[bound :-bound]**2, ssamps, p0)
        ax.plot(age_dt[ws:], p0 * age_dt[ws:] + p1, ls = '--', color = 'r', label = 'p = %.5f'%pv)
        ax.set_ylabel('Variance (w = %d yr)'%ws, color = 'r')
        plt.legend(loc = 2)

        ax2 = ax.twinx()
        ax2.plot(stack_age_dt[ws:], runstd(stack_jja_temp_dt, ws)[bound :-bound]**2, color = 'b', label = 'CWG JJA temperature [${}^{\circ}$C]')
        p0, p1 = np.polyfit(stack_age_dt[ws:], runstd(stack_jja_temp_dt, ws)[bound :-bound]**2, 1)
        pv = kendall_tau_test(runstd(stack_jja_temp_dt, ws)[bound :-bound]**2, ssamps, p0)
        ax2.plot(stack_age_dt[ws:], p0 * stack_age_dt[ws:] + p1, ls = '--', color = 'b', label = 'p = %.5f'%pv)


        ax2.set_ylabel('Variance (w = %d yr)'%ws, color = 'b')
        plt.legend(loc = 1)
        ax.set_xlim(start_year, 2020)

        ax = fig.add_subplot(313)
        ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        ax.plot(age_dt[ws:], runac(temp_dt, ws)[bound :-bound], color = 'r', label = 'Arctic temperature anomalies [${}^{\circ}$C]', alpha = .8)
        p0, p1 = np.polyfit(age_dt[ws:], runac(temp_dt, ws)[bound :-bound], 1)
        pv = kendall_tau_test(runac(temp_dt, ws)[bound :-bound], ssamps, p0)
        ax.plot(age_dt[ws:], p0 * age_dt[ws:] + p1, ls = '--', color = 'r', label = 'p = %.5f'%pv)
        ax.set_ylabel('AC1 coefficient (w = %d yr)'%ws, color = 'r')
        plt.legend(loc = 2)

        ax2 = ax.twinx()
        ax2.plot(stack_age_dt[ws:], runac(stack_jja_temp_dt, ws)[bound :-bound], color = 'b', label = 'CWG JJA temperature [${}^{\circ}$C]')
        p0, p1 = np.polyfit(stack_age_dt[ws:], runac(stack_jja_temp_dt, ws)[bound :-bound], 1)
        pv = kendall_tau_test(runac(stack_jja_temp_dt, ws)[bound :-bound], ssamps, p0)
        ax2.plot(stack_age_dt[ws:], p0 * stack_age_dt[ws:] + p1, ls = '--', color = 'b', label = 'p = %.5f'%pv)


        ax2.set_ylabel('AC1 coefficient (w = %d yr)'%ws, color = 'b')
        plt.legend(loc = 1)
        ax.set_xlim(start_year, 2020)
        fig.savefig('pics/Arctic_temps_EWS_new_gs%d_ws%d.pdf'%(gsigma, ws), bbox_inches = 'tight')
        c2+=1
    c1+=1
np.save('data/slopes_var', slopes_var)
np.save('data/slopes_ac', slopes_ac)
np.save('data/pvs_var', pvs_var)
np.save('data/pvs_ac', pvs_ac)

np.save('data/slopes_var_nu', slopes_var_nu)
np.save('data/slopes_ac_nu', slopes_ac_nu)
np.save('data/pvs_var_nu', pvs_var_nu)
np.save('data/pvs_ac_nu', pvs_ac_nu)

np.save('data/kt_var', kt_var)
np.save('data/kt_ac', kt_ac)
np.save('data/ktpvs_var', ktpvs_var)
np.save('data/ktpvs_ac', ktpvs_ac)

np.save('data/kt_var_nu', kt_var_nu)
np.save('data/kt_ac_nu', kt_ac_nu)
np.save('data/ktpvs_var_nu', ktpvs_var_nu)
np.save('data/ktpvs_ac_nu', ktpvs_ac_nu)


slopes_var = np.load('data/slopes_var.npy')
slopes_var_mask = np.ones_like(slopes_var) * .5
slopes_ac = np.load('data/slopes_ac.npy')
slopes_ac_mask = np.ones_like(slopes_ac) * .5
pvs_var = np.load('data/pvs_var.npy')
pvs_ac = np.load('data/pvs_ac.npy')

slopes_var_nu = np.load('data/slopes_var_nu.npy')
slopes_var_nu_mask = np.ones_like(slopes_var_nu) * .5
slopes_ac_nu = np.load('data/slopes_ac_nu.npy')
slopes_ac_nu_mask = np.ones_like(slopes_ac_nu) * .5
pvs_var_nu = np.load('data/pvs_var_nu.npy')
pvs_ac_nu = np.load('data/pvs_ac_nu.npy')

slopes_var[pvs_var > 0.05] = np.nan
slopes_var_mask[pvs_var <= 0.05] = np.nan
slopes_ac[pvs_ac > 0.05] = np.nan
slopes_ac_mask[pvs_ac <= 0.05] = np.nan
slopes_var_nu[pvs_var_nu > 0.05] = np.nan
slopes_var_nu_mask[pvs_var_nu <= 0.05] = np.nan
slopes_ac_nu[pvs_ac_nu > 0.05] = np.nan
slopes_ac_nu_mask[pvs_ac_nu <= 0.05] = np.nan

kt_var = np.load('data/kt_var.npy')
kt_var_mask = kt_var.copy()
kt_ac = np.load('data/kt_ac.npy')
kt_ac_mask = kt_ac.copy()
ktpvs_var = np.load('data/ktpvs_var.npy')
ktpvs_ac = np.load('data/ktpvs_ac.npy')

kt_var_nu = np.load('data/kt_var_nu.npy')
kt_var_nu_mask = kt_var_nu.copy()
kt_ac_nu = np.load('data/kt_ac_nu.npy')
kt_ac_nu_mask = kt_ac_nu.copy()
ktpvs_var_nu = np.load('data/ktpvs_var_nu.npy')
ktpvs_ac_nu = np.load('data/ktpvs_ac_nu.npy')

kt_var[ktpvs_var > 0.05] = np.nan
kt_ac[ktpvs_ac > 0.05] = np.nan
kt_var_nu[ktpvs_var_nu > 0.05] = np.nan
kt_ac_nu[ktpvs_ac_nu > 0.05] = np.nan

kt_var_mask[ktpvs_var <= 0.05] = np.nan
kt_ac_mask[ktpvs_ac <= 0.05] = np.nan
kt_var_nu_mask[ktpvs_var_nu <= 0.05] = np.nan
kt_ac_nu_mask[ktpvs_ac_nu <= 0.05] = np.nan


fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(221)
ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'CWG Var', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(slopes_var, vmin = 0, vmax = .002, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(slopes_var_mask, vmin = 0, vmax = .002, cmap = 'Greys')
cbar.ax.set_ylabel('Linear slopes')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(222)
ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'CWG AC1', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(slopes_ac, vmin = 0, vmax = .008, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(slopes_ac_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel('Linear slopes')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(223)
ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'NU Var', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(slopes_var_nu, vmin = 0, vmax = .0008, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(slopes_var_nu_mask, vmin = 0, vmax = .001, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel('Linear slopes')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(224)
ax.text(-.1, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'NU AC1', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(slopes_ac_nu, vmin = 0, vmax = .004, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(slopes_ac_nu_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel('Linear slopes')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')


fig.savefig('pics/slopes_pvs.pdf', bbox_inches = 'tight')

fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(221)
ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'CWG Var', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(kt_var, vmin = 0, vmax = 1, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(kt_var_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel(r'Kendal $\tau$')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(222)
ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'CWG AC1', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(kt_ac, vmin = 0, vmax = 1, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(kt_ac_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel(r'Kendal $\tau$')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(223)
ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'NU Var', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(kt_var_nu, vmin = 0, vmax = 1, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(kt_var_nu_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel(r'Kendal $\tau$')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')

ax = fig.add_subplot(224)
ax.text(-.1, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.text(.6, 1.1, s = 'NU AC1', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
c = ax.imshow(kt_ac_nu, vmin = 0, vmax = 1, cmap = 'plasma')
cbar = fig.colorbar(c, ax = ax, extend = 'both')
ax.imshow(kt_ac_nu_mask, vmin = 0, vmax = .002, cmap = 'Greys', alpha = .5)
cbar.ax.set_ylabel(r'Kendal $\tau$')
ax.set_xticks(np.arange(len(wss)))
ax.set_yticks(np.arange(len(gsigmas)))
ax.set_xticklabels(wss)
ax.set_yticklabels(gsigmas)
ax.set_xlabel('sliding window size')
ax.set_ylabel('Detrending bandwidth')


fig.savefig('pics/kts_pvs.pdf', bbox_inches = 'tight')




melt_offset = .4

melt_offset2 = 1.16

fit_start = 5.0

fit_end = 7.1
fit_end_2 = 7.1

h_range = np.arange(-3500, 200, 0.01)

dat = 'melt'

stack_cwg_melt_fit1 = stack_cwg_melt[stack_age > start_year] - melt_offset * stack_cwg_melt[stack_age > start_year].min()


bk_dat = np.loadtxt('data/Box_Kjeldsen.txt')
bk_age = bk_dat[:, 0]
bk_acc = bk_dat[:, 1]
bk_runoff = bk_dat[:, 3]
bk_discharge = bk_dat[:, 5]
bk_tmb = bk_dat[:, 7]
bk_start_idx = np.where(bk_age == start_year)[0][0]
bk_age = bk_age[bk_start_idx:]
bk_acc = bk_acc[bk_start_idx:]
bk_runoff = bk_runoff[bk_start_idx:]
bk_discharge = bk_discharge[bk_start_idx:]
bk_tmb_nf = bk_tmb[bk_start_idx:]
cum_bk_tmb_nf = np.cumsum(bk_tmb_nf)



for melt_offset in [.55, .6, .65]:
    for ww in [40,50, 60]:
        z_bk_acc = (bk_acc - np.mean(bk_acc)) / np.std(bk_acc)
        stack_cwg_melt_fit = stack_cwg_melt[stack_age > start_year] - stack_cwg_melt[stack_age > start_year].min() - .1 * melt_offset * T_fit

        bk_tmb = gaussian_filter1d(bk_acc, 20) - bk_runoff - bk_discharge
        runoff_fit =  - runoff[stack_age > start_year] + gaussian_filter1d(bk_acc, 20) - .93 * bk_discharge.mean()

        if dat == 'melt':
            Melt_fit = -np.cumsum(stack_cwg_melt_fit)
        elif dat == 'runoff':
            Melt_fit = np.cumsum(runoff_fit)

        model_age = GW_data[:, 0]

        con_idx = np.where(t_fit == model_age[0])[0][0]

        T_L_fit_model = T_L_fit[con_idx:]
        t_fit_model = t_fit[con_idx:]

        fac = 1

        GW_racmo_melt2 = (GW_racmo_melt - GW_racmo_melt[:30].mean()) / GW_racmo_melt[:30].std()
        GC_racmo_melt2 = (GC_racmo_melt - GC_racmo_melt[:30].mean()) / GC_racmo_melt[:30].std()
        D5_racmo_melt2 = (D5_racmo_melt - D5_racmo_melt[:30].mean()) / D5_racmo_melt[:30].std()
        GW_racmo_melt3 = GW_racmo_melt2[:-3] - GW_racmo_melt2.min() - .1 * melt_offset * T_fit[con_idx :]
        GC_racmo_melt3 = GC_racmo_melt2[:-3] - GC_racmo_melt2.min() - .1 * melt_offset * T_fit[con_idx : ]
        D5_racmo_melt3 = D5_racmo_melt2[:-3] - D5_racmo_melt2.min() - .1 * melt_offset * T_fit[con_idx : ]


        Melt_fit_GW = Melt_fit[con_idx] + 5 - fac * np.cumsum(GW_racmo_melt3)
        Melt_fit_GC = Melt_fit[con_idx] + 5 - fac * np.cumsum(GC_racmo_melt3)
        Melt_fit_D5 = Melt_fit[con_idx] + 5 - fac * np.cumsum(D5_racmo_melt3)


        fitting_range = np.logical_and(T_L_fit > fit_start, T_L_fit < fit_end)
        fitting_range_68 = np.logical_and(T_L_fit > fit_start, T_L_fit < fit_end_2)


        Melt_fit1 = Melt_fit[fitting_range]
        T_L_fit1 = T_L_fit[fitting_range]
        t_fit1 = t_fit[fitting_range]

        Melt_fit1_68 = Melt_fit[fitting_range_68]
        T_L_fit1_68 = T_L_fit[fitting_range_68]
        t_fit1_68 = t_fit[fitting_range_68]

        p_guess1 = [-4.64970085e-2,  1.64034472e+02,  6.21613019e+00]
        p_guess2 = [-4.64970085e-2,  1.64034472e+02,  6.44285233e-03,  6.21613019e+00]
        p_guess8 = [-4.64970085e-18,  1.64034472e+02,  6.44285233e-03,  6.21613019e+00]

        popt2, cov2 = curve_fit(dhdt2, Melt_fit1, T_L_fit1, maxfev = 100000, bounds = ((-np.inf, 0, 0, 6.), (-.1, np.inf, 1e-7, np.inf)))
        popt2_68, cov2_68 = curve_fit(dhdt2, Melt_fit1_68, T_L_fit1_68, p0 = p_guess2, maxfev = 100000)

        popt8, cov8 = curve_fit(dhdt8, Melt_fit1, T_L_fit1, p0 = p_guess8, maxfev = 10000000)

        popt8_68, cov8_68 = curve_fit(dhdt8, Melt_fit1_68, T_L_fit1_68, p0 = p_guess8, maxfev = 10000000)

        popt1, cov1 = curve_fit(dhdt1, Melt_fit1, T_L_fit1, p0 = p_guess1, maxfev = 10000000)
        popt1_68, cov1_68 = curve_fit(dhdt1, Melt_fit1_68, T_L_fit1_68, p0 = p_guess1, maxfev = 10000000)


        nsim = 10

        t_fit_future = np.arange(start_year, 2050)
        t_fit_future_hres = np.arange(start_year, 2050, .01)
        T_L_fit_future = LF(t_fit_future, *popt_t_T)
        T_L_fit_future_hres = LF(t_fit_future_hres, *popt_t_T)
        simulation = np.zeros((nsim, t_fit_future.shape[0]))
        simulation_est = np.zeros((nsim, 100 * t_fit_future.shape[0]))

        simulation_lin = np.zeros((nsim, t_fit_future.shape[0]))
        simulation_est_lin = np.zeros((nsim, 100 * t_fit_future.shape[0]))

        for i in range(nsim):
            simulation_est[i] = sim8(Melt_fit[0], t_fit_future, T_L_fit_future, popt8)
            simulation[i] = simulation_est[i][::100]

            simulation_est_lin[i] = sim1(Melt_fit[0], t_fit_future, T_L_fit_future, popt1)
            simulation_lin[i] = simulation_est_lin[i][::100]

        simulation[simulation < -250] = np.nan
        simulation[simulation < -2500] = np.nan


        simulation_lin[simulation_lin < -250] = np.nan
        simulation_lin[simulation_lin < -2500] = np.nan


        h_range_stab = h_range[h_range >= h_range[dhdt8(h_range, *popt8).argmax()]]
        h_range_unstab = h_range[h_range < h_range[dhdt8(h_range, *popt8).argmax()]]

        h_range_stab_68 = h_range[h_range >= h_range[dhdt8(h_range, *popt8_68).argmax()]]
        h_range_unstab_68 = h_range[h_range < h_range[dhdt8(h_range, *popt8_68).argmax()]]


        T1 = dhdt1(h_range, *popt1)
        T_max1 = T1.max()
        hmin1 = h_range[T1 == T_max1]
        h_range1 = h_range[h_range >= hmin1]
        T1 = T1[h_range >= hmin1]
        f1 = interp1d(T1, h_range1)

        T2 = dhdt2(h_range, *popt2)
        T_max2 = T2.max()
        hmin2 = h_range[T2 == T_max2]
        h_range2 = h_range[h_range >= hmin2]
        T2 = T2[h_range >= hmin2]
        f2 = interp1d(T2, h_range2)

        T8 = dhdt8(h_range, *popt8)
        T_max8 = T8.max()
        hmin8 = h_range[T8 == T_max8]
        h_range8 = h_range[h_range >= hmin8]
        T8 = T8[h_range >= hmin8]
        f8 = interp1d(T8, h_range8)

        T8_68 = dhdt8(h_range, *popt8_68)
        T_max8_68 = T8_68.max()
        hmin8_68 = h_range[T8_68 == T_max8_68]
        h_range8_68 = h_range[h_range >= hmin8_68]
        T8_68 = T8_68[h_range >= hmin8_68]
        f8_68 = interp1d(T8_68, h_range8_68)


        t_range_sim = np.arange(t_fit_future.min(), t_fit_future.max() + 1, .01)
        T_range_sim = LF(t_range_sim, *popt_t_T)

        intp_range1 = np.logical_and(T_L_fit1 >= T1.min(), T_L_fit1 <= T1.max())

        intp_range2 = np.logical_and(T_L_fit1 >= T2.min(), T_L_fit1 <= T2.max())
        intp_range8 = np.logical_and(T_L_fit1 >= T8.min(), T_L_fit1 <= T8.max())
        intp_range8_68 = np.logical_and(T_L_fit1_68 >= T8_68.min(), T_L_fit1_68 <= T8_68.max())

        intp_range1_hr = np.logical_and(T_range_sim >= T1.min(), T_range_sim <= T1.max())
        intp_range8_hr = np.logical_and(T_range_sim >= T8.min(), T_range_sim <= T8.max())
        intp_range8_68_hr = np.logical_and(T_range_sim >= T8_68.min(), T_range_sim <= T8_68.max())


        T_L_fit2 = T_L_fit1[intp_range2]
        Melt_fit2 = Melt_fit1[intp_range2]
        t_fit2 = t_fit1[intp_range2]

        T_L_fit_1 = T_L_fit1[intp_range8]
        T_L_fit8 = T_L_fit1[intp_range8]
        T_L_fit8_68 = T_L_fit1_68[intp_range8_68]

        Melt_fit_1 = Melt_fit1[intp_range8]
        Melt_fit8 = Melt_fit1[intp_range8]
        Melt_fit8_68 = Melt_fit1_68[intp_range8_68]

        t_fit_1 = t_fit1[intp_range8]
        t_fit8 = t_fit1[intp_range8]
        t_fit8_68 = t_fit1_68[intp_range8_68]

        Melt_fit_itp_o1 = f1(T_L_fit1[intp_range8])
        Melt_fit_itp_o2 = f2(T_L_fit1[intp_range2])
        Melt_fit_itp_o8 = f8(T_L_fit1[intp_range8])
        Melt_fit_itp_o8_68 = f8_68(T_L_fit1_68[intp_range8_68])

        Melt_fit_itp_o1_hr = f1(T_range_sim[intp_range8_hr])
        Melt_fit_itp_o8_hr = f8(T_range_sim[intp_range8_hr])
        Melt_fit_itp_o8_68_hr = f8_68(T_range_sim[intp_range8_68_hr])



        fluct2 = Melt_fit2 - Melt_fit_itp_o2
        std2 = runstd(fluct2, ww)
        ac2 = runac(fluct2, ww)
        dhdT2 = -runlf(T_L_fit2, Melt_fit2, ww)

        fluct1 = Melt_fit_1 - Melt_fit_itp_o1
        std1 = runstd(fluct1, ww)

        fluct8 = Melt_fit8 - Melt_fit_itp_o8

        R2 = 1 - np.sum(fluct8**2) / np.sum((Melt_fit8 - Melt_fit8.mean())**2)
        print('R2=', R2)
        std8 = runstd(fluct8, ww)
        fluct8_68 = Melt_fit8_68 - Melt_fit_itp_o8_68
        std8_68 = runstd(fluct8_68, ww)

        ac1 = runac(fluct1, ww)
        ac8 = runac(fluct8, ww)
        ac8_68 = runac(fluct8_68, ww)

        dhdT1 = -runlf(T_L_fit_1, Melt_fit_1, ww)
        dhdT8 = -runlf(T_L_fit8, Melt_fit8, ww)
        dhdT8_68 = -runlf(T_L_fit8_68, Melt_fit8_68, ww)

        dhdT_model_o1 = -runlf(T_L_fit_1, Melt_fit_itp_o1, ww)

        dhdT_model = -runlf(T_L_fit8, Melt_fit_itp_o8, ww)

        fluct_sim1 = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))
        fluct_sim_hr1 = np.zeros((nsim, Melt_fit_itp_o8_hr.shape[0]))
        std_sim1 = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))
        ac_sim1 = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))

        fluct_sim = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))
        fluct_sim_hr = np.zeros((nsim, Melt_fit_itp_o8_hr.shape[0]))
        std_sim = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))
        ac_sim = np.zeros((nsim, Melt_fit_itp_o8.shape[0]))
        for i in range(nsim):
            fluct_sim1[i] = simulation_lin[i][:Melt_fit_itp_o8.shape[0]] - Melt_fit_itp_o1
            fluct_sim_hr1[i] = simulation_est_lin[i][:Melt_fit_itp_o8_hr.shape[0]] - Melt_fit_itp_o8_hr
            std_sim1[i] = runstd(fluct_sim1[i], ww)
            ac_sim1[i] = runac(fluct_sim_hr1[i], ww * 100)[::100][:-3]

            fluct_sim[i] = simulation[i][:Melt_fit_itp_o8.shape[0]] - Melt_fit_itp_o8
            fluct_sim_hr[i] = simulation_est[i][:Melt_fit_itp_o8_hr.shape[0]] - Melt_fit_itp_o8_hr
            std_sim[i] = runstd(fluct_sim[i], ww)
            ac_sim[i] = runac(fluct_sim_hr[i], ww * 100)[::100][:-3]




        std_mean_sim1 = np.mean(std_sim1, axis = 0)
        std_std_sim1 = np.std(std_sim1, axis = 0)


        std_mean_sim = np.mean(std_sim, axis = 0)
        std_std_sim = np.std(std_sim, axis = 0)


        ac_mean_sim1 = np.mean(ac_sim1, axis = 0)
        ac_std_sim1 = np.std(ac_sim1, axis = 0)


        ac_mean_sim = np.mean(ac_sim, axis = 0)
        ac_std_sim = np.std(ac_sim, axis = 0)



        fig = plt.figure(figsize = (6, 12))

        ax = fig.add_subplot(411)
        ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(T_L_fit, Melt_fit, color = 'k', label = 'Reconstructed data', lw = 2, alpha = .9)
        ax.plot(dhdt8(h_range_stab, *popt8), h_range_stab, color = 'b', alpha = .7, label = r'Model Fixed Point')
        ax.plot(dhdt8(h_range_unstab, *popt8), h_range_unstab, color = 'b', alpha = .7, ls = '--')


        ax.grid()
        ax.legend()
        ax.set_xlim((4.9, 7.3))
        ax.set_ylim((-250, 50))
        ax.set_ylabel(r'Height change $\Delta h$ [a.u.]')
        ax.axvline(x = T8.max(), color = 'r', ls = '-.')




        ax2 = ax.twiny()
        ax2.plot(t_fit, Melt_fit, color = 'k', label = 'Reconstructed data', lw = 2, alpha = .9)
        ax2.set_xlim(((4.9 - popt_t_T[1]) / popt_t_T[0], (7.3 - popt_t_T[1]) / popt_t_T[0]))
        ax2.set_xlabel('Time [yr AD]')

        ax = fig.add_subplot(412)
        ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(T_L_fit8, fluct8, color = 'k', label = r'$\Delta h - \Delta h^{*}$ (obs.)')
        ax.legend()
        ax.grid()
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel('Fluctuations around $h^{*}$ [a.u.]')
        ax.axvline(x = T8.max(), color = 'r', ls = '-.')
        guesses = [1, 1, 49]
        tssamp = 4
        ax = fig.add_subplot(413)
        ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], std8[ww//2: -ww//2][::tssamp]**2, color = 'k', marker = 'o', alpha = .8, label = r'Variance obs.')
        ax.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], std_mean_sim[ww//2: -ww//2][::tssamp]**2, color = 'r', marker = '+', alpha = .99, label = r'Variance sim.')

        ax.axvline(x = T8_68.max(), color = 'r', ls = '-.')
        ax.legend(loc = 2)
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel(r'Variance', color = 'k')
        ax2 = ax.twinx()
        ax2.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], dhdT8[ww//2: -ww//2][::tssamp], color = 'b', marker = 'x', alpha = .8, label = r'$|d\Delta h/dT|$ obs.')

        ax2.plot(T_L_fit8[ww//2: -ww//2], dhdT_model[ww//2: -ww//2], color = 'b', ls = '--', lw = 2, label = r'$|d\Delta h/dT|$ model')

        ax2.set_ylabel(r'Sensitivity $|d\Delta h/dT|$', color = 'b')
        ax2.legend(loc = 4)
        ax.set_xlim((4.9, 7.3))
        ax2.set_ylim((0, 60))



        ax = fig.add_subplot(414)
        ax.text(-.15, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], ac8[ww//2: -ww//2][::tssamp], color = 'k', marker = 'o', alpha = .8, label = r'AC1 obs.')
        ax.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], ac_mean_sim[ww//2: -ww//2][::tssamp], color = 'r', marker = '+', alpha = .8, label = r'AC1 sim.')
        p0, p1 = np.polyfit(T_L_fit8[ww//2: -ww//2], ac8[ww//2: -ww//2], 1)
        ax.plot(T_L_fit8[ww//2: -ww//2], p0 * T_L_fit8[ww//2: -ww//2] + p1, 'k--')


        p0, p1 = np.polyfit(T_L_fit8[ww//2: -ww//2], ac_mean_sim[ww//2: -ww//2], 1)
        ax.plot(T_L_fit8[ww//2: -ww//2], p0 * T_L_fit8[ww//2: -ww//2] + p1, 'r--')
        ax.axvline(x = T8.max(), color = 'r', ls = '-.')
        ax.legend(loc = 2)
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel(r'AC1', color = 'k')
        ax.set_xlabel(r'CWG JJA Temperature $T$ [${}^{\circ}$C]')
        ax2 = ax.twinx()
        ax2.scatter(T_L_fit8[ww//2: -ww//2][::tssamp], np.exp(-1 / dhdT8[ww//2: -ww//2])[::tssamp], color = 'b', marker = 'x', alpha = .8, label = r'$e^{1/(d\Delta h/dT)}$ obs.')
        ax2.plot(T_L_fit8[ww//2: -ww//2][::tssamp], np.exp(-1 / dhdT_model[ww//2: -ww//2])[::tssamp], color = 'b', ls = '--', lw = 2, label = r'$e^{1/(d\Delta h/dT)}$ model')


        ax2.set_ylabel(r' $e^{1/(d\Delta h/dT)}$', color = 'b')
        ax2.legend(loc = 4)
        ax.set_xlim((4.9, 7.3))
        ax2.set_ylim((.75, 1.))



        fig.savefig('pics/sensitivity_std_ac1_ews_vs_Temp_mo%d_fs%d_fe%d_%s_order2_ww%d_test.pdf'%(melt_offset * 100, fit_start * 10, fit_end * 10, dat, ww), bbox_inches = 'tight')



        fig = plt.figure(figsize = (6, 12))

        ax = fig.add_subplot(411)
        ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(T_L_fit, Melt_fit, color = 'k', label = 'Reconstructed data', lw = 2, alpha = .9)
        ax.plot(dhdt1(h_range_stab, *popt1), h_range_stab, color = 'b', alpha = .7, label = r'Modelled Fixed Point')
        ax.plot(dhdt1(h_range_unstab, *popt1), h_range_unstab, color = 'b', alpha = .7, ls = '--')


        ax.grid()
        ax.legend()
        ax.set_xlim((4.9, 7.3))
        ax.set_ylim((-250, 50))
        ax.set_ylabel(r'Height change $\Delta h$ [a.u.]')


        ax2 = ax.twiny()
        ax2.set_xlim(((4.9 - popt_t_T[1]) / popt_t_T[0], (7.3 - popt_t_T[1]) / popt_t_T[0]))
        ax2.set_xlabel('Time [yr AD]')

        ax = fig.add_subplot(412)
        ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.plot(T_L_fit_1, fluct1, color = 'k', label = r'$\Delta h - \Delta h^{*}$ (obs.)')
        ax.legend()
        ax.grid()
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel('Fluctuations around $h^{*}$ [a.u.]')
        guesses = [1, 1, 49]
        tssamp = 4
        ax = fig.add_subplot(413)
        ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], std1[ww//2: -ww//2][::tssamp]**2, color = 'k', marker = 'o', alpha = .8, label = r'Variance obs.')
        ax.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], std_mean_sim1[ww//2: -ww//2][::tssamp]**2, color = 'r', marker = '+', alpha = .99, label = r'Variance sim.')

        ax.legend(loc = 2)
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel(r'Variance', color = 'k')
        ax2 = ax.twinx()
        ax2.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], dhdT1[ww//2: -ww//2][::tssamp], color = 'b', marker = 'x', alpha = .8, label = r'$|d\Delta h/dT|$ obs.')

        ax2.plot(T_L_fit_1[ww//2: -ww//2], dhdT_model_o1[ww//2: -ww//2], color = 'b', ls = '--', lw = 2, label = r'$|d\Delta h/dT|$ model')

        ax2.set_ylabel(r'Sensitivity $|d\Delta h/dT|$', color = 'b')
        ax2.legend(loc = 4)
        ax.set_xlim((4.9, 7.3))
        ax2.set_ylim((0, 60))



        ax = fig.add_subplot(414)
        ax.text(-.15, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], ac1[ww//2: -ww//2][::tssamp], color = 'k', marker = 'o', alpha = .8, label = r'AC1 obs.')
        ax.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], ac_mean_sim1[ww//2: -ww//2][::tssamp], color = 'r', marker = '+', alpha = .8, label = r'AC1 sim.')

        p0, p1 = np.polyfit(T_L_fit_1[ww//2: -ww//2], ac1[ww//2: -ww//2], 1)
        ax.plot(T_L_fit_1[ww//2: -ww//2], p0 * T_L_fit_1[ww//2: -ww//2] + p1, 'k--')


        p0, p1 = np.polyfit(T_L_fit_1[ww//2: -ww//2], ac_mean_sim1[ww//2: -ww//2], 1)
        ax.plot(T_L_fit_1[ww//2: -ww//2], p0 * T_L_fit_1[ww//2: -ww//2] + p1, 'r--')
        ax.legend(loc = 2)
        ax.set_xlim((4.9, 7.3))
        ax.set_ylabel(r'AC1', color = 'k')
        ax.set_xlabel(r'CWG JJA Temperature $T$ [${}^{\circ}$C]')
        ax2 = ax.twinx()
        ax2.scatter(T_L_fit_1[ww//2: -ww//2][::tssamp], np.exp(-1 / dhdT1[ww//2: -ww//2])[::tssamp], color = 'b', marker = 'x', alpha = .8, label = r'$e^{1/(d\Delta h/dT)}$ obs.')
        ax2.plot(T_L_fit_1[ww//2: -ww//2][::tssamp], np.exp(-1 / dhdT_model_o1[ww//2: -ww//2])[::tssamp], color = 'b', ls = '--', lw = 2, label = r'$e^{1/(d\Delta h/dT)}$ model')


        ax2.set_ylabel(r' $e^{1/(d\Delta h/dT)}$', color = 'b')
        ax2.legend(loc = 4)
        ax.set_xlim((4.9, 7.3))
        ax2.set_ylim((.75, 1.))

        fig.savefig('pics/sensitivity_std_ac1_ews_vs_Temp_mo%d_fs%d_fe%d_%s_order2_ww%d_test_lin.pdf'%(melt_offset * 100, fit_start * 10, fit_end * 10, dat, ww), bbox_inches = 'tight')



        T_range = np.arange(4.5, 8., .01)
        a, b = popt_t_T
        t_range = (T_range - b) // a

        h_range = np.arange(-280, 50, 0.01)

        pot_min = np.zeros((T_range.shape[0]), dtype = 'int')
        pot_max = np.zeros((T_range.shape[0]), dtype = 'int')
        ps = np.zeros((T_range.shape[0], h_range.shape[0]))
        for i in range(T_range.shape[0]):
            ps[i] = potential(h_range, T_range[i], popt8)
            if argrelmin(ps[i])[0].shape[0] == 1:
                pot_min[i] = argrelmin(ps[i])[0]
            if argrelmax(ps[i])[0].shape[0] == 1:
                pot_max[i] = argrelmax(ps[i])[0]

        fig = plt.figure(figsize = (8,5))
        ax = fig.add_subplot(111)
        cs = ax.pcolormesh(T_range, h_range, ps.T, vmin = -200, vmax = 200)
        cb = fig.colorbar(cs)
        cb.set_label(r'Potential $U_T(\Delta h)$')

        for i in range(10):
            sim = simulation[i]
            sim[sim < Melt_fit.min()] = np.nan
            ax.plot(LF(t_fit_future, *popt_t_T), sim, color = 'b', lw = .5, alpha = .3)

        ax.plot(dhdt8(h_range_stab, *popt8), h_range_stab, color = 'k', alpha = .4)
        ax.plot(dhdt8(h_range_unstab, *popt8), h_range_unstab, color = 'k', alpha = .4, ls = '--')
        ax.set_ylim((50, -250))


        ax.plot(T_L_fit, Melt_fit, color = 'r', lw = 2)


        ax.plot(LF(t_fit_model, *popt_t_T), Melt_fit_GW, color = 'r', lw = .5)
        ax.plot(LF(t_fit_model, *popt_t_T), Melt_fit_GC, color = 'r', lw = .5)
        ax.plot(LF(t_fit_model, *popt_t_T), Melt_fit_D5, color = 'r', lw = .5)

        ax.invert_yaxis()

        ax2 = ax.twiny()
        ax2.plot(t_fit, Melt_fit, color = 'r', lw = 2)
        ax2.set_xlabel(r'Time [yr]')
        ax.set_xlim((LF(1820, *popt_t_T), LF(2050, *popt_t_T)))
        ax2.set_xlim((1820, 2050))


        ax.set_ylabel(r'Ice sheet height change $\Delta h$ [a.u.]')
        ax.set_xlabel(r'JJA Surface Temperature $T$ [${}^{\circ}$C]')

        fig.savefig('pics/Potential2D_8thorder_vs_time_mo%d_fs%d_fe%d_test.png'%(melt_offset * 100, fit_start * 10, fit_end * 10), dpi = 300, bbox_inches = 'tight')





        a, b = popt_t_T





        h_range = h_range[::10]
        T_range = T_range[::10]
        t_range = (T_range - b) // a

        ps = np.zeros((T_range.shape[0], h_range.shape[0]))
        for i in range(T_range.shape[0]):
            ps[i] = potential(h_range, T_range[i], popt8)
            if argrelmin(ps[i])[0].shape[0] == 1:
                print(T_range[i])
                pot_min[i] = argrelmin(ps[i])[0]
            if argrelmax(ps[i])[0].shape[0] == 1:
                print(T_range[i])
                pot_max[i] = argrelmax(ps[i])[0]



        fig = plt.figure(figsize = (8, 6))
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(T_range, h_range)
        Z = ps.T
        Z[Z > 550] = np.nan
        Z[Z < -400] = np.nan
        surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap='viridis', alpha = .8)


        fp_stab = dhdt8(h_range_stab, *popt8)
        h_range_stab = h_range_stab[fp_stab > 4.5]
        fp_stab = fp_stab[fp_stab > 4.5]


        fp_unstab = dhdt8(h_range_unstab, *popt8)
        h_range_unstab = h_range_unstab[fp_unstab > 4.5]
        fp_unstab = fp_unstab[fp_unstab > 4.5]


        ax.plot(fp_stab, h_range_stab, potential(h_range_stab, fp_stab, popt8).T, color = 'k', alpha = .4, zorder = 10)
        ax.plot(fp_unstab, h_range_unstab, potential(h_range_unstab, fp_unstab, popt8).T, color = 'k', alpha = .4, ls = '--', zorder = 10)


        for i in range(10):
            sim = simulation[i]
            sim[sim < Melt_fit.min()] = np.nan
            ax.plot(T_L_fit_future, sim, potential(simulation[i], T_L_fit_future, popt8).T, color = 'b', lw = .5, alpha = .3, zorder = 10)


        ax.plot(T_L_fit, Melt_fit, potential(Melt_fit, T_L_fit, popt8).T, color = 'r', lw = 1.5, zorder = 10)
        ax.plot(T_L_fit_model, Melt_fit_GW, potential(Melt_fit_GW, T_L_fit_model, popt8).T, color = 'r', lw = .5, zorder = 10)
        ax.plot(T_L_fit_model, Melt_fit_GC, potential(Melt_fit_GC, T_L_fit_model, popt8).T, color = 'r', lw = .5, zorder = 10)
        ax.plot(T_L_fit_model, Melt_fit_D5, potential(Melt_fit_D5, T_L_fit_model, popt8).T, color = 'r', lw = .5, zorder = 10)


        ax.set_zlim3d(-400, 550)

        ax.view_init(30, -55)

        ax.set_ylabel(r'Ice sheet height change $\Delta h$ [a.u.]')
        ax.set_xlabel(r'JJA Surface Temperature $T$ [${}^{\circ}$C]')
        ax.set_zlabel(r'Potential $U_T(\Delta h)$ [a.u.]')
        ax.zaxis.set_ticks(np.arange(-400, 410, 200))
        surf.set_clim((-450, 450))
        ax.set_box_aspect((1.5, 1., 0.7))
        fig.savefig('pics/Potential3D_8thorder_vs_T_mo%d_fs%d_fe%d_test.png'%(melt_offset * 100, fit_start * 10, fit_end * 10), dpi = 500)



        idx1 = np.argmin(np.abs(T_range - 5))
        idx2 = np.argmin(np.abs(T_range - 6.92))
        idx3 = np.argmin(np.abs(T_range - 7.5))

        fig = plt.figure(figsize= (8,2))

        ax = fig.add_subplot(131)
        ax.scatter(h_range, ps[idx1], c = plt.cm.viridis((ps[idx1] + 450) / 900), edgecolor='none')
        ax.invert_xaxis()
        ax.set_ylabel(r'Potential $U_T(\Delta h)$ [a.u.]')
        ax.set_xlabel(r'$\Delta h$ [a.u.]')
        ax.set_title(r'$T < T_{c}$')
        ax = fig.add_subplot(132)
        ax.scatter(h_range, ps[idx2], c = plt.cm.viridis((ps[idx2] + 450) / 900), edgecolor='none')
        ax.invert_xaxis()
        ax.set_xlabel(r'$\Delta h$ [a.u.]')
        ax.set_title(r'$T = T_{c}$')
        ax = fig.add_subplot(133)
        ax.scatter(h_range, ps[idx3], c = plt.cm.viridis((ps[idx3] + 450) / 900), edgecolor='none')
        ax.invert_xaxis()
        ax.set_xlabel(r'$\Delta h$ [a.u.]')
        ax.set_title(r'$T > T_{c}$')
        plt.subplots_adjust(wspace = 0.25)
        fig.savefig('pics/Potentials1D_8thorder_vs_T_mo%d_fs%d_fe%d_test.pdf'%(melt_offset * 100, fit_start * 10, fit_end * 10))
