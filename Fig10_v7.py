#  Fig_fit_summary.py
#  David Perkel 30 March 2024
import numpy as np

from common_params import *  # import common values across all models
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
import seaborn as sns
import csv
import subject_data
import scipy.stats as stats
import pandas as pd

def is_scenario(scen):  # test whether this is a scenario or subject
    # if this scenario is a subject, set use_forward_model to be false
    if (scen[0] == 'A' or scen[0] == 'S') and scen[1:3].isnumeric():
        return False  # it's a subject
    else:
        return True

def read_inv_summary(res, na_scen):
    # Reads a summary file, and tests whether average rpos error is less than chance based on shuffling
    # You need to run the inverse model for all subjects before making this figure

    # construct correct path for this resistivity
    # INV_OUT_PRFIX = 'INV_OUTPUT/'
    # R_TEXT = 'R' + str(round(res))
    # INVOUTPUTDIR = INV_OUT_PRFIX + R_TEXT + ACTR_TEXT + STD_TEXT + TARG_TEXT
    new_dir_suffix = 'RE%d' % res + '/R%d' % res + '_' + 'std_%.1f' % ACT_STDREL + '_thr_%d' % THRTARG + '/'
    INVOUTPUTDIR = INV_OUT_PRFIX + new_dir_suffix

    summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.npy'
    print(summary_file_name)
    [scenarios, thresh_summ_all, rpos_summary] = np.load(summary_file_name, allow_pickle=True)
    if na_scen > 0:
        scenarios = scenarios[na_scen:]  # Trim off the artificial scenarios, leaving just the subjects
    nscen = len(scenarios)
    n_elec = NELEC
    rpos_vals = np.zeros((nscen, n_elec))
    rpos_fit_vals = np.zeros((nscen, n_elec))
    thresh_err_summary = np.zeros((nscen, 2))
    rpos_err_summary = np.zeros(nscen)
    density_err_summary = np.zeros(nscen)
    dist_corr = np.zeros(nscen)
    dist_corr_p = np.zeros(nscen)

    for i, scen in enumerate(scenarios):
        rpos_fit_vals[i, 1:-1] = rpos_summary[i+3][0]
        rpos_vals[i, 1:-1] = rpos_summary[i+3][1]

    # get detailed data from the CSV summary file
    summary_csv_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.csv'
    with open(summary_csv_file_name, mode='r') as data_file:
        entire_file = csv.reader(data_file, delimiter=',', quotechar='"')
        for row, row_data in enumerate(entire_file):
            if row < 4:  # skip header row and three scenarios
                pass
            else:
                if row < 22:
                    [_, thresh_err_summary[row-na_scen - 1, 0], thresh_err_summary[row-na_scen -1, 1],
                     rpos_err_summary[row-na_scen - 1], aaa_temp, dist_corr[row-na_scen - 1],
                     dist_corr_p[row-na_scen - 1]] = row_data
                else:
                    pass
                # Note aaa_temp is a placeholder for the density error, which is not used

        data_file.close()

    if not tp_extend:
        return [np.asarray(thresh_summ_all[0]), thresh_err_summary, rpos_fit_vals[:, 1:-1], rpos_vals[:, 1:-1], rpos_err_summary,
                aaa_temp, dist_corr, dist_corr_p]
    else:
        return [np.asarray(thresh_summ_all[0]), thresh_err_summary, rpos_fit_vals, rpos_vals, rpos_err_summary, aaa_temp,
            dist_corr, dist_corr_p]

def fig10_summary():
    # Constants
    label_ypos = 1.05
    n_subj = 18
    nscen = len(scenarios)
    n_artscen = nscen-n_subj # number of artificial scenarios, which should not be plotted here
    n_elec = 16
    plot_indiv = True  # plot individual linear fits on panels E-G?
    mpl.rcParams['font.family'] = 'Arial'

    # Color values (from Matlab plotting for Fig. 8)
    fig10_colors = np.zeros((n_subj, 3))
    fig10_colors[0, :] = [0, 0, 135]  # S22
    fig10_colors[1, :] = [0, 0, 193]  # S27
    fig10_colors[2, :] = [0, 0, 255]  # S29
    fig10_colors[3, :] = [0, 3, 255]  # S38
    fig10_colors[4, :] = [5, 73, 255]  # S40
    fig10_colors[5, :] = [14, 131, 254]  # S41
    fig10_colors[6, :] = [24, 192, 255]  # S42
    fig10_colors[7, :] = [34, 255, 255]  # S43
    fig10_colors[8, :] = [53, 255, 193]  # S46
    fig10_colors[9, :] = [91, 255, 136]  # S27
    fig10_colors[10, :] = [140, 255, 83]  # S49
    fig10_colors[11, :] = [194, 255, 39]  # S50
    fig10_colors[12, :] = [255, 255, 9]  # S52
    fig10_colors[13, :] = [254, 195, 10]  # S53
    fig10_colors[14, :] = [253, 135, 6]  # S54
    fig10_colors[15, :] = [252, 79, 5]  # S55
    fig10_colors[16, :] = [252, 24, 6]  # S56
    fig10_colors[17, :] = [252, 0, 0]  # S57
    fig10_colors /= 255.0

    # Need data from 2 resistivities
    r_vals = [70.0, 250.0]

    # Layout figure
    fig1, axs1 = plt.subplots(4, 2, figsize=(8, 12), gridspec_kw={'height_ratios': [1, 1, 3, 3]})
    fig1.tight_layout(pad=3)

    plt.figtext(0.01, 0.97, 'A', color='black', size=20, weight='bold')
    plt.figtext(0.48, 0.97, 'B', color='black', size=20, weight='bold')
    plt.figtext(0.01, 0.72, 'C', color='black', size=20, weight='bold')
    plt.figtext(0.48, 0.72, 'D', color='black', size=20, weight='bold')
    plt.figtext(0.01, 0.50, 'E', color='black', size=20, weight='bold')
    plt.figtext(0.48, 0.50, 'F', color='black', size=20, weight='bold')
    plt.figtext(0.01, 0.24, 'G', color='black', size=20, weight='bold')
    plt.figtext(0.48, 0.24, 'H', color='black', size=20, weight='bold')
    plt.figtext(0.255, 0.95, 'R' + str(round(r_vals[0])), color='black', size=16)
    plt.figtext(0.755, 0.95, 'R' + str(round(r_vals[1])), color='black', size=16)

    (thr_sum_all_0, thresh_err_summary_0, rpos_fit_vals_0, rpos_vals_0, rpos_err_summary_0, aaa_temp, dist_corr_0,
     dist_corr_p_0) = read_inv_summary(r_vals[0], n_artscen)
    (thr_sum_all_1, thresh_err_summary_1, rpos_fit_vals_1, rpos_vals_1, rpos_err_summary_1, aaa_temp, dist_corr_1,
     dist_corr_p_1) = read_inv_summary(r_vals[1], n_artscen)

    thresh_err_summary = np.zeros((2, len(scenarios), 2))
    thresh_err_summary[0, 3:, :] = thresh_err_summary_0
    thresh_err_summary[1, 3:, :] = thresh_err_summary_1
    #print('rposvals: ', rpos_vals[0], rpos_vals[1])
    rposerrs = np.zeros((2, nscen, nscen, n_elec))

    # color = iter(cm.rainbow(np.linspace(0, 1, n_subj)))
    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel A
        x = np.asarray(thr_sum_all_0[idx + n_artscen, 0, 0, :])
        y = np.asarray(thr_sum_all_0[idx + n_artscen, 1, 0, :])
        axs1[0, 0].plot(x, y, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[0, 0].set_xlabel('Measured monopolar threshold (dB)')
        axs1[0, 0].set_ylabel('Fit monopolar threshold (dB)', labelpad=-1)
        axs1[0, 0].spines['top'].set_visible(False)
        axs1[0, 0].spines['right'].set_visible(False)
        axs1[0, 0].set_xlim([-25, 10])
        axs1[0, 0].set_ylim([-25, 10])

    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel C
        x = np.asarray(thr_sum_all_0[idx + n_artscen, 0, 1, :])
        y = np.asarray(thr_sum_all_0[idx + n_artscen, 1, 1, :])
        axs1[1, 0].plot(x, y, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[1, 0].set_xlabel('Measured tripolar threshold (dB)')
        axs1[1, 0].set_ylabel('Fit tripolar threshold (dB)', labelpad=4)
        axs1[1, 0].spines['top'].set_visible(False)
        axs1[1, 0].spines['right'].set_visible(False)

    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel B
        x = np.asarray(thr_sum_all_1[idx + n_artscen, 0, 0, :])
        y = np.asarray(thr_sum_all_1[idx + n_artscen, 1, 0, :])
        axs1[0, 1].plot(x, y, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[0, 1].set_xlabel('Measured monopolar threshold (dB)')
        axs1[0, 1].set_ylabel('Fit monopolar threshold (dB)', labelpad=-2)
        axs1[0, 1].spines['top'].set_visible(False)
        axs1[0, 1].spines['right'].set_visible(False)
        axs1[0, 1].set_xlim([-25, 10])
        axs1[0, 1].set_ylim([-25, 10])

    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel D
        x = np.asarray(thr_sum_all_1[idx + n_artscen, 0, 1, :])
        y = np.asarray(thr_sum_all_1[idx + n_artscen, 1, 1, :])
        axs1[1, 1].plot(x, y, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[1, 1].set_xlabel('Measured tripolar threshold (dB)')
        axs1[1, 1].set_ylabel('Fit tripolar threshold (dB)', labelpad=-2)
        axs1[1, 1].spines['top'].set_visible(False)
        axs1[1, 1].spines['right'].set_visible(False)

    # Now distance data
    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel E
        ct_dist = 1 - rpos_vals_0[idx]
        fit_dist = 1 - rpos_fit_vals_0[idx]
        retval = subject_data.subj_thr_data(scen)
        axs1[2, 0].plot(ct_dist, fit_dist, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[2, 0].set_xlabel('Measured electrode distance (mm)')
        axs1[2, 0].set_ylabel('Fit electrode distance (mm)')
        axs1[2, 0].spines['top'].set_visible(False)
        axs1[2, 0].spines['right'].set_visible(False)
        axs1[2, 0].set_ylim(0, 2.0)
        if plot_indiv:
            [slope, intercept] = np.polyfit(ct_dist, fit_dist, 1)
            # stats.linregress(fit_dist, ct_dist)
            # print('slope: ', slope, ' and intercept: ', intercept)
            minx = np.min(ct_dist)
            maxx = np.max(ct_dist)
            axs1[2, 0].plot((minx, maxx), (minx*slope + intercept, maxx*slope + intercept), '-', c=fig10_colors[idx, :],
                            linewidth=0.5)  # plot line

    # # Best fit line to the data panel E
    res = stats.linregress(1-rpos_vals_0.flatten(), 1-rpos_fit_vals_0.flatten())
    start_pt = res.intercept
    end_pt = res.intercept + (res.slope*2.0)
    axs1[2, 0].plot([0, 2], [start_pt, end_pt], color='black')
    print('slope: ', res.slope, ' intercept: ', res.intercept, ' r^2: ', res.rvalue**2, ' p: ', res.pvalue)
    axs1[2, 0].text(1.6, 0.08, '$r^2$ = {:.2f}'.format(res.rvalue**2))


    all_mean = 0.0
    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel F
        print('panel F: scen: ', scen)
        ct_dist = 1 - rpos_vals_1[idx]
        fit_dist = 1 - rpos_fit_vals_1[idx]
        retval = subject_data.subj_thr_data(scen)
        espace = retval[3]
        axs1[2, 1].plot(ct_dist, fit_dist, '.', color=fig10_colors[idx, :], markersize=5)
        err_mean = np.mean(np.abs(np.subtract(ct_dist, fit_dist)))
        all_mean += err_mean
        print('err mean: ', err_mean)
        axs1[2, 1].set_xlabel('Measured electrode distance (mm)')
        axs1[2, 1].set_ylabel('Fit electrode distance (mm)')
        axs1[2, 1].spines['top'].set_visible(False)
        axs1[2, 1].spines['right'].set_visible(False)
        axs1[2, 1].set_ylim(0, 2.0)
        if plot_indiv:
            [slope, intercept] = np.polyfit(ct_dist, fit_dist, 1)
            # print('slope: ', slope, ' and intercept: ', intercept)
            minx = np.min(ct_dist)
            maxx = np.max(ct_dist)
            axs1[2, 1].plot((minx, maxx), (minx*slope + intercept, maxx*slope + intercept), '-', c=fig10_colors[idx, :],
                            linewidth=0.5)  # plot line

    all_mean = all_mean / 18
    print('grand mean: ', all_mean)

    # # Now for the second resistivity for panel F
    # coeffs = np.polyfit(1-rpos_vals_1.flatten(), 1-rpos_fit_vals_1.flatten(), 1)
    res = stats.linregress(1-rpos_vals_1.flatten(), 1-rpos_fit_vals_1.flatten())
    start_pt = res.intercept
    end_pt = res.intercept + (res.slope*2.0)
    axs1[2, 1].plot([0, 2.0], [start_pt, end_pt], color='black')
    print('slope: ', res.slope, ' intercept: ', res.intercept, ' r^2: ', res.rvalue**2, ' p: ', res.pvalue)
    axs1[2, 1].text(1.6, 0.08, '$r^2$ = {:.2f}'.format(res.rvalue**2))

    # axs1[2, 1].tick_params(
    #     axis='x',           # changes apply to the x-axis
    #     which='both',       # both major and minor ticks are affected
    #     bottom=False,       # ticks along the bottom edge are off
    #     top=False,          # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    #
    # # statistics
    # # res = stats.linregress(x, y)

    # Now distance error data
    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel G
        ct_dist = 1 - rpos_vals_0[idx]
        fit_dist = 1 - rpos_fit_vals_0[idx]
        fit_err = fit_dist - ct_dist
        retval = subject_data.subj_thr_data(scen)
        espace = retval[3]
        axs1[3, 0].plot(fit_dist, fit_err, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[3, 0].set_xlabel('Fit electrode distance (mm)')
        axs1[3, 0].set_ylabel('Distance fit error (mm)', labelpad=6)
        axs1[3, 0].spines['top'].set_visible(False)
        axs1[3, 0].spines['right'].set_visible(False)
        axs1[3, 0].set_ylim(-1.5, 1.1)
        if plot_indiv:
            [slope, intercept] = np.polyfit(fit_dist, fit_err, 1)
            minx = np.min(fit_dist)
            maxx = np.max(fit_dist)
            axs1[3, 0].plot((minx, maxx), (minx*slope + intercept, maxx*slope + intercept), '-', c=fig10_colors[idx, :],
                            linewidth=0.5)  # plot line

    # # Best fit line to the data for panel G
    # coeffs = np.polyfit(1-rpos_fit_vals_0.flatten(), np.subtract(1-rpos_fit_vals_0.flatten(), 1-rpos_vals_0.flatten()), 1)
    res = stats.linregress(1-rpos_fit_vals_0.flatten(),
                                                np.subtract(1-rpos_fit_vals_0.flatten(), 1-rpos_vals_0.flatten()))
    start_pt = res.intercept
    end_pt = res.intercept + (res.slope*2.0)
    axs1[3, 0].plot([0, 2], [start_pt, end_pt], color='black')
    print('slope: ', res.slope, ' intercept: ', res.intercept, ' r^2: ', res.rvalue**2, ' p: ', res.pvalue)
    axs1[3, 0].text(1.6, -1.4, '$r^2$ = {:.2f}'.format(res.rvalue**2))

    for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel H
        ct_dist = 1 - rpos_vals_1[idx]
        fit_dist = 1 - rpos_fit_vals_1[idx]
        fit_err = fit_dist - ct_dist
        retval = subject_data.subj_thr_data(scen)
        espace = retval[3]
        axs1[3, 1].plot(fit_dist, fit_err, '.', color=fig10_colors[idx, :], markersize=5)
        axs1[3, 1].set_xlabel('Fit electrode distance (mm)')
        axs1[3, 1].set_ylabel('Distance fit error (mm)', labelpad=6)
        axs1[3, 1].spines['top'].set_visible(False)
        axs1[3, 1].spines['right'].set_visible(False)
        axs1[3, 1].set_ylim(-1.5, 1.1)
        if plot_indiv:
            [slope, intercept] = np.polyfit(fit_dist, fit_err, 1)
            minx = np.min(fit_dist)
            maxx = np.max(fit_dist)
            axs1[3, 1].plot((minx, maxx), (minx*slope + intercept, maxx*slope + intercept), '-', c=fig10_colors[idx, :],
                            linewidth=0.5)  # plot line

    # # Best fit line to the data for panel H
    # coeffs = np.polyfit(1-rpos_fit_vals_1.flatten(), np.subtract(1-rpos_fit_vals_1.flatten(), 1-rpos_vals_1.flatten()), 1)
    res = stats.linregress(1 - rpos_fit_vals_1.flatten(),
                                                np.subtract(1 - rpos_fit_vals_1.flatten(), 1 - rpos_vals_1.flatten()))

    start_pt = res.intercept
    end_pt = res.intercept + (res.slope*2.0)
    axs1[3, 1].plot([0, 2], [start_pt, end_pt], color='black')
    print('slope: ', res.slope, ' intercept: ', res.intercept, ' r^2: ', res.rvalue**2, ' p: ', res.pvalue)
    axs1[3, 1].text(1.6, -1.4, '$r^2$ = {:.2f}'.format(res.rvalue**2))

    # Save and display
    figname = 'Fig10_fit_summary.eps'
    plt.savefig(figname, format='eps', pad_inches=0.1)
    figname = 'Fig10_fit_summary.pdf'
    plt.savefig(figname, format='pdf', pad_inches=0.1)
    plt.show()

    plt.figure()
    dist_corr = np.zeros(nscen)
    dist_corr_p = np.zeros(nscen)

    signif = []
    near_signif = []
    not_signif = []

    for idx, corr in enumerate(dist_corr_0):
        if dist_corr_p_0[idx] > 0.1:
            not_signif.append(idx+3)
        elif dist_corr_p_0[idx] <= 0.1 and dist_corr_p_0[idx] > 0.05:
            near_signif.append(idx)
        else:
            signif.append(idx)


    # sns.swarmplot(dist_corr_0)
    # # sns.swarmplot(dist_corr_0[signif], color='black')
    # # sns.swarmplot(dist_corr_0[near_signif], color='gray')
    # # sns.swarmplot(dist_corr_0[not_signif], color='black', marker='$\circ$', s=6)
    # plt.show()
    #
    # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    # for idx, scen in enumerate(scenarios[n_artscen:]):  # Panel F
    #     print('panel F: scen: ', scen)
    #     ct_dist = 1 - rpos_vals_1[idx]
    #     fit_dist = 1 - rpos_fit_vals_1[idx]
    #     fit_err = np.subtract(fit_dist, ct_dist)
    #     axs.plot(fit_dist, fit_err, '.', color=fig10_colors[idx, :])
    #     axs.set_xlabel('Fit electrode distance (mm)')
    #     axs.set_ylabel('Fit errer (mm)')
    #     axs.spines['top'].set_visible(False)
    #     axs.spines['right'].set_visible(False)
    #     axs.set_ylim(-2.0, 2.0)
    #     # [slope, intercept] = np.polyfit(ct_dist, fit_dist, 1)
    #     # minx = np.min(ct_dist)
    #     # maxx = np.max(ct_dist)
    #     # axs1[2, 1].plot((minx, maxx), (minx*slope + intercept, maxx*slope + intercept), '-', c=fig10_colors[idx, :])  # plot line


    # # Best fit line to the data
    # coeffs = np.polyfit(1-rpos_fit_vals_1.flatten(), np.subtract(1-rpos_fit_vals_1.flatten(), 1-rpos_vals_1.flatten()), 1)
    # start_pt = coeffs[1]
    # end_pt = coeffs[1] + (coeffs[0]*2.0)
    # axs1.plot([0, 2], [start_pt, end_pt], color='black')
    # plt.show()



if __name__ == '__main__':
    fig10_summary()
