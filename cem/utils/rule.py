import numpy as np
import matplotlib.pyplot as plt
import os


def get_health_per_cycle(c_pred, c_test, n_cycle, health_state):
    # get all values per cycle
    all_pred_mean = []
    all_pred_std = []
    hs = []
    all_test = []
    for cycle in np.unique(n_cycle):

        pred = c_pred[n_cycle==cycle]
        hs_now = health_state[n_cycle==cycle]
        mean_pred = np.mean(pred, axis=0)
        std_pred = np.std(pred, axis=0)

        test_mean = np.mean(c_test[n_cycle==cycle], axis=0)
        all_test.append(test_mean)
        all_pred_mean.append(mean_pred)
        all_pred_std.append(std_pred)
        hs.append(np.mean(hs_now))
    return all_pred_mean, all_pred_std, all_test, hs


def plot_theta(c_pred_mean, c_pred_std, c_test, hs, theta, class_names, output_dir, filename):
    plt.clf()
    # Create four polar axes and access them through the returned array
    c_pred_mean = np.asarray(c_pred_mean)
    c_pred_std = np.asarray(c_pred_std)
    c_test = np.asarray(c_test)
    hs = np.asarray(hs)

    fig, axs = plt.subplots(1, c_test.shape[-1], sharex=True, sharey=True, figsize=[7*c_test.shape[-1],5])
    # else:
    #     fig, axs = plt.subplots(2, c_test.shape[-1]//2, sharex=True, sharey=True, figsize=[20,10])
    #     axs = axs.flatten()
    hs_cycle = np.where(hs==0)[0][0]
    first_fault_detection = None
    if c_test.shape[-1] == 1:
        axs = [axs]

    for class_ in range(c_test.shape[-1]):
        ##########
        ## Plot all failure values
        ##########

        x = np.arange(0, len(c_pred_mean[:,class_]))
        mu = c_pred_mean[:,class_]
        sigma = c_pred_std[:,class_]

        axs[class_].plot(c_pred_mean[:,class_], color='blue', label='Pred concept', linewidth=2)
        axs[class_].fill_between(x, mu+sigma, mu-sigma, facecolor='blue', alpha=0.1)
        axs[class_].set_ylim(-0.1,1.1)

        ax2 = axs[class_] #.twinx()
        ax2.set_ylim(-0.1,1.1)
        ax2.plot(c_test[:,class_], color='green', label='True concept', linewidth=2)
        axs[class_].vlines(hs_cycle, -0.1, 1.05, color='green', linestyles='dashdot', label='Health state change')

        fault_detection = (c_pred_mean[:,class_] > 0.5).argmax()
        if fault_detection > 0:
            axs[class_].vlines(fault_detection, -0.1, 1.05, color='red', linestyles='dashdot', label='Fault detection')
            first_fault_detection = min(first_fault_detection, fault_detection) if first_fault_detection is not None else fault_detection

        ax2.set_title(class_names[class_], fontsize=14)
        ax2.legend(loc="upper left", fontsize=12)
        ax2.set_xlabel('Cycle', fontsize=14)
        ax2.set_ylabel('Concept activation', fontsize=14)
        ax2.xaxis.set_tick_params(labelsize=14, labelbottom=True)
        ax2.yaxis.set_tick_params(labelsize=14, labelleft=True)

    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.savefig(os.path.join(output_dir, "pdf", f"{filename}.pdf"), bbox_inches="tight")

    return hs_cycle, first_fault_detection
