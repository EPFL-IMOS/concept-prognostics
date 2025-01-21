import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, auc, roc_auc_score, roc_curve, accuracy_score
from fire import Fire
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List

from cem.models.cem_regression import ConceptEmbeddingModel
from cem.models.cbm_regression import ConceptBottleneckModel

from models.cnn import CNN
from models.mlp import MLP
from models.cem import latent_cnn_code_generator_model, latent_mlp_code_generator_model
from data.ncmapss import NCMAPSSDataset
from data.cmapss import CMAPSSDataset
from utils.rule import get_health_per_cycle, plot_theta
from utils.eval_with_interventions import eval_with_interventions
from cmapss_train import MODELS, DATASETS

import warnings
warnings.simplefilter("ignore", UserWarning)

def nasa_score(y_true, y_pred, scale=100):
    w_over = 1/10
    w_under = 1/13
    s = np.mean(np.exp(w_under * scale * np.maximum(y_true - y_pred, 0) + w_over * scale * np.maximum(y_pred - y_true, 0)), axis=0) - 1
    return s


def evaluation(
    model,
    output_dir: str,
    data_path: str,
    dataset: str = "N-CMAPSS",
    test_n_ds: List[str] = ["02"],
    batch_size: int = 256,
    test_units: List[int] = None,
    downsample: int = 10,
    concepts: List[str] = ["LPT", "HPT"],
    binary_concepts: bool = True,
    combined_concepts: bool = False,
    RUL: str = "flat",
    window_size: int = 50,
    stride: int = 1,
    scaling: str = "legacy",
    interventions: bool = False,
    **kwargs):

    dataset = kwargs["dataset"] if "dataset" in kwargs else "N-CMAPSS"
    assert dataset in DATASETS, f"dataset must be one of: ${DATASETS}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {scaling} scaling")

    trainer = pl.Trainer(
        accelerator=device,
        logger=False, # No logs to be dumped for this trainer
    )

    if dataset == "N-CMAPSS":
        Dataset = NCMAPSSDataset
    else:
        Dataset = CMAPSSDataset

    # save weights and bias of last linear layer as txt files
    # if isinstance(model, CNN) or isinstance(model, MLP):
    #     np.savetxt(os.path.join(output_dir, "weights.txt"),
    #         np.hstack((model.regression[0].weight.detach().cpu().numpy(), model.regression[0].bias.detach().cpu().numpy().reshape(-1, 1))))
    # else:
    #     np.savetxt(os.path.join(output_dir, "weights.txt"),
    #         np.hstack((model.c2y_model[0].weight.detach().cpu().numpy(), model.c2y_model[0].bias.detach().cpu().numpy().reshape(-1, 1))))

    corr_values = []
    mean_by_cycle_all = pd.DataFrame()
    for n_ds in test_n_ds:
        print("DS", n_ds)
        if test_units is None:  # for C-MAPSS
            test_units = range(1, 101)
        for unit in test_units:
            test_ds = Dataset(data_path, n_DS=n_ds, units=[unit], mode="test", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, include_healthy=True, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling)
            if len(test_ds) == 0:
                test_ds = Dataset(data_path, n_DS=n_ds, units=[unit], mode="train", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, include_healthy=True,subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling)
            if len(test_ds) == 0:
                continue
            print("unit", unit)
            test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=16, shuffle=False, drop_last=False)

            if dataset in ("N-CMAPSS", "N-CMAPSS-features"):
                # test_dl, cycles, hs, theta = load_torch_data(mode, batch_size, per_unit=unit)
                cycles = test_ds.df_A["cycle"].values
                hs = test_ds.df_A["hs"].values
                if concepts == "all":
                    concepts = [f"{c}-{m}" for c in ("Fan", "LPC", "HPC", "LPT", "HPT") for m in ("E", "F")]
                theta = pd.DataFrame({
                    "Fan": test_ds.df_T[['fan_eff_mod', 'fan_flow_mod']].min(axis=1),
                    "LPC": test_ds.df_T[['LPC_eff_mod', 'LPC_flow_mod']].min(axis=1),
                    "HPC": test_ds.df_T[['HPC_eff_mod', 'HPC_flow_mod']].min(axis=1),
                    "LPT": test_ds.df_T[['LPT_eff_mod', 'LPT_flow_mod']].min(axis=1),
                    "HPT": test_ds.df_T[['HPT_eff_mod', 'HPT_flow_mod']].min(axis=1),
                    "Fan-E": test_ds.df_T['fan_eff_mod'],
                    "Fan-F": test_ds.df_T['fan_flow_mod'],
                    "LPC-E": test_ds.df_T['LPC_eff_mod'],
                    "LPC-F": test_ds.df_T['LPC_flow_mod'],
                    "HPC-E": test_ds.df_T['HPC_eff_mod'],
                    "HPC-F": test_ds.df_T['HPC_flow_mod'],
                    "LPT-E": test_ds.df_T['LPT_eff_mod'],
                    "LPT-F": test_ds.df_T['LPT_flow_mod'],
                    "HPT-E": test_ds.df_T['HPT_eff_mod'],
                    "HPT-F": test_ds.df_T['HPT_flow_mod'],
                })[[c for c in concepts if c not in ["healthy", "Fc"]]]
            else:
                cycles = test_ds.df_X["cycle"].values
                hs = test_ds.df_X["hs"].values

            # x_test = test_ds.X.values
            y_test = test_ds.df_Y.values.ravel()
            c_test = test_ds.concepts.values

            n_classes = c_test.shape[1]
            concept_names = test_ds.concepts.columns

            if interventions:
                y_pred_inter, inter = eval_with_interventions(model, test_dl, cycles, n_concepts=n_classes, theta_df=theta)
                # ind_inter = np.where(inter == 1)[0]

            batch_results = trainer.predict(model, test_dl)
            # print(batch_results[0][0].shape, batch_results[0][1].shape, batch_results[0][2].shape)

            # Then we combine all results into numpy arrays by joining over the batch dimension
            if isinstance(model, CNN) or isinstance(model, MLP):
                if model.cls_head:
                    y_pred = np.concatenate(
                        list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)),
                        axis=0,
                    ).ravel()
                    c_pred = np.concatenate(
                        list(map(lambda x: x[1].sigmoid().detach().cpu().numpy(), batch_results)), # apply sigmoid as it returns logits
                        axis=0,
                    )
                else:
                    y_pred = np.concatenate(
                        list(map(lambda x: x.detach().cpu().numpy(), batch_results)),
                        axis=0,
                    ).ravel()
                    c_pred = None
            else:  # CBM or CEM
                y_pred = np.concatenate(
                    list(map(lambda x: x[2].detach().cpu().numpy(), batch_results)),
                    axis=0,
                ).ravel()

                c_pred = np.concatenate(
                    list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)),  # removed sigmoid, already applied in CBM and CEM
                    axis=0,
                )

            ##########
            ## Compute test task performance
            ##########
            task_accuracy = mean_squared_error(y_test, y_pred)
            print(f"Our model's test task RMSE is {np.sqrt(task_accuracy)*100:.3f}")
            task_nasa = nasa_score(y_test, y_pred)
            print(f"Our model's test task NASA score is {task_nasa:.3f}")
            if interventions:
                task_accuracy_inter = mean_squared_error(y_test, y_pred_inter)
                print(f"Our model's test task RMSE is {np.sqrt(task_accuracy_inter)*100:.3f} (intervened)")
                task_nasa_inter = nasa_score(y_test, y_pred_inter.numpy())
                print(f"Our model's test task NASA score is {task_nasa_inter:.3f} (intervened)")

            mean_by_cycle = pd.DataFrame({"cycle": cycles, "y_true": y_test, "y_pred": y_pred, "hs": hs})
            if interventions:
                mean_by_cycle["y_pred_inter"] = y_pred_inter
                for i,con in enumerate(concept_names):
                    mean_by_cycle[f"inter_{con}"] = inter[i]
            if c_pred is not None:
                for i,con in enumerate(concept_names):
                    mean_by_cycle[f"c_pred_{con}"] = c_pred[:,i]
            mean_by_cycle = mean_by_cycle.groupby("cycle", group_keys=False).mean().reset_index()
            rmse_by_cycle = mean_squared_error(mean_by_cycle["y_true"], mean_by_cycle["y_pred"])
            print(f"Our model's test task RMSE by cycle is {np.sqrt(rmse_by_cycle)*100:.3f}")
            nasa_score_by_cycle = nasa_score(mean_by_cycle["y_true"], mean_by_cycle["y_pred"])
            print(f"Our model's test task NASA score by cycle is {nasa_score_by_cycle:.3f}")
            if interventions:
                rmse_by_cycle_inter = mean_squared_error(mean_by_cycle["y_true"], mean_by_cycle["y_pred_inter"])
                print(f"Our model's test task RMSE by cycle is {np.sqrt(rmse_by_cycle_inter)*100:.3f} (intervened)")
                nasa_score_by_cycle_inter = nasa_score(mean_by_cycle["y_true"], mean_by_cycle["y_pred_inter"])
                print(f"Our model's test task NASA score by cycle is {nasa_score_by_cycle_inter:.3f} (intervened)")
                mean_by_cycle["inter"] = mean_by_cycle[[f"inter_{con}" for con in concept_names]].max(axis=1)

            mean_by_cycle["DS"] = n_ds
            mean_by_cycle["unit"] = unit
            mean_by_cycle_all = pd.concat([mean_by_cycle_all, mean_by_cycle])

            ##########
            ## Compute test task performance - non-healthy part only
            ##########
            task_accuracy_nhs = mean_squared_error(y_test[hs == 0], y_pred[hs == 0])
            print(f"Our model's test task RMSE is {np.sqrt(task_accuracy_nhs)*100:.3f} (non-healthy only)")
            task_nasa_nhs = nasa_score(y_test[hs == 0], y_pred[hs == 0])
            print(f"Our model's test task NASA score is {task_nasa_nhs:.3f} (non-healthy only)")
            if interventions:
                task_accuracy_inter_nhs = mean_squared_error(y_test[hs == 0], y_pred_inter[hs == 0])
                print(f"Our model's test task RMSE is {np.sqrt(task_accuracy_inter_nhs)*100:.3f} (intervened, non-healthy only)")
                task_nasa_inter_nhs = nasa_score(y_test[hs == 0], y_pred_inter[hs == 0].numpy())
                print(f"Our model's test task NASA score is {task_nasa_inter_nhs:.3f} (intervened, non-healthy only)")

            rmse_by_cycle_nhs = mean_squared_error(mean_by_cycle["y_true"][mean_by_cycle["hs"] == 0], mean_by_cycle["y_pred"][mean_by_cycle["hs"] == 0])
            print(f"Our model's test task RMSE by cycle is {np.sqrt(rmse_by_cycle_nhs)*100:.3f} (non-healthy only)")
            nasa_score_by_cycle_nhs = nasa_score(mean_by_cycle["y_true"][mean_by_cycle["hs"] == 0], mean_by_cycle["y_pred"][mean_by_cycle["hs"] == 0])
            print(f"Our model's test task NASA score by cycle is {nasa_score_by_cycle_nhs:.3f} (non-healthy only)")
            if interventions:
                rmse_by_cycle_inter_nhs = mean_squared_error(mean_by_cycle["y_true"][mean_by_cycle["hs"] == 0], mean_by_cycle["y_pred_inter"][mean_by_cycle["hs"] == 0])
                print(f"Our model's test task RMSE by cycle is {np.sqrt(rmse_by_cycle_inter_nhs)*100:.3f} (intervened, non-healthy only)")
                nasa_score_by_cycle_inter_nhs = nasa_score(mean_by_cycle["y_true"][mean_by_cycle["hs"] == 0], mean_by_cycle["y_pred_inter"][mean_by_cycle["hs"] == 0])
                print(f"Our model's test task NASA score by cycle is {nasa_score_by_cycle_inter_nhs:.3f} (intervened, non-healthy only)")

            ##########
            ## Plot (per cycle)
            ##########
            plt.clf()
            plt.subplots(figsize=(7, 5))
            scale = max(mean_by_cycle["y_pred"].max(), mean_by_cycle["y_true"].max())*100
            plt.plot(mean_by_cycle["cycle"], mean_by_cycle["y_true"]*100, color="green", label="True RUL", linewidth=1)
            plt.plot(mean_by_cycle["cycle"], mean_by_cycle["y_pred"]*100, "-", color="blue", label="Predicted RUL", linewidth=2)
            plt.ylim(0, scale*1.1)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('RUL', fontsize=14)
            plt.xlabel('Cycle', fontsize=14)
            plt.legend(loc="lower left", fontsize=14, title_fontsize=14);
            plt.savefig(os.path.join(output_dir, "pdf", f"{n_ds}_{unit}_cycle.pdf"), bbox_inches="tight")
            if interventions:
                for i,con in enumerate(concept_names):
                    if (mean_by_cycle[f"inter_{con}"] == 1).any():
                        inter_row = mean_by_cycle.iloc[(mean_by_cycle[f"inter_{con}"] == 1).idxmax()]
                        plt.plot([inter_row["cycle"]-1, inter_row["cycle"]-1], [0, inter_row[f"inter_{con}"] * scale], "r--")
                        plt.text(s=f"{con} intervention", x=inter_row["cycle"]-1-13, y=inter_row[f"inter_{con}"] * scale + 1, fontsize=14, color="red")
                inter_idx = (mean_by_cycle["inter"] == 1).idxmax()
                plt.plot(mean_by_cycle.iloc[inter_idx-1:]["cycle"], mean_by_cycle.iloc[inter_idx-1:]["y_pred_inter"]*100, "-", color="red", label='Predicted RUL (intervened)', linewidth=2)
                plt.legend(loc="lower left", fontsize=14, title_fontsize=14);
                plt.savefig(os.path.join(output_dir, f"{n_ds}_{unit}_cycle_inter.png"))
                plt.savefig(os.path.join(output_dir, "pdf", f"{n_ds}_{unit}_cycle_inter.pdf"), bbox_inches="tight")
            plt.close()

            ##########
            ## Scatter plot
            ##########
            plt.clf()
            plt.subplots(figsize=(7, 5))
            scale = max(mean_by_cycle["y_pred"].max(), mean_by_cycle["y_true"].max())*100
            plt.scatter(mean_by_cycle["y_true"]*100, mean_by_cycle["y_pred"]*100, color="blue")
            plt.plot([0, scale], [0, scale], "k--", linewidth=1)
            plt.ylim(-scale*0.1, scale*1.1)
            plt.ylim(-scale*0.1, scale*1.1)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('Predicted RUL', fontsize=14)
            plt.xlabel('True RUL', fontsize=14)
            plt.savefig(os.path.join(output_dir, f"{n_ds}_{unit}_cycle_scatter.png"))
            plt.savefig(os.path.join(output_dir, "pdf", f"{n_ds}_{unit}_cycle_scatter.pdf"), bbox_inches="tight")
            if interventions:
                inter_cycle = float("inf")
                inter_con = None
                for i,con in enumerate(concept_names):
                    if (mean_by_cycle[f"inter_{con}"] == 1).any():
                        inter_row = mean_by_cycle.iloc[(mean_by_cycle[f"inter_{con}"] == 1).idxmax()]
                        if inter_row["cycle"] < inter_cycle:
                            inter_cycle = inter_row["cycle"]
                            inter_con = con
                inter_idx = (mean_by_cycle["inter"] == 1).idxmax()
                if inter_con is not None:
                    plt.scatter(mean_by_cycle.iloc[inter_idx-1:]["y_true"]*100, mean_by_cycle.iloc[inter_idx-1:]["y_pred_inter"]*100, color="red", label=f'{inter_con} intervention')
                    plt.legend(loc="upper left", fontsize=14, title_fontsize=14);
                plt.savefig(os.path.join(output_dir, f"{n_ds}_{unit}_cycle_scatter_inter.png"))
                plt.savefig(os.path.join(output_dir, "pdf", f"{n_ds}_{unit}_cycle_scatter_inter.pdf"), bbox_inches="tight")
            plt.close()

            if c_pred is None:
                corr_values.append((n_ds, unit, np.sqrt(task_accuracy)*100, np.sqrt(rmse_by_cycle)*100, task_nasa, nasa_score_by_cycle, np.sqrt(task_accuracy_nhs)*100, np.sqrt(rmse_by_cycle_nhs)*100, task_nasa_nhs, nasa_score_by_cycle_nhs, None, *([None]*n_classes), None, None, None, None, None, None))
                continue

            ##########
            ## Plot fault information
            ##########
            all_pred_mean, all_pred_std, test_mean, hs_ = get_health_per_cycle(c_pred, c_test, cycles, hs)
            try:
                hs_cycle, fault_detection = plot_theta(all_pred_mean, all_pred_std, test_mean, hs_, all_pred_mean, concept_names, output_dir, f"{n_ds}_{unit}_failure")
            except Exception as e:
                print(e)
                print(hs_)

            try:
                concept_roc = roc_auc_score(c_test, c_pred)
                print(f"Our CEM's test concept ROC is {concept_roc*100:.3f}%")
            except:
                concept_roc = 0

            # store the fpr, tpr, and roc_auc for all averaging strategies
            fpr, tpr, roc_auc = dict(), dict(), dict()
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(c_test.ravel(), c_test.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            try:
                micro_roc_auc_ovr = roc_auc_score(
                    c_test,
                    c_pred,
                    multi_class="ovr",
                    average="micro",
                )
            except ValueError as e:
                print(e)
                micro_roc_auc_ovr = 0

            print(f"Micro-averaged One-vs-Rest ROC AUC score (concepts):\n{micro_roc_auc_ovr}")

            try:
                binary_roc_auc = roc_auc_score(
                    1 - hs,
                    c_pred.max(axis=1)
                )
            except ValueError as e:
                print(e)
                binary_roc_auc = 0

            print(f"Binary ROC AUC score:\n{binary_roc_auc}")

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(c_test[:, i], c_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr_grid = np.linspace(0.0, 1.0, 1000)

            # Interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(fpr_grid)

            for i in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

            # Average it and compute AUC
            mean_tpr /= n_classes

            # Average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr
            try:
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            except ValueError as e:
                print(e)
                roc_auc = 0

            print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

            ##########
            ## Compute test concept accuracy
            ##########

            c_pred_hard = c_pred > 0.5
            concept_acc = accuracy_score(c_test.ravel(), c_pred_hard.ravel())
            print(f"Concept accuracy is {concept_acc*100:.2f}%")

            class_acc = []
            for j in range(c_test.shape[-1]):
                test = c_test[:,j]
                pred = c_pred_hard[:,j]
                class_acc.append(accuracy_score(test, pred))
                print('Class '+str(j+1) + ' accuracy: ' + str(accuracy_score(test, pred)))

            ##########
            ## Plot fault detection
            ##########

            thresh = np.linspace(0, 1, num=1000)
            det = np.array([[(np.array(all_pred_mean)[:,c] > t).argmax() if np.any(np.array(all_pred_mean)[:,c] > t) else np.inf for t in thresh ] for c in range(len(concept_names))])

            delays = det.min(axis=0) - hs_cycle
            theta_cycle = np.array(test_mean).max(axis=1).argmax()

            oracle = theta_cycle - hs_cycle
            thresh = thresh[delays < np.inf]
            delays = delays[delays < np.inf]
            try:
                cmap_late = matplotlib.cm.Reds
                cmap_early = matplotlib.cm.Greens_r
                plt.clf()
                plt.subplots(figsize=(7,5))
                plt.plot(thresh, delays, "b", linewidth=2, label="Fault detection")
                y = np.arange(0, delays.max()+1, step=1)
                normalize = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
                for i in range(len(y)-1):
                    plt.fill_between(thresh, y[i], y[i+1], color=cmap_late(0.8*normalize(y[i])))
                y = np.arange(delays.min(), 1, step=1)
                normalize = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
                for i in range(len(y)-1):
                    plt.fill_between(thresh, y[i], y[i+1], color=cmap_early(1*normalize(y[i])))
                plt.text(x=(thresh[0]+thresh[-1])/2+0.1, y=delays.max() - 6, s="Late detection", fontsize=14, bbox=dict(facecolor='white', linewidth=0, alpha=0.75, boxstyle='round'))
                plt.text(x=(thresh[0]+thresh[-1])/2+0.1, y=delays.min() + 5, s="Early detection", fontsize=14, bbox=dict(facecolor='white', linewidth=0, alpha=0.75, boxstyle='round'))
                plt.plot([thresh[0], thresh[-1]], [oracle, oracle], "b-.", linewidth=2, label="$\Theta < -0.015$")
                plt.plot([thresh[0], thresh[-1]], [0, 0], "g-.", linewidth=2, label="Health state change")
                plt.xlim(thresh[0], thresh[-1])
                plt.ylim(delays.min(), delays.max())
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(loc="upper left", fontsize=12)
                plt.xlabel("Detection threshold", fontsize=14)
                plt.ylabel("$\Delta$ Cycles", fontsize=14)
                plt.savefig(os.path.join(output_dir, f'{n_ds}_{unit}_detection.png'))
                plt.savefig(os.path.join(output_dir, "pdf", f"{n_ds}_{unit}_detection.pdf"), bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(e)

            if interventions:
                corr_values.append((
                    n_ds, unit,
                    np.sqrt(task_accuracy)*100, np.sqrt(rmse_by_cycle)*100, np.sqrt(task_accuracy_inter)*100, np.sqrt(rmse_by_cycle_inter)*100,
                    task_nasa, nasa_score_by_cycle, task_nasa_inter, nasa_score_by_cycle_inter,
                    np.sqrt(task_accuracy_nhs)*100, np.sqrt(rmse_by_cycle_nhs)*100, np.sqrt(task_accuracy_inter_nhs)*100, np.sqrt(rmse_by_cycle_inter_nhs)*100,
                    task_nasa_nhs, nasa_score_by_cycle_nhs, task_nasa_inter_nhs, nasa_score_by_cycle_inter_nhs,
                    concept_acc, *class_acc, micro_roc_auc_ovr, binary_roc_auc, hs_cycle, theta_cycle, fault_detection, np.mean(np.abs(delays)) if len(delays) > 0 else 0))
            else:
                corr_values.append((
                    n_ds, unit,
                    np.sqrt(task_accuracy)*100, np.sqrt(rmse_by_cycle)*100,
                    task_nasa, nasa_score_by_cycle,
                    np.sqrt(task_accuracy_nhs)*100, np.sqrt(rmse_by_cycle_nhs)*100,
                    task_nasa_nhs, nasa_score_by_cycle_nhs,
                    concept_acc, *class_acc, micro_roc_auc_ovr, binary_roc_auc, hs_cycle, theta_cycle, fault_detection, np.mean(np.abs(delays)) if len(delays) > 0 else 0))

        ##########
        ## Save results
        ##########
        if interventions:
            corr = pd.DataFrame(corr_values, columns=['DS', 'unit', 'RUL_RMSE', 'RUL_RMSE_cycle', 'RUL_RMSE_inter', 'RUL_RMSE_cycle_inter', 'RUL_NASA', 'RUL_NASA_cycle', 'RUL_NASA_inter', 'RUL_NASA_cycle_inter', 'RUL_RMSE_nhs', 'RUL_RMSE_cycle_nhs', 'RUL_RMSE_inter_nhs', 'RUL_RMSE_cycle_inter_nhs', 'RUL_NASA_nhs', 'RUL_NASA_cycle_nhs', 'RUL_NASA_inter_nhs', 'RUL_NASA_cycle_inter_nhs', 'Concept_accuracy', *(f"acc_{i+1}" for i in range(n_classes)), 'AUC', 'AUC_HS', 'Fault_hs', 'Fault_theta', 'Fault_0.5', 'Delay_abs_integrated'])
        else:
            corr = pd.DataFrame(corr_values, columns=['DS', 'unit', 'RUL_RMSE', 'RUL_RMSE_cycle', 'RUL_NASA', 'RUL_NASA_cycle', 'RUL_RMSE_nhs', 'RUL_RMSE_cycle_nhs', 'RUL_NASA_nhs', 'RUL_NASA_cycle_nhs', 'Concept_accuracy', *(f"acc_{i+1}" for i in range(n_classes)), 'AUC', 'AUC_HS', 'Fault_hs', 'Fault_theta', 'Fault_0.5', 'Delay_abs_integrated'])
        corr.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        mean_by_cycle_all.to_csv(os.path.join(output_dir, 'mean_by_cycle.csv'), index=False)


def main(
    output_dir: str,
    dataset: str = "N-CMAPSS",
    model_type: str = "cnn",
    emb_size: int = 16,
    concepts: List[str] = ["LPT", "HPT"],
    combined_concepts: bool = False,
    seed: int = 42,
    checkpoint: str = None,
    exclusive_concepts: bool = False,
    extra_dims: int = 0,
    boolean_cbm: bool = False,
    window_size: int = 50,
    **kwargs):

    n_concepts = len(concepts)
    if combined_concepts and dataset == "N-CMAPSS":
        n_concepts += len(list(combinations([c for c in concepts if c not in ["healthy", "Fc"]], 2)))
    if "Fc" in concepts:
        n_concepts += 2
    assert model_type in MODELS, f"model_type must be one of: ${MODELS}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "pdf")):
        os.makedirs(os.path.join(output_dir, "pdf"))

    seed_everything(seed)

    if dataset == "N-CMAPSS":
        input_dims = 18
    elif dataset == "N-CMAPSS-features":
        input_dims = 18
    else:
        input_dims = 24

    if model_type == "cnn":
        model = CNN.load_from_checkpoint(checkpoint, cls_head=False).eval()
    elif model_type == "cnn_cls":
        model = CNN.load_from_checkpoint(checkpoint, cls_head=True, num_classes=n_concepts, cls_weight=0.1).eval()
    elif model_type == "cnn_cbm":
        model = ConceptBottleneckModel.load_from_checkpoint(checkpoint,
            n_concepts=n_concepts, # Number of training-time concepts. Dot has 2
            extra_dims=extra_dims, # 2 + 30 = k*m (k=2, m=16)
            bool=boolean_cbm,
            n_tasks=1, # Number of output labels. Dot is binary so it has 1.
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            c_extractor_arch=latent_cnn_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels
            exclusive_concepts=exclusive_concepts
        ).eval()
    elif model_type == "cnn_cem":
        model = ConceptEmbeddingModel.load_from_checkpoint(checkpoint,
            n_concepts=n_concepts, # Number of training-time concepts. Dot has 2
            n_tasks=1, # Number of output labels. Dot is binary so it has 1.
            emb_size=emb_size, # We will use an embedding size of 128
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            training_intervention_prob=0.1, #0.25, # RandInt probability. We recommend setting this to 0.25.
            c_extractor_arch=latent_cnn_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels
            exclusive_concepts=exclusive_concepts
        ).eval()
    elif model_type == "mlp":
        model = MLP.load_from_checkpoint(checkpoint, cls_head=False, input_dims=input_dims*window_size).eval()
    elif model_type == "mlp_cls":
        model = MLP.load_from_checkpoint(checkpoint, cls_head=True, input_dims=input_dims*window_size, num_classes=n_concepts, cls_weight=0.1).eval()
    elif model_type == "mlp_cbm":
        model = ConceptBottleneckModel.load_from_checkpoint(checkpoint,
            n_concepts=n_concepts, # Number of training-time concepts.
            extra_dims=extra_dims, # 2 + 30 = k*m (k=2, m=16)
            bool=boolean_cbm,
            n_tasks=1, # Number of output labels.
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            c_extractor_arch=latent_mlp_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels,
            exclusive_concepts=exclusive_concepts
        ).eval()
    elif model_type == "mlp_cem":
        model = ConceptEmbeddingModel.load_from_checkpoint(checkpoint,
            n_concepts=n_concepts, # Number of training-time concepts.
            n_tasks=1, # Number of output labels.
            emb_size=emb_size, #128 # We will use an embedding size of 128
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            training_intervention_prob=0.1, #0.25, # RandInt probability. We recommend setting this to 0.25.
            c_extractor_arch=latent_mlp_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels,
            exclusive_concepts=exclusive_concepts
        ).eval()

    return evaluation(
        model,
        output_dir=output_dir,
        dataset=dataset,
        concepts=concepts,
        combined_concepts=combined_concepts,
        **kwargs
    )


if __name__ == "__main__":
    Fire(main)
