import os
from typing import List
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from fire import Fire
import numpy as np
import pandas as pd
from itertools import combinations

from cem.models.cem_regression import ConceptEmbeddingModel
from cem.models.cbm_regression import ConceptBottleneckModel
from cem.metrics.cas import concept_alignment_score

from models.cnn import CNN
from models.mlp import MLP
from models.cem import latent_cnn_code_generator_model, latent_mlp_code_generator_model
from data.ncmapss import NCMAPSSDataset
from data.cmapss import CMAPSSDataset
from data.ncmapss_features import NCMAPSSFeaturesDataset
from cmapss_train import MODELS, DATASETS

import warnings
warnings.simplefilter("ignore", UserWarning)


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
    **kwargs):

    assert dataset in DATASETS, f"dataset must be one of: ${DATASETS}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {scaling} scaling")

    trainer = pl.Trainer(
        accelerator=device,
        logger=False, # No logs to be dumped for this trainer
    )

    if dataset == "N-CMAPSS":
        Dataset = NCMAPSSDataset
    elif dataset == "N-CMAPSS-features":
        Dataset = NCMAPSSFeaturesDataset
    else:
        Dataset = CMAPSSDataset

    corr_values = []
    test_ds = ConcatDataset([Dataset(data_path, n_DS=n_ds, units=test_units, mode="test", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling) for n_ds in test_n_ds])
    if len(test_ds) == 0:
        test_ds = ConcatDataset([Dataset(data_path, n_DS=n_ds, units=test_units, mode="train", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling) for n_ds in test_n_ds])
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    y_test = np.concatenate([ds.df_Y.values.ravel() for ds in test_ds.datasets])
    c_test = np.concatenate([ds.concepts.values for ds in test_ds.datasets])

    concept_names = test_ds.datasets[0].concepts.columns

    batch_results = trainer.predict(model, test_dl)

    # Then we combine all results into numpy arrays by joining over the batch dimension
    if isinstance(model, CNN) or isinstance(model, MLP):
        if model.cls_head:
            c_embs = np.concatenate(
                list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
                axis=0,
            )
        else:
            c_embs = np.concatenate(
                list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
                axis=0,
            )
    else:  # CBM or CEM
        c_embs = np.concatenate(
            list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
            axis=0,
        )

    ##########
    ## Compute test concept alignment score
    ##########

    subsample = 100
    cas, _ = concept_alignment_score(
        c_vec=c_embs[::subsample],
        c_test=c_test[::subsample],
        y_test=y_test[::subsample],
        step=10,
        progress_bar=True,
    )
    print("CAS (per class): ", cas)
    mean_cas = np.mean(cas)
    print(f"Concept alignment score (CAS) is {mean_cas*100:.2f}%")

    corr_values.append((mean_cas, *cas))

    ##########
    ## Save results
    ##########
    corr = pd.DataFrame(corr_values, columns=['CAS', *concept_names])
    corr.to_csv(os.path.join(output_dir, 'cas.csv'), index=False)


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

    seed_everything(seed)

    if dataset == "N-CMAPSS":
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
