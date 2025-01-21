import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from fire import Fire
from itertools import combinations
from typing import List

from cem.models.cem_regression import ConceptEmbeddingModel
from cem.models.cbm_regression import ConceptBottleneckModel

from models.cnn import CNN
from models.mlp import MLP
from models.cem import latent_cnn_code_generator_model, latent_mlp_code_generator_model
from data.ncmapss import NCMAPSSDataset
from data.cmapss import CMAPSSDataset


MODELS = ("cnn", "cnn_cls", "cnn_cem", "cnn_cbm", "mlp", "mlp_cls", "mlp_cem", "mlp_cbm")
DATASETS = ("CMAPSS", "N-CMAPSS")


def main(
    output_dir: str,
    data_path: str,
    dataset: str = "N-CMAPSS",
    train_n_ds: List[str] = ["02"],
    test_n_ds: List[str] = ["02"],
    train_units: List[int] = None,
    test_units: List[int] = None,
    eval: bool = False,
    model_type: str = "cnn",
    emb_size: int = 16,
    concepts: List[str] = ["LPT", "HPT"],
    combined_concepts: bool = False,
    binary_concepts: bool = True,
    max_epochs: int = 20,
    batch_size: int = 256,
    seed: int = 42,
    checkpoint: str = None,
    downsample: int = 10,
    exclusive_concepts: bool = False,
    extra_dims: int = 0,
    boolean_cbm: bool = False,
    RUL: str = "flat",
    window_size: int = 50,
    stride: int = 1,
    scaling: str = "legacy",
    **kwargs
    ):

    assert dataset in DATASETS, f"dataset must be one of: ${DATASETS}"
    n_concepts = len(concepts)
    if combined_concepts and dataset == "N-CMAPSS":
        n_concepts += len(list(combinations([c for c in concepts if c not in ["healthy", "Fc"]], 2)))
    if "Fc" in concepts:
        n_concepts += 2
    assert model_type in MODELS, f"model_type must be one of: ${MODELS}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed_everything(seed)

    if dataset == "N-CMAPSS":
        Dataset = NCMAPSSDataset
        input_dims = 18
    else:
        Dataset = CMAPSSDataset
        input_dims = 24

    if eval:
        test_ds = ConcatDataset([Dataset(data_path, n_DS=n_ds, units=test_units, mode="test", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling) for n_ds in test_n_ds])
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

        trainer = pl.Trainer(
            accelerator=device
        )

        if model_type == "cnn":
            model = CNN.load_from_checkpoint(checkpoint, cls_head=False, num_classes=n_concepts).eval()
        elif model_type == "cnn_cls":
            model = CNN.load_from_checkpoint(checkpoint, cls_head=True, num_classes=n_concepts).eval()
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
            ).eval()
        return trainer.test(model=model, dataloaders=test_dl)

    train_ds = ConcatDataset([Dataset(data_path, n_DS=n_ds, units=train_units, mode="train", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling) for n_ds in train_n_ds])
    val_ds = ConcatDataset([Dataset(data_path, n_DS=n_ds, units=test_units, mode="test", concepts=concepts, combined_concepts=combined_concepts, binary_concepts=binary_concepts, RUL=RUL, subsampling_rate=downsample, window_size=window_size, stride=stride, scaling=scaling) for n_ds in test_n_ds])

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    if model_type == "cnn":
        model = CNN(cls_head=False, cls_weight=0.1)
    elif model_type == "cnn_cls":
        model = CNN(cls_head=True, num_classes=n_concepts, cls_weight=0.1)
    elif model_type == "cnn_cbm":
        model = ConceptBottleneckModel(
            n_concepts=n_concepts, # Number of training-time concepts.
            extra_dims=extra_dims, # 2 + 30 = k*m (k=2, m=16)
            bool=boolean_cbm,
            n_tasks=1, # Number of output labels.
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            c_extractor_arch=latent_cnn_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels,
            exclusive_concepts=exclusive_concepts
        )
    elif model_type == "cnn_cem":
        model = ConceptEmbeddingModel(
            n_concepts=n_concepts, # Number of training-time concepts.
            n_tasks=1, # Number of output labels.
            emb_size=emb_size, #128 # We will use an embedding size of 128
            concept_loss_weight=0.1, # The weight assigned to the concept prediction loss relative to the task predictive loss.
            learning_rate=1e-3,  # The learning rate to use during training.
            optimizer="adam",  # The optimizer to use during training.
            training_intervention_prob=0.1, #0.25, # RandInt probability. We recommend setting this to 0.25.
            c_extractor_arch=latent_cnn_code_generator_model, # Here we provide our generating function for the latent code generator model.
            c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels,
            exclusive_concepts=exclusive_concepts
        )
    elif model_type == "mlp":
        model = MLP(cls_head=False, input_dims=input_dims*window_size, cls_weight=0.1)
    elif model_type == "mlp_cls":
        model = MLP(cls_head=True, input_dims=input_dims*window_size, num_classes=n_concepts, cls_weight=0.1)
    elif model_type == "mlp_cbm":
        model = ConceptBottleneckModel(
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
        )
    elif model_type == "mlp_cem":
        model = ConceptEmbeddingModel(
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
        )

    checkpointer = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints/",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpointer],
        # resume_from_checkpoint=checkpoint,
        log_every_n_steps=10,
        val_check_interval=10,
        limit_val_batches=1 if dataset == "N-CMAPSS" else None,
        # check_val_every_n_epoch=1,
    )

    return trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    Fire(main)
