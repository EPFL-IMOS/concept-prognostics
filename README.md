<div align='center'>

# Interpretable Prognostics with Concept Bottleneck Models

[Florent Forest](https://florentfo.rest)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Olga Fink](https://people.epfl.ch/olga.fink)<sup>1</sup>
<br/>
<sub>
<sup>1</sup> Intelligent Maintenance and Operations Systems (IMOS), EPFL, Lausanne, Switzerland<br/>
</sub>

</div>

Source code for the implementation of the paper "Interpretable Prognostics with Concept Bottleneck Models".

## Getting started

Install the dependencies:

```shell
pip install -r requirements.txt
```

### Training the RUL prediction models

Train the RUL prediction models on the N-CMAPSS datasets using following command:

```shell
python cem/cmapss_train.py \
    --data-path /path/to/data/N-CMAPSS/ \
    --output-dir output/output-dir \
    --batch-size 256 \
    --model-type MODEL \
    --concepts "['Fan', 'HPC', 'HPT', 'LPT']" \  # Set the desired degradation concepts
    --train-ds "['01', '04', '05', '07']" \
    --train-units "[1, 2, 3, 4, 5, 6]" \
    --test-ds "['01', '04', '05', '07']" \
    --test-units "[7, 8, 9, 10]" \
    --downsample 10 \
    --RUL flat \  # 'flat': constant RUL before fault onset. 'linear': linear RUL from first cycle.
    [--emb-size 16 \]  # for CEM only
    [--boolean_cbm \]  # for Boolean CBM only
    [--extra-dims 60 \]  # for Hybrid CBM only
```

where `MODEL` can be one of the following model types:

Model | `MODEL` | Description | Interpretablility
------|---------|-------------|------------------
CNN | `cnn` | CNN regressor | :x: (black-box)
CNN+CLS | `cnn_cls` | CNN regressor + classification head | :x: (black-box)
CBM | `cnn_cbm` | Concept Bottleneck Model (boolean, fuzzy or hybrid depending on parameters) | :white_check_mark:
CEM | `cnn_cem` | Concept Embedding Model | :white_check_mark:

### Evaluation

Evaluate trained RUL prediction models using following command:

```shell
python cem/cmapss_eval.py \
    --checkpoint output/output-dir/checkpoints/last.ckpt \  # Path to trained weights
    --data-path /path/to/data/N-CMAPSS/ \
    --output-dir output/ \
    --batch-size 256 \
    --model-type MODEL \
    --concepts "['Fan', 'HPC', 'HPT', 'LPT']" \  # Set the desired degradation concepts
    --train-ds "['01', '04', '05', '07']" \
    --train-units "[1, 2, 3, 4, 5, 6]" \
    --test-ds "['01', '04', '05', '07']" \
    --test-units "[7, 8, 9, 10]" \
    --downsample 10 \
    --RUL flat \  # 'flat': constant RUL before fault onset. 'linear': linear RUL from first cycle.
    [--emb-size 16 \]  # for CEM only
    [--boolean_cbm \]  # for Boolean CBM only
    [--extra-dims 60 \]  # for Hybrid CBM only
    [--interventions \]  # enable test-time interventions
```

For evaluating the concept alignment score (CAS), use the script `cem/cmapss_eval_cas.py` in a similar way. Note that this computation is very slow.

For test-time interventions, we use the strategy for prognostics described in the paper by default. Different intervention strategies can be implemented in `cem/utils/eval_with_interventions.py`.

Evaluation will produce many figures and CSV files in the output directory; feel free to comment out some outputs in `cem/cmapss_eval.py` if these are not useful for you.

## Acknowledgements

Code for CEM and CBM borrowed from [cem](https://github.com/mateoespinosa/cem), thanks to the original authors!
