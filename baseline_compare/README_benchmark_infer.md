# baseline_compare benchmark_infer scripts

This folder contains benchmark.py-style inference entrypoints for the three TFM baselines:

- `TabPFN-main/benchmark_infer.py`
- `LimiX/benchmark_infer.py`
- `tabular-dl-tabr/benchmark_infer.py`

All three scripts are self-contained classification inference entrypoints. They read the repo-root `data178/<dataset>` layout by default through `--data-root ../../data178`, process only `binclass` and `multiclass`, skip other task types, and write:

- `worker_*.csv`
- `all_classification_results.csv`
- `summary.txt`

## Local commands

Run each script from its own model directory:

```bash
cd baseline_compare/TabPFN-main
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose

cd ../LimiX
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose

cd ../tabular-dl-tabr
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose
```

The default output roots are:

```text
baseline_compare/TabPFN-main/results
baseline_compare/LimiX/results
baseline_compare/tabular-dl-tabr/results
```

## Environment setup

Unified `TFM` environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n TFM --clone tabpfn_benchmark -y
conda activate TFM
pip install --no-deps -e baseline_compare/TabPFN-main
pip install --no-deps delu==0.0.15 rtdl==0.0.13
pip install -r baseline_compare/requirements_pipeline.txt
```

For a fresh server without `tabpfn_benchmark`, create it from scratch:

```bash
conda env create -f baseline_compare/environment_TFM.yml
conda activate TFM
pip install -r baseline_compare/TabPFN-main/requirements_benchmark.txt
pip install -r baseline_compare/LimiX/requirements_benchmark.txt
pip install --no-deps -e baseline_compare/TabPFN-main
pip install --no-deps delu==0.0.15 rtdl==0.0.13
pip install -r baseline_compare/requirements_pipeline.txt
```

This unified environment keeps `torch==2.7.1` for TabPFN/LimiX. TabPFN is installed from the local checkout in editable mode so `import tabpfn` works outside `benchmark_infer.py` as well. TaBR's legacy `delu/rtdl` metadata requests `torch<2,numpy<2`, so they are installed with `--no-deps`; the local TaBR checkout includes a PyTorch 2.6+ checkpoint-loading compatibility patch.

TabPFN:

```bash
conda create -n tabpfn_benchmark python=3.11 -y
conda activate tabpfn_benchmark
pip install -r baseline_compare/TabPFN-main/requirements_benchmark.txt
cd baseline_compare/TabPFN-main
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose
```

The TabPFN benchmark requirements pin `torch==2.7.1` for RTX 2080 Ti compatibility on 238.

LimiX:

```bash
conda create -n limix_benchmark python=3.12 -y
conda activate limix_benchmark
pip install -r baseline_compare/LimiX/requirements_benchmark.txt
cd baseline_compare/LimiX
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose
```

`benchmark_infer.py` defaults to `config/cls_default_noretrieval.json` because the LimiX README says retrieval inference is intended for GPUs above RTX 4090. To use retrieval explicitly:

```bash
cd baseline_compare/LimiX
python benchmark_infer.py \
  --config-path config/cls_default_16M_retrieval.json \
  --max-datasets 1 --workers 1 --gpus auto --verbose
```

`flash-attn` is left as an optional line in the requirements file because the default no-retrieval smoke path does not require it, and building it on older GPUs can be fragile.

TaBR:

```bash
conda create -n tabr_benchmark python=3.10 -y
conda activate tabr_benchmark
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.13.1 pytorch-cuda=11.7 faiss-gpu=1.7.2 -y
pip install -r baseline_compare/tabular-dl-tabr/requirements_benchmark.txt
cd baseline_compare/tabular-dl-tabr
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus auto --verbose
```

TaBR writes per-dataset native temporary files under:

```text
baseline_compare/tabular-dl-tabr/results/_tabr_work/<dataset_name>/
```

## 238 test workflow

Current discovered 238 repo path:

```text
/data0/zhuhao2025/tabiclv2_test
```

Sync the new baseline files:

```bash
rsync -av baseline_compare/ 238:/data0/zhuhao2025/tabiclv2_test/baseline_compare/
```

Then run smoke tests on GPU 2:

```bash
ssh 238
cd /data0/zhuhao2025/tabiclv2_test

source ~/miniconda3/etc/profile.d/conda.sh
conda activate TFM

cd baseline_compare/TabPFN-main
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus 2 --verbose

cd ../LimiX
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus 2 --verbose

cd ../tabular-dl-tabr
python benchmark_infer.py --max-datasets 1 --workers 1 --gpus 2 --verbose
```

After one-dataset smoke tests pass, expand to:

```bash
cd /data0/zhuhao2025/tabiclv2_test/baseline_compare/TabPFN-main
python benchmark_infer.py --max-datasets 3 --workers 1 --gpus 2 --verbose

cd ../LimiX
python benchmark_infer.py --max-datasets 3 --workers 1 --gpus 2 --verbose

cd ../tabular-dl-tabr
python benchmark_infer.py --max-datasets 3 --workers 1 --gpus 2 --verbose
```

## 238 validation record

- Local and 238 `py_compile` passed for all three self-contained `benchmark_infer.py` files.
- TabPFN on 238 under `pipeline`/`TFM`, from `baseline_compare/TabPFN-main`, GPU 2, `--max-datasets 1`: script produced `results/all_classification_results.csv` and `results/summary.txt`, but the row failed at official weight download with `TabPFNLicenseError` because `TABPFN_TOKEN` is not configured. Set `TABPFN_TOKEN` or pass `--model-path` to a downloaded checkpoint, then rerun.
- LimiX on 238 under `pipeline`/`TFM`, from `baseline_compare/LimiX`, GPU 2, `--max-datasets 1`: script produced `results/all_classification_results.csv` and `results/summary.txt`; the first dataset has 11 classes and was rejected by the official LimiX 2..10 class limit. Pass `--model-path` to a local `LimiX-16M.ckpt` to avoid network download on supported datasets.
- TaBR on 238: `tabr_benchmark` environment creation succeeded, but conda failed while downloading `libcublas-11.10.3.66` from the `nvidia` channel (`CondaHTTPError: HTTP 000 CONNECTION FAILED`). Running the script in the available test environment still produced result files, with the expected backend dependency error `ModuleNotFoundError: No module named 'delu'`.
- Unified `TFM` environment: cloned from `tabpfn_benchmark` or created from `environment_TFM.yml`, installed local TabPFN in editable mode, added TaBR runtime dependencies via pip, and verified imports for `torch/tabpfn/tabpfn_common_utils/kditransform/delu/rtdl/faiss/tomli/tomli_w/loguru/tensorboard`. `pip check` still reports legacy metadata conflicts for `delu/rtdl` with `torch==2.7.1` and `numpy==2.4.4`; this is intentional for the unified environment.
- TaBR under `pipeline`/`TFM`, from `baseline_compare/tabular-dl-tabr`, GPU 2, `--max-datasets 1`: ran successfully on `ASP-POTASSCO-classification`, `accuracy=0.277992`, and wrote `results/all_classification_results.csv` plus `results/summary.txt`.
