# Neural Compression Framework

A PyTorch-based neural compression framework implementing different architectures for time-series data compression and representation learning.

## Model Training

To train the model, use the general training script: 

```bash
source .venv/bin/activate

python -m framework.train_model --config use_cases/uni_edgecodec.yaml
```

---

## Model Evaluation

To evaluate any trained model (baseline, tokenization, etc.), use the general evaluation script:

```bash
python -m framework.evaluate_model \
    <config_path> \
    <checkpoint_path> \
```

**Options:**
- `--output <dir>`: Output directory for results
- `--num-vis-samples <N>`: Number of samples to visualize (default: 3)
