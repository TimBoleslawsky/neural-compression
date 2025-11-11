## 🧭 1. Problem Formulation (Articulation)

**Motivation:**
Edge devices in industrial or vehicular systems produce vast amounts of time-series data (CAN, vibration, sensor fusion). Transmitting all raw data is infeasible; classical compression minimizes *bitrate*, not *utility*. Current “task-aware neural compression” approaches have focused mostly on images or audio, with limited exploration for time-series telemetry, where preserving temporal dependencies and diagnostic signals is crucial.

**Problem:**
*How can we learn discrete, compact token representations of telemetry data that maximize downstream ML task performance under a communication constraint?*

**Research Question:**

> Can task-aware tokenization improve the compression–utility trade-off for time-series analytics compared to standard neural or handcrafted compression methods?

---

## 🧩 2. Background and Hypothesis

* **Theoretical anchor:** Representation learning and information bottleneck theory (Tishby & Zaslavsky 2015) → trade-off between compression (rate) and relevance (utility).
* **Empirical anchor:** Neural tokenizers (e.g., VQ-VAE, DVAE) have demonstrated efficient discrete latent compression in other domains.
* **Gap:** Most neural compression for time-series (e.g., task-oriented codecs in *IEEE IoT J 2024*) uses continuous latent codes and does not exploit discrete “token” vocabularies that may improve interpretability, entropy coding, and adaptation to multiple downstream tasks.

**Hypothesis:**

> A neural tokenizer that discretizes sensor time-series into task-aware tokens can achieve comparable or better task performance at lower bitrate than standard autoencoder-based compression.

---

## ⚙️ 3. Proposed Approach (Contribution)

### Concept:

Design a **Task-Aware Neural Tokenizer (TANT)** consisting of:

1. **Encoder** — maps sensor windows to latent vectors
2. **Vector Quantizer (VQ)** — assigns each latent to a codebook entry (token)
3. **Entropy model** — models token probabilities for bitrate estimation
4. **Task head** — consumes tokens for classification/anomaly detection
5. **Joint optimization** — minimize total loss

   ```
   L_total = L_task + λ * L_rate + β * L_recon
   ```

   where:

   * `L_task`: task loss (e.g., cross-entropy or MSE)
   * `L_rate`: estimated token bitrate via entropy model
   * `L_recon`: optional reconstruction loss

### Novelty:

* Tokens are **discrete** and **task-optimized**, not just compressed features.
* Investigate **adaptation of token vocabulary size and quantization granularity** to balance rate and utility.
* Extend to **multi-task settings** (e.g., anomaly detection + forecasting).

---

## 🔬 4. Methodology

### Dataset:

* Public CAN-bus or vibration dataset (e.g., Bosch Production Line, NASA Bearing, or the Volvo internal equivalent if available).
* Possibly synthetic time-series with controlled noise and drift.

### Baselines:

1. **Classical compression:** PCA + quantization, delta encoding.
2. **Neural compression (continuous):** AE, VAE, or LSTM autoencoders.
3. **Rate–distortion optimized codec:** Learned compression without task signal.

### Evaluation Metrics:

* **Compression rate (bits per timestep / bitrate)**
* **Task accuracy / F1 / anomaly score**
* **Rate–utility Pareto frontier**
* **Token interpretability (optional)**

### Method:

1. Implement tokenizers using VQ-VAE or Gumbel-Softmax.
2. Jointly train with downstream task.
3. Sweep λ (rate weight) to trace Pareto curve.
4. Compare against baselines.
5. (Optional) Visualize token usage patterns for interpretability.

---

## 🧪 5. Expected Results

* Moderate compression gains (2–5×) at equal or slightly better task accuracy.
* Clear Pareto frontier between bitrate and task performance.
* Insights into how learned tokens cluster semantically meaningful patterns (e.g., vibration modes, fault signatures).

---

## 📚 6. Evaluation and Contribution Type

* **Type:** Empirical, design–evaluation study.
* **Contribution:** Demonstration that task-aware tokenization can outperform classical and continuous latent neural compression in time-series analytics.
* **Scientific significance:** Strengthens the bridge between **representation learning** and **task-optimized compression**, potentially extending to adaptive telemetry pipelines.

---

## 🧠 7. Possible Extensions (if time allows)

* Adaptive token vocabulary via reinforcement learning.
* Online retraining for domain drift.
* Plugging the tokenizer into a closed-loop system (bridging to the “end-to-end co-design” direction).
