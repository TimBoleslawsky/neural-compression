# **Master Thesis Project Plan: Utility-Aware Adaptive Telemetry**

## **1. Research Objective**

Design, implement, and evaluate a **ML-guided adaptive logging framework** for modern vehicle systems, where telemetry is collected selectively based on **informational value** (uncertainty, novelty, or predicted utility for downstream tasks), aiming to:

1. Reduce bandwidth and storage usage compared to baseline event- or threshold-based logging.
2. Maintain or improve downstream ML task performance (predictive maintenance, anomaly detection, perception tasks).
3. Provide empirical evidence of the trade-off between logging efficiency and ML performance.

---

## **2. Research Questions (RQs)**

1. **RQ1:** Can ML-driven utility metrics guide selective telemetry logging without significantly degrading downstream ML performance?
2. **RQ2:** Which uncertainty/novelty metrics correlate best with downstream ML task utility?
3. **RQ3:** How does utility-aware adaptive logging compare to traditional event-triggered logging in terms of data volume reduction and ML accuracy?
4. **RQ4 (optional/future):** What is the empirical rate–utility curve for utility-aware logging under different compression or sampling rates?

---

## **3. Scope & Data**

* **Data sources:**

  * Public vehicle sensor datasets (e.g., CAN logs from UAH-DriveSet, VehiSense, or simulated datasets from CARLA or LGSVL for LiDAR/camera fusion).
  * Multi-modal: CAN bus signals, camera images, radar point clouds (if available).
* **Downstream ML tasks:**

  * **Predictive Maintenance (PdM):** classification/regression to predict faults or remaining useful life.
  * **Anomaly detection:** event detection in time-series (CAN signal deviations).
  * **Optional:** perception task (segmentation/object detection on compressed sensor streams).

---

## **4. Methodology**

### **A. Adaptive Telemetry Framework Design**

1. **Utility metric definition**

   * Candidate metrics:

     * **Uncertainty:** model-predicted probability entropy, Bayesian/Monte Carlo dropout, ensemble variance.
     * **Novelty:** distance in learned feature space (autoencoder reconstruction error or embedding distance).
     * **Information gain / task relevance:** expected improvement in downstream ML loss if data point is logged.
2. **Logging policy**

   * Data point is logged if: (U(x_t) > \theta)
   * Policy parameters: threshold (\theta), buffer size, optional adaptive thresholding (e.g., percentile-based).
   * Compare against baselines:

     * Fixed-rate logging
     * Event-triggered logging (threshold-based)
     * Full logging (upper bound)

---

### **B. ML Models for Utility Estimation**

* **Autoencoder-based anomaly detection:** use reconstruction error as novelty metric.
* **Uncertainty-aware models:** ensemble of LSTMs/GRUs for time-series; MC-dropout for Bayesian uncertainty estimation.
* **Feature embeddings:** for multi-modal sensor streams (camera/LiDAR/CAN).

---

### **C. Downstream ML Evaluation**

1. Train downstream ML models **only on the logged dataset**, evaluate on a full test set (to measure performance loss due to selective logging).
2. Metrics per task:

   * **PdM / classification:** Accuracy, F1, precision/recall, AUROC
   * **Regression / RUL:** RMSE, MAE, R²
   * **Anomaly detection:** precision@k, F1, recall
   * **Optional perception task:** mIoU (segmentation), mAP (detection)

---

### **D. Empirical Study & Analysis**

* **Experiment 1: Compare logging policies**

  * Measure: % data reduction, downstream ML task performance, correlation between utility metric and task relevance.
* **Experiment 2: Sensitivity analysis**

  * Vary utility threshold (\theta) to explore trade-offs (rate–utility).
* **Experiment 3: Ablation study**

  * Compare different utility metrics: uncertainty vs. novelty vs. hybrid.
* **Experiment 4 (optional/future):** co-design with learned compression

  * Apply neural compression to logged data and measure impact on downstream ML.

---

### **E. Tools & Frameworks**

* **Deep learning:** PyTorch or TensorFlow
* **Time-series / anomaly detection:** LSTM, GRU, Transformer encoders, autoencoders
* **Evaluation & logging simulation:** Python (NumPy, Pandas), possibly CARLA/LGSVL for synthetic data
* **Visualization:** Matplotlib, Seaborn (trade-offs, rate–utility curves)

---

## **5. Expected Contributions**

1. **Methodological contribution:** utility-aware adaptive telemetry framework for vehicle systems.
2. **Empirical contribution:** quantitative evaluation showing bandwidth savings with minimal impact on downstream ML performance.
3. **Analysis contribution:** comparison of different utility metrics and logging policies; sensitivity analysis of thresholds and logging rates.
4. **Potential future extension:** integrating rate–utility modeling or end-to-end co-design with ML models.

---

## **6. Validation / Empirical Soundness**

Following **ABC principles** (empirical, replicable, clearly documented):

* Use **public datasets or reproducible simulated data**.
* Implement logging policies and ML models **as independent modules**, with clear train/test separation.
* Provide **quantitative evaluation**: performance metrics, trade-off curves, statistical significance if possible.
* Include **visualization of key findings** (e.g., % bandwidth saved vs. downstream task performance).
* Document **hyperparameters, data preprocessing, and code** for reproducibility.

---

## **7. Timeline (rough)**

| Week  | Milestone                                                                                    |
| ----- | -------------------------------------------------------------------------------------------- |
| 1–2   | Literature review: adaptive logging, event-triggered telemetry, ML for PdM/anomaly detection |
| 3–4   | Dataset selection & preprocessing; baseline logging policies                                 |
| 5–6   | Implement uncertainty/novelty utility metrics; design logging policy                         |
| 7–8   | Implement downstream ML tasks for evaluation                                                 |
| 9–10  | Conduct experiments (policy comparison, sensitivity analysis)                                |
| 11    | Analyze results; prepare rate–utility curves, visualizations                                 |
| 12    | Write thesis draft (methods + experiments)                                                   |
| 13–14 | Refine, finalize, discuss limitations, future work                                           |

---
