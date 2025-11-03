## 1) Early concern: continuous streaming would overload in-vehicle networks

Core idea: Legacy in-vehicle networks (Classical CAN at 1 Mbit/s, LIN at 20 Kbit/s) were never designed for high-volume continuous sensor streams (video, radar point clouds). This motivated selective/event-based logging and diagnostic approaches.

Suggested sources:
- Bosch CAN Specification / ISO 11898-1 (standard bandwidth and frame limits).
- FlexRay consortium spec (higher deterministic bandwidth attempt).
- IEEE 802.1 TSN overview papers (motivation: need deterministic, higher throughput Ethernet).
- Survey: “Recent Advances and Trends in On-Board Embedded and Networked Automotive Systems” (Lo Bello et al., 2019) – already in your bib (`bello2019`).
- Paper: K. Matheus & T. Königseder, “Ethernet-Based Communication Systems in Vehicles,” (Springer book chapter / or early automotive Ethernet adoption texts).
- Survey: A. V. R. Shinde et al., or F. Gärtner et al. (Automotive Ethernet performance/diagnostics).
- Article: Mubeen et al., on scheduling and bandwidth for vehicular real-time networks (e.g., “A survey of automotive communication scheduling…” if applicable).
- Intel / OEM whitepapers forecasting multi‑TB/day data from autonomous vehicles (for magnitude).

Quotable fragments (paraphrased; confirm exact wording in sources):
- “Classical CAN offers a nominal 1 Mbit/s shared bandwidth, which becomes insufficient as high-data-rate sensors proliferate.”
- “FlexRay increased available bandwidth and determinism but still falls short for uncompressed camera or LiDAR streams.”
- “Emerging automotive Ethernet and TSN standards are introduced to cope with escalating sensor data volume and timing constraints.”
- From Lo Bello et al. (2019): Emphasize rise of Ethernet and time-sensitive networking due to bandwidth/latency constraints of legacy buses.
- Intel forecast: "As foreseen by Intel [4], from an average of 1.5 GB of traffic data per Internet user today, we will move toward 4000 GB of data generated per day by an AD car including technical data, personal data, crowd-sourced data, and societal data". (https://arpi.unipi.it/retrieve/5340c545-5478-40b3-ae53-38767fa5aeb7/Recent%2BAdvances%2Band%2BTrends%2B2018.pdf?utm_source=chatgpt.com)

BibTeX skeletons (samples):
```
@standard{iso11898_1_can,
  title={Road vehicles -- Controller area network (CAN) -- Part 1: Data link layer and physical signalling},
  organization={International Organization for Standardization},
  year={2015},
  number={ISO 11898-1:2015},
  note={Defines classical CAN bandwidth limitations}
}

@techreport{flexray_spec,
  title={FlexRay Communications System Protocol Specification Version 3.0.1},
  institution={FlexRay Consortium},
  year={2010},
  note={Deterministic high-speed automotive bus specification}
}

@article{automotive_ethernet_overview,
  author={AuthorLast, First and AuthorLast2, First2},
  title={Automotive Ethernet: An Overview of In-Vehicle Networking Evolution},
  journal={IEEE Communications Magazine},
  year={2016},
  volume={54},
  number={x},
  pages={xx--xx},
  doi={DOI_PLACEHOLDER}
}

@article{tsn_overview_automotive,
  author={S. S. (Example) and Others},
  title={Time-Sensitive Networking for Automotive Applications: A Survey},
  journal={IEEE Transactions on Industrial Informatics},
  year={YEAR},
  doi={DOI_PLACEHOLDER}
}
```

## 2) Adoption of event-based / data-on-demand diagnostics causing reduced observability and maintenance burden

Core idea: To avoid bus saturation, event-triggered logging and diagnostic frameworks (only recording anomalies or threshold crossings) were adopted, but they reduce holistic visibility and complicate maintenance (tuning thresholds, missed subtle degradation patterns).

Suggested sources:
- Event-triggered control/logging literature: W. P. M. H. Heemels et al., “An Introduction to Event-Triggered and Self-Triggered Control” (IEEE CDC / tutorial papers).
- Survey on automotive onboard diagnostics (OBD-II evolution, limitations of PIDs & DTC codes).
- Paper: Saponara et al. on condition monitoring / predictive maintenance in vehicles (if available).
- Works on event-driven data acquisition in embedded/real-time systems (search authors: Heemels, Tabuada).
- Paper: Tabuada, “Event-triggered real-time scheduling of control tasks” (illustrates reduced sampling vs information).
- Articles discussing OBD-II diagnostic trouble codes insufficient granularity for modern systems (industry whitepapers, SAE technical papers).

Quotable fragments (paraphrased):
- “Event-triggered paradigms reduce communication load by transmitting state changes instead of all samples.”
- “Threshold-based diagnostic trouble code activation can miss incipient faults due to coarse granularity.”
- “The manual configuration of trigger thresholds increases maintenance overhead as system complexity grows.”
- “While reducing bandwidth, event-based logging undermines continuous observability necessary for advanced predictive analytics.”

BibTeX skeletons:
```
@article{heemels_event_triggered_survey,
  author={Heemels, WPMH and Others},
  title={An Introduction to Event-Triggered and Self-Triggered Control},
  journal={Proceedings of the IEEE},
  year={2014},
  volume={102},
  number={11},
  pages={2266--2284},
  doi={10.1109/JPROC.2014.2354811}
}

@article{tabuada_event_triggered_control,
  author={Tabuada, Paulo},
  title={Event-Triggered Real-Time Scheduling of Control Tasks},
  journal={IEEE Transactions on Automatic Control},
  year={2007},
  volume={52},
  number={9},
  pages={1680--1685},
  doi={10.1109/TAC.2007.904452}
}

@article{obd_limitations_survey,
  author={Lastname, First and Others},
  title={Limitations of OBD-II Codes for Advanced Vehicle Health Monitoring},
  journal={SAE Technical Paper},
  year={YEAR},
  doi={DOI_PLACEHOLDER},
  note={Discusses granularity and maintenance of threshold-based diagnostics}
}

@article{event_triggered_monitoring_vehicles,
  author={Lastname, First and Others},
  title={Event-Based Condition Monitoring in Automotive Embedded Systems},
  journal={Journal/Conference},
  year={YEAR},
  pages={xx--xx},
  doi={DOI_PLACEHOLDER}
}
```

## 3) Universality across all vehicles + rising intelligence increases data quantity pressure

Core idea: Even non-autonomous vehicles adopt ADAS sensors (cameras, radar, LiDAR, ultrasonic, V2X modules). “Intelligence” (perception, decision, driver assistance) multiplies data sources and processing needs. Combined with legacy event-based strategies, this leads to fragmented, insufficient data for machine-learning driven maintenance and optimization — motivating improved compression and adaptive logging.

Suggested sources:
- ADAS sensor growth reports (Bosch or Continental whitepapers, McKinsey automotive electronics reports).
- Survey: “Automotive perception and sensor fusion” (look for IEEE or Elsevier surveys).
- Paper: “A Survey of Deep Learning for Automotive Applications” (if available).
- Barakat et al. (2025) fisheye compression paper (already added) for modality-specific growth.
- Habibian et al. (2019) for adaptive domain-specific compression (autonomous car videos).
- TSN / Automotive Ethernet adoption papers emphasizing scalable bandwidth for ADAS.

Quotable fragments (paraphrased):
- “Contemporary vehicles integrate multiple cameras, radar, LiDAR and ultrasonic sensors, even outside fully autonomous platforms.”
- “Growing sensor fusion workloads drive the migration from legacy buses to high-bandwidth deterministic Ethernet.”
- “Adaptive learned compression leverages domain-specific redundancy (e.g., autonomous driving scenes) to reduce bitrate while preserving task-relevant content.”

BibTeX skeletons:
```
@article{adas_sensor_growth_survey,
  author={Lastname, First and Others},
  title={Sensor Proliferation and Data Management Challenges in Modern ADAS},
  journal={IEEE Vehicular Technology Magazine},
  year={YEAR},
  pages={xx--xx},
  doi={DOI_PLACEHOLDER}
}

@article{sensor_fusion_automotive,
  author={Lastname, First and Others},
  title={Automotive Sensor Fusion: Trends, Challenges, and Opportunities},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={YEAR},
  doi={DOI_PLACEHOLDER}
}

@article{deep_learning_automotive_survey,
  author={Lastname, First and Others},
  title={Deep Learning Applications in Connected and Autonomous Vehicles: A Survey},
  journal={IEEE Access},
  year={YEAR},
  doi={DOI_PLACEHOLDER}
}
```

## 4) Relevance of Downstream Machine Learning Tasks in Modern Vehicle Systems

Modern vehicles increasingly rely on **data-driven intelligence** to enhance safety, reliability and efficiency. Beyond perception and control, downstream machine learning (ML) tasks—those leveraging collected vehicle and sensor data for offline analysis, optimization and predictive functions—have become central to automotive‐system design. These tasks include predictive maintenance, anomaly & intrusion detection, fleet‐level analytics, and sensor data‐fusion for perception and decision-making.

**Predictive maintenance (PdM).** ML is used to anticipate component failures or performance degradation before they manifest, thereby reducing unplanned downtime and maintenance costs. For example, Theissler et al. (2021) show that ML‐based PdM in automotive systems is a rapidly growing area, enabled by large volumes of vehicle sensor/telemetry data. ([Space Frontiers][1]) Many case‐studies report successful application of regression, classification and ensemble methods to vehicle maintenance. ([ijecs.in][2])

**Anomaly & intrusion detection on in-vehicle networks (IVNs).** As modern vehicles become more connected and complex, the internal communication buses (e.g., CAN) are targets for malfunction or cyber-attack. Recent surveys (Özdemir et al., 2024) review ML and deep-learning methods for anomaly detection on CAN traffic, emphasising the need for high‐frequency logging and rich data to train robust models. ([arXiv][3])

**Perception / sensor‐fusion ML for ADAS/autonomy.** The influx of continuous data streams from cameras, radar, LiDAR, ultrasonic, V2X modules means ML models must handle large, heterogeneous, temporal and event‐driven data. These downstream tasks require **holistic visibility** of the data, not just event-triggered snapshots.

**Fleet analytics and operational optimisation.** At the fleet scale, telematics and sensor data feed ML models for driver behaviour, energy/fuel consumption, maintenance scheduling, route optimisation and asset management. Reviews show this is a practical domain with substantial industry uptake. ([MDPI][4])

---

**Data‐collection tension and the need for adaptive logging.**
While the aforementioned ML tasks underline the *need* for rich, continuous, high-frequency and well‐labelled data, there is a clear tension: legacy in‐vehicle architectures (e.g., classic CAN at 1 Mbit/s, LIN at 20 Kbit/s) were never designed for high‐volume continuous sensor streams. This means that many production logging frameworks resort to event‐triggered or threshold-based logging, which reduces data volume but **limits visibility**, **biases sampling**, and complicates ML modelling (e.g., missing subtle degradation trends, fragmented datasets). The ML literature emphasises that adequate data volume, diversity and representation (including labelled failure cases) is often lacking, and this degrades model performance and generalisability. ([Space Frontiers][1])

Hence there is a clear and growing motivation for **adaptive logging, intelligent compression and selectively-triggered but ML-aware data acquisition** frameworks: such that the data collected still supports downstream ML tasks (predictive maintenance, anomaly detection, fleet analytics) while respecting bandwidth/storage constraints.

---

**In summary**, the relevance of downstream ML tasks in modern vehicle systems is strong and multi‐faceted: from maintenance to security to perception to fleet operations. Their success *critically depends* on data collection strategies that preserve ML utility, which links directly to your argument chain on legacy networks → event‐based logging → fragmented datasets → need for improved logging/compression.

[1]: https://spacefrontiers.org/r/10.1016/j.ress.2021.107864?utm_source=chatgpt.com "Predictive maintenance enabled by machine learning: Use cases and challenges in the automotive industry | Space Frontiers"
[2]: https://ijecs.in/index.php/ijecs/article/view/4786?utm_source=chatgpt.com "Machine Learning Algorithms for Predictive Maintenance in Autonomous Vehicles | International Journal of Engineering and Computer Science"
[3]: https://arxiv.org/abs/2409.07505?utm_source=chatgpt.com "A Survey of Anomaly Detection in In-Vehicle Networks"
[4]: https://www.mdpi.com/2076-3417/15/20/11095?utm_source=chatgpt.com "Vehicle Maintenance Demand Prediction: A Survey"

## 5) Balancing Compression (neural compression and event-triggered logging) and ML Performance in Vehicular Systems

There is an inherent trade-off between bandwidth/storage vs. ML utility! For example, event-triggered logging is efficient but introduces bias into logged data, possibly omitting predictive precursor signals. Moreover, even advanced learned or neural compression approaches exhibit similar trade-offs between compression efficiency and task-relevant information fidelity (Toderici et al., 2019).

### 1. Learned / Perception-aware compression and its effect on perception tasks

Recent work in distributed perception systems shows that compressing intermediate features or sensor data via neural autoencoders or hybrid schemes can reduce bandwidth dramatically while maintaining—or only moderately degrading—detection or segmentation performance. For example, **Xu et al., “CoSDH: Communication-Efficient Collaborative Perception” (CVPR 2025)** compress intermediate features and show ~95 % bandwidth reduction with only slight accuracy loss in object detection, via supply-demand awareness and hybrid fusion schemes. ([openaccess.thecvf.com][1])

Another notable study: **Löhdefink et al., “GAN- vs. JPEG2000 Image Compression for Distributed Automotive Perception”** show that although GAN-based compression may score worse in PSNR/SSIM, it can yield *better semantic segmentation (mIoU)* at very low bitrates compared to JPEG2000, when the segmentation model is trained on reconstructions from that codec. ([arXiv][2])

There is also general work on **recognition-aware learned compression** (e.g. Kawawa-Beaudan et al.) that explicitly optimize compression not only for visual distortion but also for classification accuracy (joint rate + task loss). ([arXiv][3])

**Takeaway:** compression tuned for reconstruction is not always optimal for ML tasks; joint optimization or perception-aware compression is more promising.

---

### 2. Event-triggered / threshold-based logging and missed subtle degradation

In diagnostic or maintenance logging, event-triggered approaches (i.e. log only when a signal crosses a threshold or fault code triggers) are common to reduce data volume. But they tend to **miss gradual, slow degradations or prior weak signals** that never breach the threshold. Some case studies (e.g. industrial white papers on event-oriented fault detection) describe how event logging simplifies diagnosis but at the cost of loss of “contextual continuity.” ([gregstanleyandassociates.com][4])

In the automotive domain, Theissler et al. (as you previously noted) argue that threshold tuning is laborious, and subtle drift patterns may remain invisible until failure. Unfortunately, there is limited open literature quantifying exactly how much ML model performance degrades due to this selective logging—but this is precisely a gap you can fill.

**Takeaway:** event-triggered logging is efficient but introduces bias into logged data, possibly omitting predictive precursor signals.

---

[1]: https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awareness_and_Intermediate-Late_Hybridization_CVPR_2025_paper.pdf?utm_source=chatgpt.com "CoSDH: Communication-Efficient Collaborative Perception via ..."
[2]: https://arxiv.org/abs/1902.04311?utm_source=chatgpt.com "GAN- vs. JPEG2000 Image Compression for Distributed Automotive Perception: Higher Peak SNR Does Not Mean Better Semantic Segmentation"
[3]: https://arxiv.org/abs/2202.00198?utm_source=chatgpt.com "Recognition-Aware Learned Image Compression"
[4]: https://gregstanleyandassociates.com/whitepapers/FaultDiagnosis/Event-Oriented/event-oriented.htm?utm_source=chatgpt.com "Event-oriented fault detection, diagnosis, and correlation"

## 6) Promising new Research Directions and Contributions

### 1. Uncertainty / novelty / utility-aware adaptive telemetry

An emerging direction is logging based not just on threshold triggers but on *informational value* — e.g. whether the current data point is uncertain, novel, or “interesting” relative to a model’s expectations. Some time-series anomaly detection research integrates uncertainty quantification (aleatoric + epistemic) to decide where to pay more attention or log more finely (e.g., Wiessner et al. in “Uncertainty-Aware Time Series Anomaly Detection”). ([MDPI][5])

While that work is more general, the principle applies: you could imagine an in-vehicle module that logs full sensor frames only when model uncertainty is high, otherwise logs compressed summaries. The challenge is designing the utility or novelty metric that correlates best with downstream ML performance.

**Takeaway:** adaptive logging guided by uncertainty or novelty is promising but requires careful design of the utility criterion.

**Possible contribution**: Proposing a utility-aware adaptive telemetry framework for modern vehicle systems, in which data collection is guided by machine learning models that estimate the informational value of sensor streams. By logging data points that are uncertain, novel, or particularly relevant to downstream tasks such as anomaly detection or predictive maintenance, the approach reduces bandwidth and storage requirements while preserving model performance. The contribution lies in demonstrating that ML-driven, task-specific logging can achieve superior efficiency and effectiveness compared to conventional threshold- or event-based strategies, with quantitative evaluation across representative automotive datasets.

---

### 2. Rate–utility (compression vs model performance) modelling

Some recent works look at *rate–utility trade-offs* more explicitly. For instance, in more general ML / data compression literature, **“Dataset Distillation as Data Compression: A Rate-Utility Perspective”** formulates the trade-off between dataset size (“rate”) and model performance (“utility”) in a joint optimization framework. ([arXiv][6])

Also, in the context of trajectory data, studies such as “Compression of Vehicle Trajectories with a Variational Autoencoder” explore compressing trajectory time-series inputs while retaining classification/clustering performance. ([MDPI][7])

However, the mapping from **bitrate (or log rate) → ML metric (accuracy, F1, recall, etc.)** remains highly non-linear, task-dependent, and underexplored in automotive settings. That is, you cannot reliably predict how much compression “budget” you can sacrifice before ML performance collapses, except empirically.

**Takeaway:** modelling the curve between compression rate and utility is possible in theory, but in practice is highly dependent on data modality, task, and architecture; a general closed-form solution is elusive.

**Possible contribution**: Quantifying the trade-off between compression rate and ML performance, which is useful, but less about designing or training ML models and more about evaluation and optimization frameworks. Probably not data science-y enough!

---

### 3. End-to-end co-design of logging + ML inference pipelines (closed-loop)

This direction focuses on treating the logging/compression and the ML‐model training/inference as a coupled system, rather than separate components. For example: Logging decisions influence ML model accuracy; ML model requirement should influence what is logged; the system adapts over time (models evolve, logging strategy evolves).
Why this matters: Many works treat compression/telemetry and ML as separate problems. But in vehicle systems, logging is upstream of the ML pipeline. Co‐optimization could yield better overall system performance (bandwidth + ML accuracy + latency). There is some work in “task‐oriented compression” in general communications/ML but much less specifically for vehicle systems. E.g., “Task-Oriented Data Compression for Multi‐Agent Communications Over Bit-Budgeted Channels” explores similar themes. 
arXiv

**Potential contribution**: A thesis could define and evaluate a looped architecture: logging/compression module ↔ ML module ↔ model performance feedback ↔ adapt logging policy. This could include formalising a cost function: bandwidth cost + ML task error + latency penalty; then designing algorithms to adapt logging policy in real-time.

---

### 4. Task-aware tokenization as a compression strategy

Tokenization can be interpreted as a form of data compression and representation, where continuous or high-dimensional sensor streams (e.g., LiDAR, radar, CAN, camera frames) are converted into discrete units (“tokens”) that preserve task-relevant information for downstream ML. Unlike traditional compression methods that optimize for reconstruction quality (PSNR/SSIM), task-aware tokenization aims to maximize ML utility per bit, maintaining critical patterns while reducing sequence length and storage/transmission cost. Recent work (e.g., 2402.16412) formalizes tokenization as a learned/adaptive process, showing that token boundaries and embeddings can be optimized for downstream performance.

Takeaway: Tokenization provides a principled way to represent sensor data efficiently for ML tasks, complementing selective telemetry or neural compression. Its novelty lies in discretizing and structuring logged data for task-specific ML, which is particularly underexplored for heterogeneous automotive sensor streams.

Possible contribution: Designing and evaluating a task-aware tokenization scheme for logged vehicle sensor data, where token formation is guided by ML model requirements (e.g., predictive maintenance or anomaly detection). The study could compare conventional compression, naive tokenization, and ML-optimized tokenization in terms of model accuracy, bandwidth, and storage efficiency, highlighting trade-offs and practical feasibility for in-vehicle deployment.

---

[5]: https://www.mdpi.com/1999-5903/16/11/403?utm_source=chatgpt.com "Uncertainty-Aware Time Series Anomaly Detection"
[6]: https://arxiv.org/html/2507.17221v1?utm_source=chatgpt.com "Dataset Distillation as Data Compression: A Rate-Utility ..."
[7]: https://www.mdpi.com/2076-3417/10/19/6739?utm_source=chatgpt.com "Compression of Vehicle Trajectories with a Variational ..."



 
