Below is a structured set of literature suggestions and sample quotable fragments to support your 3‑step argument chain. I’ve focused on established, citable sources (standards, surveys, seminal papers). For each step you get: purpose, suggested sources, short quotes (paraphrasable), and BibTeX skeletons you can add to `references.bib`. Replace placeholders after you retrieve full metadata (page numbers, etc.). Where possible I include canonical DOIs.

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
- Intel forecast: “An autonomous vehicle may generate several terabytes of data per day” (supports magnitude; you already cited 4000 GB/day).

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

Core idea: Even non-autonomous vehicles adopt ADAS sensors (cameras, radar, LiDAR, ultrasonic, V2X modules). “Intelligence” (perception, decision, driver assistance) multiplies data sources and processing needs. Combined with legacy event-based strategies, this leads to fragmented, insufficient data for machine-learning driven maintenance and optimization—motivating improved compression and adaptive logging.

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

## Integrating into your argument chain

Suggested narrative bridging sentences (you can adapt):
1. “Early in-vehicle networking architectures (Classical CAN at 1 Mbit/s; later FlexRay) were sufficient for control loops but not for the sustained high-throughput streams produced by cameras, radar, and LiDAR, prompting selective data acquisition strategies.” (cite ISO11898_1_can, flexray_spec, bello2019)
2. “To prevent bus saturation, manufacturers adopted event-triggered and threshold-based diagnostic logging (Heemels_event_triggered_survey; tabuada_event_triggered_control), which—while reducing communication load—introduced maintenance overhead and diminished holistic observability.” (cite heemels_event_triggered_survey, tabuada_event_triggered_control, obd_limitations_survey)
3. “As ADAS and autonomous functionalities permeate even conventional vehicles, the proliferation of sensors and perception workloads magnifies the limitations of legacy event-driven data collection, necessitating scalable high-bandwidth networks and task-aware compression.” (cite adas_sensor_growth_survey, sensor_fusion_automotive, habibian2019video, barakat2025fisheye)

## Next steps

- Add definite sources (search for “Automotive Ethernet overview IEEE Communications Magazine” and “Vehicle sensor fusion survey”).
- Replace placeholder BibTeX entries with actual authors / DOIs.
- Cite multiple types (standard + survey + application) for credibility.
- Consider adding a short table aligning data sources (control sensors vs high-data sensors) with typical bitrates.

If you want, I can:
- Add the skeleton BibTeX entries directly to `references.bib`.
- Draft the revised paragraph with integrated citations.

Just tell me which action you’d like next.