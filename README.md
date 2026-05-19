# Intrusion Detection and Prevention System (IDS/IPS)

A multi-phase cybersecurity project focused on anomaly-based intrusion detection and prevention using Machine Learning and Deep Learning techniques.

This project was developed as a graduation project and evolved from a small-scale ML prototype into a more advanced and realistic ML-DL fusion architecture trained on custom-generated network traffic.

---

# Project Evolution

## Phase 1 — Basic ML IDS Prototype

📁 `basic-ml-ids-system`

The first phase focused on understanding the fundamentals of anomaly detection and supervised Machine Learning in cybersecurity environments.

At this stage, the project was designed as a relatively small-scale prototype and tested under three different scenarios:

* **Ready Dataset (KDD Cup)** — benchmark testing environment
* **Custom Mini Dataset** — small-scale custom traffic experiments
* **Original Traffic Dataset** — manually collected network traffic samples

Core studies in this phase included:

* Data preprocessing
* Feature extraction
* Isolation Forest anomaly filtering
* Supervised ML model training
* Threshold tuning
* Basic IDS architecture development

This phase served as the foundation of the overall system.

---

## Phase 2 — Advanced ML-DL Fusion IDS System

📁 `advanced-ml-dl-fusion-ids-system`

In the second phase, the project architecture was significantly redesigned and improved to create a more realistic and scalable cybersecurity solution.

Unlike the first prototype-oriented stage, this version focuses on real-world traffic behavior, advanced feature engineering, and hybrid AI-based intrusion detection.

Major improvements in this phase:

* Realistic Wireshark-based traffic collection
* Large-scale custom dataset generation
* Smart traffic labeling strategy
* Advanced feature engineering
* Deep Learning integration
* ML + DL weighted fusion architecture
* Risk-based decision mechanism
* Improved attack detection performance
* Modular and scalable project structure

The final architecture combines Machine Learning and Deep Learning outputs using a weighted fusion strategy in order to improve detection quality while maintaining strong overall system accuracy.

---

# Technologies Used

* Python
* Scikit-learn
* TensorFlow / Keras
* Pandas
* NumPy
* CatBoost
* Wireshark

---

# Final Objective

The primary goal of this project is to build an intelligent and scalable IDS/IPS architecture capable of detecting anomalous network behavior using hybrid AI-based approaches and realistic traffic analysis.

---

# Technical Blog

- [What Building an IDS/IPS System Taught Me About Real-World AI Systems](https://medium.com/@edanurfirat/what-building-an-ids-ips-system-taught-me-about-real-world-ai-systems-92c6f9b79ba6)

This article explains the engineering decisions, dataset challenges, feature engineering process, smart labeling strategy, and the lessons learned while developing the IDS/IPS system over time.
