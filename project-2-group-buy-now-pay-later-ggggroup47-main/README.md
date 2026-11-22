# Buy Now Pay Later — Merchant Ranking System

A **fancy way to describe what traditional credit cards are**, but without the gravity of them.  
No credit scoring, no difficult PINs, and quick, easy approval.

This repository is made from the perspective of a **BNPL (Buy Now Pay Later) firm** selecting its **top candidate merchants** to onboard.

---

## 🧭 Overview

This project builds a **merchant ranking system** based on customer and revenue data.  
The goal is to identify merchants with the best performance and lowest risk for onboarding into a BNPL network.

---

## ⚙️ Instructions to Run
### 🧩 Preprocessing and External Data Downloads
**Notebooks:** `00`, `01s`  
**Script:** `external_data_processing.py`

These will **preprocess and download** the required datasets.  
Sources have been added in for simplicity.

---

### 🏪 Merchant-Level Preprocessing
**Notebooks:** `02`, `03`

These are for **merchant-level preprocessing**.  
All merchant-related features and the **customer aggregations** from previous work are finalised here before being passed on for ranking.

---

### 🏆 Ranking System
**Notebook:** `04`

This notebook builds the **ranking system**, which is a **simple linear model**.

**Ranking formula:**

**Ranking Score** = (Revenue × Take Rate) + (Weights × Selected Features)

The weights are unique to each of the sub-classes. They are determined by PCA in the 04 notebook.

--

## 📊 Additional Information

### Data Visualisations
The folder `data_visualisations/` contains helper functions used to create **geospatial maps** for merchant performance and customer activity.

Example visualisations include:
- Merchant density by region  
- Customer distribution heatmaps  
- Repeat-customer rate by merchant type  

These visualisations help identify clusters of high-value or high-risk merchants.

---

### Outlier Analysis
`notebook: 05` focuses on discovering **interesting merchant outliers**.  
Key findings include:
- Merchants with **high repeat customers but low revenue** — potential opportunities for upselling or marketing support.  
- Merchants with **high revenue but elevated fraud risk** — candidates for further review or stricter verification.

---

### Ranking System Notes
- The ranking system uses a **simple linear model** for transparency and interpretability.  
- Each merchant’s final score combines **financial performance**, **customer metrics**, and **risk indicators**.  
- Because merchant performance varies widely, scores are **standardised (z-scores)** rather than normalised, to prevent extreme values from skewing the rankings.

