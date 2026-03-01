---

# <h1><center>😊 Welcome!</center></h1>

## 📌 Project Description

This project analyzes the **direct marketing campaigns of a Portuguese banking institution**. The campaigns were primarily conducted through phone calls, where clients were contacted multiple times to assess whether they would subscribe to a **term deposit** (a type of fixed-term savings account).

The main challenge lies in predicting the likelihood of a client subscribing to a term deposit based on their personal, financial, and socio-economic characteristics, as well as details of the marketing campaign.

---

## 🎯 Project Objective

The objective of this project is to:

1. **Build a classification model** to predict whether a client will subscribe to a term deposit (`yes`/`no`).
2. **Analyze the key factors** influencing client decisions, such as demographic data, previous campaign outcomes, and macroeconomic indicators.
3. **Provide actionable insights** that can help the bank improve marketing strategies, optimize resources, and increase subscription rates.

---

## 📊 Dataset Summary

The dataset contains **21 attributes**, which can be grouped into four categories:

1. **Bank Client Data** (e.g., age, job, marital status, education, loans).
2. **Campaign-related Data** (e.g., contact type, last contact month/day, duration).
3. **Other Campaign Attributes** (e.g., number of contacts, previous outcomes).
4. **Social and Economic Context** (e.g., employment variation, consumer confidence index, euribor rate).

* **Target Variable**: `y` → whether the client subscribed to a term deposit (`yes` or `no`).
* **Size**: The dataset includes thousands of client records with both categorical and numeric features.
* **Special Note**: The variable `duration` (call duration in seconds) is highly predictive but unrealistic for real-world deployment since it is only known **after** the call is completed. For practical modeling, this variable should be excluded.

__[Dataset link](https://data.world/data-society/bank-marketing-data)__

---


### 🔍 Project Phases

1. **Data Understanding, Cleaning & Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Selection**
4. **Machine Learning Modeling**
5. **Model Evaluation & Interpretation**
6. **Insights & Recommendations**

---