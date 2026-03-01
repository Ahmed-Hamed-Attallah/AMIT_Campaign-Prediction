## Project Code Explanation

### 1. Libraries Import

The project begins by importing essential libraries used for:

* Data manipulation → `pandas`, `numpy`
* Visualization → `plotly`, `matplotlib`, `seaborn`
* Statistical analysis → `scipy`
* Outlier detection → `datasist`
* General utilities → `warnings`, `math`

These tools support analysis, preprocessing, and model development.

---

### 2. Data Loading

The dataset is loaded from:

```
bank-additional-full.csv
```

Using:

```
pd.read_csv(..., sep=';')
```

The semicolon separator is required because the dataset is not comma-separated.

---

### 3. Initial Data Inspection

Performed using:

* `df.info()` → Understand structure & datatypes
* Duplicate checking → Detect repeated records
* Duplicate removal → Clean dataset

Purpose:
Ensure dataset integrity before analysis.

---

### 4. Exploratory Data Analysis (EDA)

EDA is performed to understand:

* Distribution of features
* Relationships with the target variable (`y`)
* Potential data quality issues

Examples:

#### Age Analysis

* Descriptive statistics
* Histogram distribution
* Boxplot for outlier detection
* Investigation of customers aged ≥ 70 and their subscription behavior

Goal:
Check if extreme ages impact campaign success.

---

#### Categorical Features Analysis

Analyzed features:

* Job
* Marital status
* Education
* Default
* Housing loan
* Personal loan
* Contact type
* Month

For each feature:

Steps included:

1. Unique value inspection
2. Value counts
3. Target relationship using:

   ```
   pd.crosstab()
   ```
4. Visual relationship with campaign success using histogram charts

Purpose:
Understand which customer segments respond positively to marketing campaigns.

---

### 5. Handling "Unknown" Values

Columns containing `"unknown"` were evaluated.

Decision logic:

* If meaningful → keep
* If noise → convert to NULL
* Apply imputation where needed

Example:
Housing status was converted and prepared for imputation.

---

### 6. Target Variable Analysis

```
df['y']
```

Checked for:

* Class balance
* Distribution of campaign success vs failure

This is critical for classification modeling.

---

### 7. Feature Relationship Analysis

Used:

```
pd.crosstab(feature, target)
```

To evaluate:

* Behavioral impact
* Financial indicators
* Contact strategy effectiveness

This helps identify predictive signals.

---

### 8. Outlier Investigation

Outliers were not blindly removed.

Instead:

Checked their effect on target outcome before deciding removal.

Example:
Older customers were analyzed before exclusion.

This avoids losing meaningful business insights.

---

### 9. Visualization Strategy

Used Plotly to:

* Understand distributions
* Compare success vs failure
* Detect skewness and imbalance

Visual analysis supports feature engineering decisions.

---

### 10. Dataset Ready for Modeling

After:

* Cleaning
* Understanding distributions
* Handling unknown values
* Checking relationships

The dataset becomes ready for:

* Encoding
* Feature engineering
* Model training

---
