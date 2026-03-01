import pandas as pd
import numpy as np


def preprocess_input(df):
    """
    df: DataFrame with the original 20 columns (single row).
    Returns: DataFrame with the exact 32 features your model expects.
    """

    # Work on copy
    data = df.copy()

    # 1. Log transformations
    data['age_logs'] = np.log1p(data['age'])
    data['duration_logs'] = np.log1p(data['duration'])
    data['campaign_logs'] = np.log1p(data['campaign'])
    data['pdays_logs'] = np.log1p(data['pdays'])
    data['previous_logs'] = np.log1p(data['previous'])
    data['nr.employed_logs'] = np.log1p(data['nr.employed'])



    # 2. Cyclic encoding of month
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    month_num = data['month'].map(month_map).astype(float)
    data['month_sin'] = np.sin(2 * np.pi * month_num / 12)
    data['month_cos'] = np.cos(2 * np.pi * month_num / 12)



    # 3. Binary encoding
    data['default'] = data['default'].replace({
        'no': 0,
        'yes': 1,
        'unknown': 1
    })

    # 4. Feature engineering (match training logic)

    # Previous contact flag
    data['previous_contact'] = (data['pdays'] != 999).astype(int)

    # Season
    def classify_season(month):
        if month in ['dec', 'jan', 'feb']:
            return "Winter"
        elif month in ['mar', 'apr', 'may']:
            return "Spring"
        elif month in ['jun', 'jul', 'aug']:
            return "Summer"
        elif month in ['sep', 'oct', 'nov']:
            return "Autumn"
        return "Invalid"

    data['season'] = data['month'].apply(classify_season)

    # Age category
    data['age_category'] = pd.cut(
        data['age'],
        bins=[0, 30, 45, 60, 120],
        labels=['Young', 'Mid-Young', 'Mid-Old', 'Old'],
        include_lowest=True
        )

    # Contact readiness
    data['contact_readiness'] = (
        (data["previous"] > 0).astype(int) +
        (data["pdays"] != 999).astype(int) +
        (data["poutcome"] == "success").astype(int)
    )

    # Economic pressure
    data["economic_pressure"] = (
        data["emp.var.rate"] +
        data["cons.conf.idx"] +
        data["euribor3m"]
    )

    # Financial stress (use original categorical BEFORE encoding logic change)
    data["financial_stress"] = (
        (data["loan"] == "yes").astype(int) +
        (data["housing"] == "yes").astype(int) +
        (data["default"] == 1).astype(int)
    )

    # Engagement intensity
    data["engagement_intensity"] = data["campaign"] / (data["previous"] + 1)

    # Season day
    data["season_day"] = data["season"] + "_" + data["day_of_week"]

    # Market momentum
    data["market_momentum"] = data["cons.price.idx"] * data["euribor3m"]

    # Contact fatigue
    data["contact_fatigue"] = data["campaign"] * data["pdays_logs"]

    # Job economy fit
    data["job_economy_fit"] = data["job"] + "_" + data["season"]

    # 5. Final column order
    final_columns = [
        'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'day_of_week', 'poutcome', 'emp.var.rate',
        'cons.price.idx', 'cons.conf.idx', 'euribor3m',
        'month_sin', 'month_cos', 'previous_contact',
        'age_logs', 'duration_logs', 'campaign_logs', 'pdays_logs',
        'previous_logs', 'nr.employed_logs', 'season', 'age_category',
        'contact_readiness', 'economic_pressure', 'financial_stress',
        'engagement_intensity', 'season_day', 'market_momentum',
        'contact_fatigue', 'job_economy_fit'
    ]

    # Safety check
    missing_cols = [c for c in final_columns if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns after preprocessing: {missing_cols}")

    return data[final_columns]