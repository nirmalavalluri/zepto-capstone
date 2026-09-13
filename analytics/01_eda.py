from pathlib import Path
import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

RAW_FILE = ROOT / "titanic.csv"
report = ["# Titanic Exploratory Analysis\n"]


def note(text):
    print(text)
    report.append(text + "\n")


def table(frame):
    # Text tables require no additional Markdown dependency.
    report.append("```text\n" + frame.to_string() + "\n```\n")


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    report.append(f"![{filename}](outputs/{filename})\n")


def main():
    # The only network/cache dataset load in the whole module.
    # Later stages read this saved raw CSV.
    if RAW_FILE.exists():
        raw = pd.read_csv(RAW_FILE)
        source = "Existing local titanic.csv"
    else:
        raw = sns.load_dataset("titanic")
        raw.to_csv(RAW_FILE, index=False)
        source = "Seaborn Titanic loader; immediately saved to titanic.csv"

    note(f"Dataset source: {source}")
    note(f"Raw shape: {raw.shape}")

    buffer = io.StringIO()
    raw.info(buf=buffer)
    note("## Dataset profile")
    report.append("```text\n" + buffer.getvalue() + "```\n")
    print(buffer.getvalue())
    table(raw.describe())

    balance = raw["survived"].value_counts().sort_index()
    balance_table = pd.DataFrame({
        "count": balance,
        "percentage": balance / balance.sum() * 100,
    })
    note("## Survival class balance")
    table(balance_table)

    note(
        "A stratified modeling split will preserve approximately the "
        "same survived/not-survived proportions in training and test data."
    )

    # Measure all percentages before dropping any rows.
    missing = raw.isna().mean().mul(100)
    affected = missing[missing > 0].sort_values(ascending=False)

    note("## Missing values and cleaning decisions")
    table(affected.rename("missing_percent").to_frame())

    df = raw.copy()

    low_missing = affected[affected < 5].index.tolist()
    if low_missing:
        df = df.dropna(subset=low_missing).copy()

    for column, percentage in affected.items():
        if percentage < 5:
            strategy = "Drop rows missing this column."
        elif percentage <= 30:
            if pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
                df[column] = df[column].fillna(fill_value)
                strategy = (
                    f"Impute with median {fill_value:.3f} for EDA. "
                    "The median is less sensitive to extreme values."
                )
            else:
                fill_value = df[column].mode().iloc[0]
                df[column] = df[column].fillna(fill_value)
                strategy = f"Impute with mode {fill_value!r} for EDA."
        else:
            df = df.drop(columns=column)
            strategy = (
                "Drop this column because most values are missing; "
                "filling them would introduce substantial assumptions."
            )

        note(f"- {column}: {percentage:.4f}% missing. {strategy}")

    df.to_csv(ROOT / "titanic_eda_cleaned.csv", index=False)
    note(f"Cleaned EDA shape: {df.shape}")
    note(
        "titanic.csv remains the unmodified offline fallback. "
        "titanic_eda_cleaned.csv is the exploratory view of that same load. "
        "Modeling will use the raw fallback and fit its own imputer, "
        "encoder, and scaler only within training data."
    )

    note("## Age and fare distributions")
    outlier_rows = []

    for column in ["age", "fare"]:
        values = df[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())

        outlier_rows.append({
            "column": column,
            "Q1": q1,
            "Q3": q3,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": count,
        })

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=df, x=column, bins=30, ax=axes[0])
        sns.boxplot(data=df, x=column, ax=axes[1])
        fig.suptitle(f"{column.title()} distribution")
        save_chart(f"{column}_distribution.png")

        note(
            f"{column.title()} has {count} observations outside the "
            f"IQR bounds [{lower:.3f}, {upper:.3f}]. "
            "These are flagged for inspection and retained because "
            "an extreme value alone does not establish a data error."
        )

    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(OUTPUT / "outliers.csv", index=False)
    table(outliers)

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_modes = df["fare"].mode().tolist()

    note(
        f"Fare mean = {fare_mean:.3f}; median = {fare_median:.3f}; "
        f"mode(s) = {fare_modes}; sample skewness = {df['fare'].skew():.3f}. "
        "Compare these values with the histogram: a mean above the "
        "median and a long upper tail support right skew. The mode "
        "describes the most frequent fare and need not follow a strict "
        "ordering for every skewed distribution."
    )

    note("## Survival rates using boolean masks")
    rows = []

    for sex in sorted(df["sex"].unique()):
        mask = df["sex"].eq(sex)
        rows.append({
            "group": f"sex={sex}",
            "passengers": int(mask.sum()),
            "survival_rate": df.loc[mask, "survived"].mean(),
        })

    for pclass in sorted(df["pclass"].unique()):
        mask = df["pclass"].eq(pclass)
        rows.append({
            "group": f"class={pclass}",
            "passengers": int(mask.sum()),
            "survival_rate": df.loc[mask, "survived"].mean(),
        })

        for sex in sorted(df["sex"].unique()):
            mask = df["sex"].eq(sex) & df["pclass"].eq(pclass)
            rows.append({
                "group": f"sex={sex}, class={pclass}",
                "passengers": int(mask.sum()),
                "survival_rate": df.loc[mask, "survived"].mean(),
            })

    rates = pd.DataFrame(rows)
    rates.to_csv(OUTPUT / "survival_rates.csv", index=False)
    table(rates)

    note("## Correlations")
    columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    correlation = df[columns].corr()
    correlation.to_csv(OUTPUT / "correlations.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
    )
    plt.title("Titanic correlations: six required columns")
    save_chart("correlations.png")

    pairs = [
        (columns[i], columns[j], correlation.iloc[i, j])
        for i in range(len(columns))
        for j in range(i + 1, len(columns))
    ]
    strongest = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)[:2]

    for left, right, coefficient in strongest:
        direction = "positive" if coefficient > 0 else "negative"
        note(
            f"- {left} and {right}: r = {coefficient:.4f}, a "
            f"{direction} association. Higher values of {left} "
            f"tend to accompany {'higher' if coefficient > 0 else 'lower'} "
            f"values of {right}; this does not establish causation."
        )

    note(
        "Passenger class uses 1 for first class and 3 for third class, "
        "so a higher class number means a lower travel class. "
        "adult_male and alone are excluded as required. Each pair "
        "is ranked once, excluding the diagonal and mirrored duplicates."
    )

    note("## Four-chart survival story")

    # Chart 1: sex, passenger class, survival.
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x="pclass", y="survived",
        hue="sex", errorbar=None,
    )
    plt.ylabel("Survival proportion")
    plt.title("Survival by sex and passenger class")
    save_chart("story_1_sex_class.png")

    sex_rates = df.groupby("sex")["survived"].mean()
    note(
        f"Overall female survival was {sex_rates['female']:.1%}, "
        f"compared with {sex_rates['male']:.1%} for males. "
        "The grouped bars show whether this difference also appears "
        "within passenger classes. These are observational comparisons "
        "and do not isolate the effect of sex from all other factors."
    )

    # Chart 2: class, age, survival.
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="pclass", y="age", hue="survived")
    plt.title("Age distributions by class and survival")
    save_chart("story_2_age_class.png")

    age_medians = df.groupby("survived")["age"].median()
    note(
        f"Median age was {age_medians.get(1, float('nan')):.1f} for "
        f"survivors and {age_medians.get(0, float('nan')):.1f} for "
        "non-survivors in the exploratory data. "
        "Class-specific boxes show variation that an overall median "
        "can hide. Age imputation adds observations at the median "
        "and must be considered when interpreting these distributions."
    )

    # Chart 3: sex, fare, survival.
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="sex", y="fare", hue="survived")
    plt.yscale("symlog", linthresh=1)
    plt.title("Fare by sex and survival")
    plt.ylabel("Fare in GBP, symmetric log scale")
    save_chart("story_3_fare_sex.png")

    fare_medians = df.groupby("survived")["fare"].median()
    note(
        f"Median fare was GBP {fare_medians.get(1, float('nan')):.2f} "
        f"for survivors and GBP {fare_medians.get(0, float('nan')):.2f} "
        "for non-survivors. The transformed axis keeps zero fares "
        "visible while compressing the long upper tail. Fare is "
        "associated with travel class, so this does not show that "
        "paying more directly caused survival."
    )

    # Chart 4: family size, sex, survival.
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df, x="family_size", y="survived",
        hue="sex", errorbar=None,
    )
    plt.ylabel("Survival proportion")
    plt.title("Survival by family size and sex")
    save_chart("story_4_family_sex.png")

    family_summary = df.groupby(
        ["family_size", "sex"]
    )["survived"].agg(["count", "mean"])
    table(family_summary)

    note(
        "Family size counts the passenger plus siblings/spouses and "
        "parents/children aboard. The chart examines whether the "
        "sex-related survival pattern varies across family sizes. "
        "Consult the accompanying counts: rates for very small "
        "groups are unstable and should not drive strong conclusions."
    )

    note("## Exploratory standardization")
    standardized = df[["age", "fare"]].copy()
    summaries = []

    for column in standardized:
        before = standardized[column]
        standardized[column] = (
            (before - before.mean()) / before.std(ddof=0)
        )
        summaries.append({
            "column": column,
            "before_mean": before.mean(),
            "before_std_population": before.std(ddof=0),
            "after_mean": standardized[column].mean(),
            "after_std_population": standardized[column].std(ddof=0),
        })

    summary = pd.DataFrame(summaries)
    table(summary)
    summary.to_csv(OUTPUT / "standardization.csv", index=False)

    assert np.allclose(standardized.mean(), 0, atol=1e-10)
    assert np.allclose(standardized.std(ddof=0), 1, atol=1e-10)

    note(
        "Both transformed columns have approximately mean 0 and "
        "population standard deviation 1. This full-data calculation "
        "is an EDA check only; these values will not enter modeling."
    )

    report_path = ROOT / "EDA_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"Charts and statistics: {OUTPUT}")
    print("Module 2 EDA completed successfully.")


if __name__ == "__main__":
    main()