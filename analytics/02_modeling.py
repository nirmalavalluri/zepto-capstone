from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

SEED = 42
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]
REPORT = ["# Titanic Modeling Report\n"]


def note(text):
    print(text)
    REPORT.append(text + "\n")


def show_table(frame):
    text = frame.to_string(index=False)
    print(text)
    REPORT.append("```text\n" + text + "\n```\n")


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    REPORT.append(f"![{filename}](outputs/{filename})\n")


def make_preprocessor(numeric_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])


def make_classifier(estimator):
    return Pipeline([
        ("preprocess", make_preprocessor(NUMERIC)),
        ("model", estimator),
    ])


def evaluate(name, pipeline, X_test, y_test):
    predicted = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(
            y_test, predicted, zero_division=0
        ),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }


def main():
    raw_file = ROOT / "titanic.csv"

    if not raw_file.exists():
        raise FileNotFoundError(
            "Run analytics/01_eda.py first to create titanic.csv."
        )

    df = pd.read_csv(raw_file)

    note("## Data source and preprocessing")
    note(
        "This script continues from titanic.csv saved by 01_eda.py. "
        "It does not independently reload the dataset from Seaborn. "
        "The raw fallback is retained so full-data exploratory "
        "imputation and scaling do not enter model evaluation."
    )
    note(
        "Numeric features use training-only median imputation and "
        "StandardScaler. Categorical features use training-only mode "
        "imputation and one-hot encoding. Each preprocessing operation "
        "is fitted inside the training folds during cross-validation."
    )
    note(
        "The target is survived. alive is excluded because it reveals "
        "the target directly. Redundant fields such as class, "
        "embark_town, who, adult_male, and alone are excluded. "
        "deck is omitted because of its high missingness."
    )

    X = df[NUMERIC + CATEGORICAL].copy()
    y = df["survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )

    balance = pd.DataFrame([
        {
            "split": label,
            "rows": len(target),
            "not_survived": int((target == 0).sum()),
            "survived": int((target == 1).sum()),
            "survival_rate": target.mean(),
        }
        for label, target in [
            ("full", y),
            ("train", y_train),
            ("test", y_test),
        ]
    ])

    note("## Class balance and stratified split")
    show_table(balance.round(4))
    balance.to_csv(OUTPUT / "class_balance.csv", index=False)

    note(
        f"The full survival proportion is {y.mean():.2%}. "
        "Stratification preserves approximately the same class "
        "proportions in training and test data, reducing the risk "
        "of an unrepresentative test split."
    )

    models = {
        "Logistic Regression": make_classifier(
            LogisticRegression(
                max_iter=2000, random_state=SEED
            )
        ),
        "Decision Tree": make_classifier(
            DecisionTreeClassifier(
                max_depth=5, random_state=SEED
            )
        ),
        "Random Forest": make_classifier(
            RandomForestClassifier(
                n_estimators=200,
                random_state=SEED,
                n_jobs=1,
            )
        ),
    }

    classifier_rows = []
    selection_scores = {}

    note("## Three classifier comparison")

    for name, pipeline in models.items():
        print(f"\nTraining {name}...", flush=True)

        fold_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )

        selection_scores[name] = float(fold_scores.mean())

        pipeline.fit(X_train, y_train)
        classifier_rows.append(
            evaluate(name, pipeline, X_test, y_test)
        )

        predictions = pipeline.predict(X_test)
        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        matrix_frame = pd.DataFrame(
            matrix,
            index=["actual_not_survived", "actual_survived"],
            columns=["predicted_not_survived", "predicted_survived"],
        )

        slug = name.lower().replace(" ", "_")
        matrix_frame.to_csv(OUTPUT / f"{slug}_confusion.csv")

        note(f"### {name} confusion matrix")
        show_table(matrix_frame.reset_index(names="actual_class"))

        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Not survived", "Survived"],
        ).plot(cmap="Blues")

        plt.title(name)
        save_chart(f"{slug}_confusion.png")

    comparison = pd.DataFrame(classifier_rows)
    comparison.to_csv(
        OUTPUT / "classifier_metrics.csv", index=False
    )
    show_table(comparison.round(4))

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipeline in models.items():
        RocCurveDisplay.from_estimator(
            pipeline, X_test, y_test, name=name, ax=ax
        )

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Held-out classifier ROC curves")
    save_chart("classifier_roc.png")

    tree_pipeline = models["Decision Tree"]
    feature_names = (
        tree_pipeline.named_steps["preprocess"]
        .get_feature_names_out()
    )

    plt.figure(figsize=(26, 14))
    plot_tree(
        tree_pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=6,
    )
    save_chart("decision_tree.png")

    note("## Imbalance handling comparison")

    imbalance_models = {
        "Baseline": make_classifier(
            LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "Balanced weights": make_classifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
        "SMOTE": ImbalancedPipeline([
            ("preprocess", make_preprocessor(NUMERIC)),
            ("smote", SMOTE(random_state=SEED)),
            ("model", LogisticRegression(
                max_iter=2000, random_state=SEED
            )),
        ]),
    }

    imbalance_rows = []

    for name, pipeline in imbalance_models.items():
        print(f"Evaluating imbalance strategy: {name}", flush=True)

        fold_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        row = evaluate(name, pipeline, X_test, y_test)
        row["training_cv_f1"] = float(fold_scores.mean())
        imbalance_rows.append(row)

    imbalance = pd.DataFrame(imbalance_rows)
    imbalance.to_csv(
        OUTPUT / "imbalance_comparison.csv", index=False
    )

    show_table(
        imbalance[
            ["model", "precision", "recall", "f1", "training_cv_f1"]
        ].round(4)
    )

    strongest_strategy = imbalance.loc[
        imbalance["training_cv_f1"].idxmax()
    ]

    note(
        f"{strongest_strategy['model']} achieved the highest training "
        f"CV F1 among the imbalance variants "
        f"({strongest_strategy['training_cv_f1']:.4f}). "
        f"Its test precision was {strongest_strategy['precision']:.4f}, "
        f"recall {strongest_strategy['recall']:.4f}, and "
        f"F1 {strongest_strategy['f1']:.4f}. "
        "F1 is used to balance precision and recall."
    )
    note(
        "SMOTE is inside an imbalanced-learn pipeline, so only "
        "training folds are oversampled. Test and validation rows "
        "are never resampled. Applying ordinary SMOTE after one-hot "
        "encoding can produce fractional indicator values; this "
        "is a limitation of this baseline experiment."
    )

    note("## Random Forest tuning")
    print("Running Random Forest GridSearchCV...", flush=True)

    search = GridSearchCV(
        estimator=make_classifier(
            RandomForestClassifier(
                oob_score=True,
                bootstrap=True,
                random_state=SEED,
                n_jobs=1,
            )
        ),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, None],
            "model__max_features": ["sqrt", 0.8],
        },
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    tuned_pipeline = search.best_estimator_
    tuned_metrics = evaluate(
        "Tuned Random Forest", tuned_pipeline, X_test, y_test
    )
    oob_score = float(
        tuned_pipeline.named_steps["model"].oob_score_
    )

    note(f"Best parameters: {search.best_params_}")
    note(f"Best training CV F1: {search.best_score_:.4f}")
    note(f"OOB accuracy: {oob_score:.4f}")

    show_table(pd.DataFrame([tuned_metrics]).round(4))

    note(
        "OOB accuracy is a training diagnostic, not the same metric "
        "as CV F1. The final preprocessor is fitted on the training "
        "split, so OOB performance does not replace fold-isolated "
        "cross-validation or held-out evaluation."
    )

    tuning_results = {
        "best_parameters": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "oob_accuracy": oob_score,
        "test_metrics": tuned_metrics,
    }

    (OUTPUT / "random_forest_tuning.json").write_text(
        json.dumps(tuning_results, indent=2),
        encoding="utf-8",
    )

    # Select the saved pipeline using training CV scores only.
    candidates = dict(models)
    candidates["Tuned Random Forest"] = tuned_pipeline
    selection_scores["Tuned Random Forest"] = float(
        search.best_score_
    )

    for name in ["Balanced weights", "SMOTE"]:
        candidate_name = f"Logistic Regression ({name})"
        candidates[candidate_name] = imbalance_models[name]
        selection_scores[candidate_name] = float(
            imbalance.loc[
                imbalance["model"] == name, "training_cv_f1"
            ].iloc[0]
        )

    selected_name = max(
        selection_scores, key=selection_scores.get
    )
    selected_pipeline = candidates[selected_name]
    selected_metrics = evaluate(
        selected_name, selected_pipeline, X_test, y_test
    )

    note("## Classifier recommendation")

    selection_table = pd.DataFrame([
        {"model": name, "training_cv_f1": score}
        for name, score in selection_scores.items()
    ])
    show_table(selection_table.round(4))
    selection_table.to_csv(
        OUTPUT / "selection_scores.csv", index=False
    )

    note(
        f"I recommend {selected_name} as the candidate for further "
        f"validation because it had the highest training CV F1 "
        f"among the evaluated candidates "
        f"({selection_scores[selected_name]:.4f}). "
        f"Its held-out accuracy was {selected_metrics['accuracy']:.4f}, "
        f"precision {selected_metrics['precision']:.4f}, "
        f"recall {selected_metrics['recall']:.4f}, "
        f"F1 {selected_metrics['f1']:.4f}, and "
        f"AUC {selected_metrics['auc']:.4f}. "
        "Selection used training CV scores rather than test scores. "
        "The tuned forest's best CV score is subject to tuning "
        "selection optimism, so independent validation would be "
        "needed before deployment. Titanic data supports this "
        "educational comparison, not operational deployment claims."
    )

    model_path = OUTPUT / "best_survival_pipeline.joblib"
    joblib.dump(selected_pipeline, model_path)

    reloaded = joblib.load(model_path)
    sample = X_test.head(5).copy()

    np.testing.assert_array_equal(
        selected_pipeline.predict(sample),
        reloaded.predict(sample),
    )

    sample.to_csv(
        OUTPUT / "sample_raw_passengers.csv", index=False
    )

    reload_results = sample.copy()
    reload_results["predicted_survived"] = reloaded.predict(sample)
    reload_results.to_csv(
        OUTPUT / "reload_predictions.csv", index=False
    )

    note(
        "PASS: The complete saved pipeline was reloaded and produced "
        "identical predictions on raw, unprocessed passenger records."
    )

    note("## Fare regression")

    regression_numeric = ["pclass", "age", "sibsp", "parch"]
    regression_features = regression_numeric + CATEGORICAL

    regression_data = df.dropna(subset=["fare"])
    Xr = regression_data[regression_features]
    yr = regression_data["fare"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.20, random_state=SEED
    )

    regression = Pipeline([
        ("preprocess", make_preprocessor(regression_numeric)),
        ("model", LinearRegression()),
    ])

    regression.fit(Xr_train, yr_train)
    predicted_fare = regression.predict(Xr_test)

    r2 = r2_score(yr_test, predicted_fare)
    n = len(yr_test)
    p = regression.named_steps["preprocess"].transform(
        Xr_train
    ).shape[1]

    adjusted_r2 = (
        1 - (1 - r2) * (n - 1) / (n - p - 1)
        if n > p + 1 else float("nan")
    )

    regression_metrics = {
        "model": "Linear Regression",
        "mae": mean_absolute_error(yr_test, predicted_fare),
        "rmse": float(np.sqrt(
            mean_squared_error(yr_test, predicted_fare)
        )),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }

    regression_table = pd.DataFrame([regression_metrics])
    show_table(regression_table.round(4))
    regression_table.to_csv(
        OUTPUT / "regression_metrics.csv", index=False
    )

    note(
        "Regression predicts fare using class, age, family counts, "
        "sex, and embarkation port. Fare and survival-related "
        "outcomes are excluded from predictors. "
        f"Adjusted R-squared applies the requested adjustment using "
        f"{n} held-out rows and {p} encoded predictors, excluding "
        "the intercept; it is a descriptive held-out adjustment."
    )

    residuals = yr_test.to_numpy() - predicted_fare

    plt.figure(figsize=(8, 5))
    plt.scatter(predicted_fare, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Actual minus predicted fare")
    plt.title("Held-out fare regression residuals")
    save_chart("fare_residuals.png")

    residual_frame = pd.DataFrame({
        "predicted_fare": predicted_fare,
        "residual": residuals,
    })
    residual_frame.to_csv(
        OUTPUT / "regression_residuals.csv", index=False
    )

    residual_frame["prediction_band"] = pd.qcut(
        residual_frame["predicted_fare"],
        q=4,
        duplicates="drop",
    )

    spread = (
        residual_frame
        .groupby("prediction_band", observed=True)["residual"]
        .agg(["count", "std"])
        .reset_index()
    )
    show_table(spread)

    standard_deviations = spread["std"].dropna()
    spread_ratio = (
        standard_deviations.max() / standard_deviations.min()
        if len(standard_deviations) >= 2
        and standard_deviations.min() > 0
        else float("nan")
    )

    if np.isnan(spread_ratio):
        conclusion = (
            "The spread ratio cannot be evaluated reliably; "
            "inspect the residual plot directly."
        )
    elif spread_ratio > 2:
        conclusion = (
            "The residual spread varies substantially across fitted "
            "values, suggesting heteroscedasticity."
        )
    else:
        conclusion = (
            "This check does not flag strong heteroscedasticity "
            "using a spread-ratio threshold of 2."
        )

    note(
        f"Residual standard-deviation ratio across prediction bands: "
        f"{spread_ratio:.3f}. {conclusion} "
        "This threshold is a descriptive heuristic, not a formal "
        "statistical test; interpret it alongside the residual plot."
    )

    note("## Final model comparison")

    classifier_group = pd.concat(
        [comparison, pd.DataFrame([tuned_metrics])],
        ignore_index=True,
    )

    if selected_name not in classifier_group["model"].values:
        classifier_group = pd.concat(
            [classifier_group, pd.DataFrame([selected_metrics])],
            ignore_index=True,
        )

    classifier_group = classifier_group.rename(columns={
        key: f"classification_{key}"
        for key in ["accuracy", "precision", "recall", "f1", "auc"]
    })

    regression_group = regression_table.rename(columns={
        key: f"regression_{key}"
        for key in ["mae", "rmse", "r2", "adjusted_r2"]
    })

    final_comparison = pd.concat(
        [classifier_group, regression_group],
        ignore_index=True,
    )

    final_comparison.to_csv(
        OUTPUT / "final_model_comparison.csv", index=False
    )
    show_table(final_comparison.round(4))

    note(
        "Classification and regression metrics are separate groups "
        "and are not comparable on one shared scale. Missing cells "
        "mean that the metric does not apply to that model type."
    )

    report_path = ROOT / "MODELING_REPORT.md"
    report_path.write_text(
        "\n".join(REPORT), encoding="utf-8"
    )

    print("\nPASS: Saved pipeline reload predictions match.")
    print(f"Selected classifier: {selected_name}")
    print(f"Report: {report_path}")
    print("Module 2 modeling completed successfully.")


if __name__ == "__main__":
    main()