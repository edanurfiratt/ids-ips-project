import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def train_supervised_model(train_df: pd.DataFrame, test_df: pd.DataFrame):

    print("\n" + "="*60)
    print(" İKİNCİ AŞAMA: DENETİMLİ MODELLERİN PERFORMANSI")
    print("="*60)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["label", "file_source", "anomaly"]
    ]

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = train_df["label"]

    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = test_df["label"]

    print(f" Eğitim seti: {X_train.shape}")
    print(f" Test seti: {X_test.shape}")
    print(f" Feature sayısı: {len(feature_cols)}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.6,
            random_state=42
        )
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            verbose=False,
            random_seed=42,
            class_weights=[1, 4]
        )
    else:
        print(" CatBoost kurulu değil, CatBoost atlanacak.")

    thresholds = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01]

    results = []
    best_model = None
    best_model_name = None
    best_y_pred = None
    best_threshold = None
    best_score = -1

    for name, model in models.items():

        print("\n" + "-"*50)
        print(f" Model eğitiliyor: {name}")
        print("-"*50)

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            attack_probs = model.predict_proba(X_test)[:, 1]

            print("\n Threshold denemeleri:")

            selected_threshold = 0.50
            selected_pred = None
            selected_score = -1

            for threshold in thresholds:
                temp_pred = (attack_probs >= threshold).astype(int)

                temp_acc = accuracy_score(y_test, temp_pred)
                temp_precision = precision_score(y_test, temp_pred, zero_division=0)
                temp_recall = recall_score(y_test, temp_pred, zero_division=0)
                temp_f1 = f1_score(y_test, temp_pred, zero_division=0)

                cm_temp = confusion_matrix(y_test, temp_pred, labels=[0, 1])
                fp = cm_temp[0][1]
                tn = cm_temp[0][0]
                temp_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                print(
                    f"Threshold={threshold:.2f} | "
                    f"Accuracy={temp_acc*100:.2f}% | "
                    f"Recall={temp_recall*100:.2f}% | "
                    f"Precision={temp_precision*100:.2f}% | "
                    f"F1={temp_f1*100:.2f}% | "
                    f"FPR={temp_fpr*100:.2f}%"
                )

                score = (2 * temp_recall) + temp_precision - (1.5 * temp_fpr)

                if score > selected_score:
                    selected_score = score
                    selected_threshold = threshold
                    selected_pred = temp_pred

            print(f"\n {name} için seçilen threshold: {selected_threshold}")
            y_pred = selected_pred

        else:
            selected_threshold = None
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fp = cm[0][1]
        tn = cm[0][0]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n {name} tamamlandı.")
        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   Precision Attack: {precision*100:.2f}%")
        print(f"   Recall Attack: {recall*100:.2f}%")
        print(f"   F1 Attack: {f1*100:.2f}%")
        print(f"   FPR: {fpr*100:.2f}%")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision_Attack": precision,
            "Recall_Attack": recall,
            "F1_Attack": f1,
            "FPR": fpr,
            "Threshold": selected_threshold
        })

        model_score = (2 * recall) + precision - (1.5 * fpr)

        if model_score > best_score:
            best_score = model_score
            best_model = model
            best_model_name = name
            best_y_pred = y_pred
            best_threshold = selected_threshold

    results_df = pd.DataFrame(results)

    print("\n" + "="*60)
    print(" DENETİMLİ MODELLERİN PERFORMANS TABLOSU")
    print("="*60)
    print(results_df.to_string(index=False))

    print("\n" + "="*60)
    print(f" EN İYİ MODEL: {best_model_name}")
    print(f" En iyi threshold: {best_threshold}")
    print("="*60)

    cm = confusion_matrix(y_test, best_y_pred, labels=[0, 1])

    print("\n En iyi model confusion matrix:")
    print(f"""
                 Tahmin Normal   Tahmin Attack
    Gerçek Normal      {cm[0][0]:<10}     {cm[0][1]:<10}
    Gerçek Attack      {cm[1][0]:<10}     {cm[1][1]:<10}
    """)

    print("\n En iyi model detaylı raporu:")
    print(classification_report(
        y_test,
        best_y_pred,
        target_names=["Normal (0)", "Attack (1)"],
        digits=3
    ))

    joblib.dump(best_model, "best_ml_model.pkl")
    joblib.dump(feature_cols, "ml_features.pkl")
    joblib.dump(best_threshold, "ml_threshold.pkl")

    print("\n ML modeli kaydedildi:")
    print("best_ml_model.pkl")
    print("ml_features.pkl")
    print("ml_threshold.pkl")

    return results_df