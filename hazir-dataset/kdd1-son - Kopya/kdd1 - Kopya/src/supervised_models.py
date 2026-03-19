import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier


def run_supervised_extended(df):
    df = df.copy()

    y = df["label"].astype(int)
    X = df.drop(columns=["label", "anomaly"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    df2 = X.copy()
    df2["label"] = y
    df2 = df2.drop_duplicates()

    y = df2["label"].astype(int)
    X = df2.drop(columns=["label"])

    groups = pd.util.hash_pandas_object(X, index=False)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=120, random_state=42)
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        verbose=0,
        random_state=42
    )

    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    cb.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)
    pred_et = et.predict(X_test)
    pred_gb = gb.predict(X_test)
    pred_cb = cb.predict(X_test)

    vote = ((pred_rf + pred_et + pred_gb + pred_cb) >= 2).astype(int)

    print("RF accuracy:", accuracy_score(y_test, pred_rf))
    print("ET accuracy:", accuracy_score(y_test, pred_et))
    print("GB accuracy:", accuracy_score(y_test, pred_gb))
    print("CB accuracy:", accuracy_score(y_test, pred_cb))

    print("\n[Voting] 4 Model (>=2 oy)")
    print(confusion_matrix(y_test, vote))
    print(classification_report(y_test, vote))

    return rf, et, gb, cb


def predict_vote_extended(df, rf, et, gb, cb):
    df = df.copy()

    X = df.drop(columns=["label", "anomaly"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    p1 = rf.predict(X)
    p2 = et.predict(X)
    p3 = gb.predict(X)
    p4 = cb.predict(X)

    return ((p1 + p2 + p3 + p4) >= 2).astype(int)
