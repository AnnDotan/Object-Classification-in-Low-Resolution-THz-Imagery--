from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
FIGURES_DIR = Path("artifacts/figures")
TABLES_DIR = Path("artifacts/tables")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# התאמת שמות עמודות אפשריים
# --------------------------------------------------
EPOCH_CANDIDATES = ["epoch", "Epoch"]
VAL_ACC_CANDIDATES = ["val_acc", "val_accuracy", "valid_acc", "accuracy", "top1"]
TRAIN_ACC_CANDIDATES = ["train_acc", "train_accuracy"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_model_name(run_name: str) -> str:
    name = run_name.lower()

    if "transnext" in name:
        return "TransNeXt"
    if "densenet121" in name:
        return "DenseNet121"
    if "resnet50" in name:
        return "ResNet50"
    if "resnet18" in name:
        return "ResNet18"
    if "resnet" in name:
        return "ResNet"
    return "Unknown"


def infer_pretraining(run_name: str) -> str:
    name = run_name.lower()

    if "scratch" in name:
        return "scratch"
    if "_pt_" in name or "pretrained" in name:
        return "pretrained"
    return "unknown"


def infer_linear_probe(run_name: str) -> bool:
    return "linear_probe" in run_name.lower()


def infer_split(run_path: Path) -> str:
    # archive / official / pilot
    parts = [p.lower() for p in run_path.parts]
    for candidate in ["archive", "official", "pilot"]:
        if candidate in parts:
            return candidate
    return "unknown"


def infer_degradation(run_name: str) -> int | None:
    """
    מזהה lowres / lr מתוך שם הריצה.
    דוגמאות:
    - lowres16
    - lr16
    - lowres8
    - lr32
    """
    name = run_name.lower()

    patterns = [
        r"lowres(\d+)",
        r"(?<![a-z])lr(\d+)(?![a-z0-9])",   # יתפוס lr16 אבל לא lr1e-03
    ]

    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            value = int(m.group(1))
            # מסנן learning rates כמו lr1
            if value in [8, 16, 32, 64, 224]:
                return value

    return None


def infer_learning_rate(run_name: str) -> str | None:
    name = run_name.lower()
    m = re.search(r"lr([0-9]+e-[0-9]+)", name)
    if m:
        return m.group(1)
    return None


def collect_runs() -> pd.DataFrame:
    rows = []

    for metrics_path in RUNS_DIR.rglob("metrics.csv"):
        run_dir = metrics_path.parent
        run_name = run_dir.name

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Could not read {metrics_path}: {e}")
            continue

        epoch_col = find_column(df, EPOCH_CANDIDATES)
        val_acc_col = find_column(df, VAL_ACC_CANDIDATES)
        train_acc_col = find_column(df, TRAIN_ACC_CANDIDATES)

        if epoch_col is None or val_acc_col is None:
            print(f"Skipping {metrics_path} בגלל שחסרות עמודות epoch/val_acc")
            continue

        df = df.dropna(subset=[epoch_col, val_acc_col]).copy()

        if df.empty:
            continue

        best_idx = df[val_acc_col].idxmax()
        best_val_acc = float(df.loc[best_idx, val_acc_col])
        best_epoch = int(df.loc[best_idx, epoch_col])

        last_val_acc = float(df[val_acc_col].iloc[-1])
        last_epoch = int(df[epoch_col].iloc[-1])

        row = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "split": infer_split(run_dir),
            "model": infer_model_name(run_name),
            "pretraining": infer_pretraining(run_name),
            "linear_probe": infer_linear_probe(run_name),
            "degradation": infer_degradation(run_name),
            "learning_rate": infer_learning_rate(run_name),
            "epoch_col": epoch_col,
            "val_acc_col": val_acc_col,
            "train_acc_col": train_acc_col,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "last_val_acc": last_val_acc,
            "last_epoch": last_epoch,
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("לא נמצאו קבצי metrics.csv תקינים בתוך runs/")

    summary_df = summary_df.sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    return summary_df


def plot_accuracy_vs_epoch(summary_df: pd.DataFrame) -> None:
    """
    גרף ראשי למצגת.
    נעדיף official, ואם אין מספיק אז נוסיף pilot.
    ניקח ריצה אחת טובה לכל מודל.
    """
    preferred = summary_df[summary_df["split"] == "official"].copy()

    if preferred["model"].nunique() < 2:
        preferred = summary_df.copy()

    selected_rows = []

    for model in ["ResNet50", "ResNet18", "DenseNet121", "TransNeXt"]:
        model_df = preferred[preferred["model"] == model].sort_values("best_val_acc", ascending=False)
        if not model_df.empty:
            selected_rows.append(model_df.iloc[0])

    if not selected_rows:
        print("No runs selected for accuracy_vs_epoch")
        return

    plt.figure(figsize=(10, 6))

    for row in selected_rows:
        metrics_path = Path(row["run_dir"]) / "metrics.csv"
        df = pd.read_csv(metrics_path)

        epoch_col = row["epoch_col"]
        val_acc_col = row["val_acc_col"]

        label = f'{row["model"]} ({row["split"]})'
        plt.plot(df[epoch_col], df[val_acc_col], label=label, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_vs_epoch.png", dpi=300)
    plt.close()


def plot_model_comparison_bar(summary_df: pd.DataFrame) -> None:
    """
    גרף עמודות: הריצה הכי טובה לכל מודל.
    עדיפות ל-official.
    """
    preferred = summary_df[summary_df["split"] == "official"].copy()
    if preferred.empty:
        preferred = summary_df.copy()

    best_rows = []
    for model in ["ResNet50", "ResNet18", "DenseNet121", "TransNeXt"]:
        model_df = preferred[preferred["model"] == model].sort_values("best_val_acc", ascending=False)
        if not model_df.empty:
            best_rows.append(model_df.iloc[0])

    if not best_rows:
        print("No runs selected for model_comparison_bar")
        return

    plot_df = pd.DataFrame(best_rows)

    plt.figure(figsize=(9, 6))
    plt.bar(plot_df["model"], plot_df["best_val_acc"])
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Model Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=300)
    plt.close()


def plot_accuracy_vs_degradation(summary_df: pd.DataFrame) -> None:
    """
    גרף Accuracy vs degradation.
    ייקח רק ריצות עם degradation מזוהה.
    """
    df = summary_df.dropna(subset=["degradation"]).copy()
    if df.empty:
        print("No degradation data found")
        return

    # עדיפות ל-official, אחרת הכל
    preferred = df[df["split"] == "official"].copy()
    if preferred.empty:
        preferred = df.copy()

    # עבור כל מודל וכל degradation ניקח את הריצה הטובה ביותר
    agg = (
        preferred.sort_values("best_val_acc", ascending=False)
        .groupby(["model", "degradation"], as_index=False)
        .first()
    )

    if agg.empty:
        print("No aggregated data for degradation plot")
        return

    plt.figure(figsize=(10, 6))

    for model in ["ResNet50", "ResNet18", "DenseNet121", "TransNeXt"]:
        sub = agg[agg["model"] == model].sort_values("degradation")
        if sub.empty:
            continue

        plt.plot(
            sub["degradation"],
            sub["best_val_acc"],
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.xlabel("Low-Resolution Level")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Accuracy vs Degradation Level")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_vs_degradation.png", dpi=300)
    plt.close()


def main():
    summary_df = collect_runs()
    summary_df.to_csv(TABLES_DIR / "plot_summary.csv", index=False)

    print("Top runs:")
    print(summary_df[["run_name", "split", "model", "best_val_acc", "degradation"]].head(15))

    plot_accuracy_vs_epoch(summary_df)
    plot_model_comparison_bar(summary_df)
    plot_accuracy_vs_degradation(summary_df)

    print("\nSaved files:")
    print(FIGURES_DIR / "accuracy_vs_epoch.png")
    print(FIGURES_DIR / "model_comparison_bar.png")
    print(FIGURES_DIR / "accuracy_vs_degradation.png")
    print(TABLES_DIR / "plot_summary.csv")


if __name__ == "__main__":
    main()