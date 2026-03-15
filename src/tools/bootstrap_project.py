from __future__ import annotations

from pathlib import Path


DIRS = [
    "artifacts",
    "artifacts/figures",
    "artifacts/tables",
    "artifacts/summaries",
    "configs",
    "runs",
    "runs/pilot",
    "runs/official",
    "src/tools",
]

JOURNAL_TEMPLATE = """# Experiment Journal

מטרת הקובץ:
לתעד כל ניסוי בצורה ידנית-מחקרית, בנוסף ללוגים האוטומטיים.

## איך לתעד כל ניסוי
לכל ניסוי להוסיף:
- תאריך
- מטרה
- מודל
- האם pretrained
- out_size
- low_res
- epochs
- batch_size
- subset
- best_val_acc
- זמן ריצה
- מסקנה קצרה
- האם זה pilot או official

---

## Suggested Sections

### Pilot Runs
ניסויי sanity check, בדיקות מהירות, בדיקות pipeline.

### Official Runs
ניסויי baseline, DenseNet, TransNeXt, וניסויי השוואה שיופיעו בדוח.

"""

GITIGNORE_TEMPLATE = r""".venv/
.idea/
__pycache__/
*.pyc
*.pyo
*.pyd
desktop.ini

artifacts/figures/
artifacts/tables/

# keep run outputs if you want, otherwise uncomment:
# runs/
"""


def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main():
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

    write_if_missing(Path("artifacts/summaries/experiment_journal.md"), JOURNAL_TEMPLATE)
    write_if_missing(Path(".gitignore"), GITIGNORE_TEMPLATE)

    print("Project structure initialized successfully.")
    print("Created/verified directories and documentation templates.")


if __name__ == "__main__":
    main()