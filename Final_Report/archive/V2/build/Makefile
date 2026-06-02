# GNU Make entrypoint (mirror of build_all.ps1).
# Run from the repository root:
#   make -f Final_Report/build/Makefile all

PY := python
ROOT := $(CURDIR)
SCRIPTS := Final_Report/scripts

export PYTHONPATH := $(ROOT)

.PHONY: all data tables figures samples reuse refdoc report validate clean

all: data tables figures samples reuse refdoc report validate

data:
	$(PY) $(SCRIPTS)/aggregate_pipelines.py

tables: data
	$(PY) $(SCRIPTS)/build_tables.py

figures: data
	$(PY) $(SCRIPTS)/build_figures.py

samples:
	$(PY) $(SCRIPTS)/build_samples.py

reuse:
	$(PY) $(SCRIPTS)/copy_reused_figures.py

refdoc:
	$(PY) $(SCRIPTS)/make_reference_docx.py

report: tables figures samples reuse refdoc
	$(PY) $(SCRIPTS)/build_report.py pdf docx

validate: report
	$(PY) $(SCRIPTS)/validate_report.py

clean:
	rm -rf Final_Report/build/tex
	rm -f Final_Report/fin-2026-061.pdf Final_Report/fin-2026-061.docx
