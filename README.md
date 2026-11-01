# TVM4173 Drinking Water Treatment – interactive notebooks

Self-study notebooks for the course **TVM4173 Drinking Water Treatment** at the Norwegian University of Science and Technology (NTNU). They run in JupyterLab or Jupyter Notebook.

## How to use

- Keep each topic folder together: the notebooks import the Python modules from the same folder.
- Open the start notebook of a topic (for stabilisation: `stabilisation/00_Start_here.ipynb`).
- In a notebook: *Run → Run All Cells*, then read from the top. You do not need to write code, except in the input cells of worksheets (stabilisation, Part C).
- **Before you work in a worksheet**, save your own copy: *File → Save Notebook As…* with a new name.
- Something looks wrong? *Kernel → Restart Kernel and Run All Cells*.

## Contents

### `filtration_models/` – Rapid filtration

`Filtration_models_just_RT.ipynb` with the helper module `filtration_functions.py`: particle transport to filter grains and filtration models.

### `stabilisation/` – Stabilisation and corrosion control (lecture M08)

Start with `00_Start_here.ipynb`.

| Part | Notebooks | What it is |
| --- | --- | --- |
| A – Understanding the chemistry | A1–A8 | CO₂ and carbonic acid, alkalinity and buffering, calcium carbonate saturation, Deffeyes diagram, corrosion control in Norwegian practice, self-test |
| B – A worked design example | B1–B8 | a fictitious plant from design basis to costs (CO₂ + marble filter), plus a groundwater example (stripping + marble); every value with its source |
| C – Your own design | C | worksheet for the semester project: design basis, effect of pre-treatment, doses, unit sizing, points to check, export for the report |

Python modules (in the same folder, needed by the notebooks):

| File | Content |
| --- | --- |
| `carbonate.py` | carbonate chemistry: equilibrium constants, activity, pH, saturation index, CCPP, treatment steps, Deffeyes diagram |
| `stab_views.py` | interactive figures and self-checks of Part A; common look and helpers |
| `design_example.py`, `design_views.py` | values, calculations and panels of Part B |
| `my_design.py`, `my_design_views.py` | calculations, checks and display of Part C |

Pictures from the lecture are not part of this repository. Without them the notebooks still run and refer to the lecture slides instead.

## Requirements

Python 3 with the packages in `requirements.txt` (numpy, scipy, pandas, matplotlib, ipywidgets 8; optional plotly + anywidget for interactive figures).

## Scope and limitations

The notebooks are teaching tools. The chemistry uses the constants of Plummer & Busenberg (1982) with the Davies activity correction and ignores ion pairs; design values are quoted from Norsk Vann reports and the lecture with their source. Results are suitable for learning and for checking a design, not as a substitute for a professional design.

## Licence

- Python code (`*.py`): MIT licence – see `LICENSE`.
- Notebook content (`*.ipynb`): CC BY 4.0 – see `LICENSE-CONTENT.md`. Quoted values from Norsk Vann reports and other sources remain the property of their owners.

## Contact

Thomas Meyn, Department of Civil and Environmental Engineering, NTNU – thomas.meyn@ntnu.no
