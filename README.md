# kinematic-morphospace

<a href="https://doi.org/10.5281/zenodo.19169784"><img src="https://zenodo.org/badge/1183862622.svg" alt="DOI"></a>

PCA-based decomposition of morphing shape changes in animal locomotion from motion capture data.

---

## Reviewers start here

Two routes — pick whichever feels easier.

| Route | What you do | Setup time | Best if you… |
|---|---|---|---|
| **A. Browser (GitHub Codespaces)** | Click a button, wait, paste a token, run notebooks in the browser. | ~5 min | …want zero local install. |
| **B. Local machine** | Install [uv](https://docs.astral.sh/uv/), clone, run a couple of commands. | ~10 min | …prefer your own editor / faster reruns. |

The dataset (~2.3 GB) lives on Figshare under DOI [`10.6084/m9.figshare.32101528`](https://doi.org/10.6084/m9.figshare.32101528). It is currently **embargoed**; access requires a private share token printed in the manuscript's *Data Availability* section. Once the dataset is published the token is no longer needed.

---

### Route A — GitHub Codespaces (browser, no install)

1. Click **[Open in GitHub Codespaces](https://codespaces.new/LydiaFrance/kinematic-morphospace?quickstart=1)**.
   Sign in with GitHub if prompted, then click *Create codespace on main*.
2. Wait ~3–5 min while the environment builds. A VS Code window opens in the browser. The terminal at the bottom prints a green welcome banner when it's ready.
3. In that terminal, run:
   ```bash
   uv run python scripts/download_figshare_data.py
   ```
   When prompted, paste the share token from the manuscript's *Data Availability* section (or paste the full `https://figshare.com/s/…` URL — either works). The script downloads ~2.3 GB into `./data/` (~5 min on Codespaces).
4. Open any notebook under `examples/hawks/`. Suggested order: `00_ExperimentalSetup.ipynb` → `15_AlternativeMethods.ipynb`. Click *Run All* on each.

**If something fails**: the Codespace is disposable — delete it from <https://github.com/codespaces> and create a fresh one. Free GitHub accounts include 60 hours/month of 4-core Codespaces, which is more than enough for one full review pass.

---

### Route B — Local machine

#### Step 1 — Install [uv](https://docs.astral.sh/uv/)

uv is a single-binary Python package manager. No admin/sudo needed; it installs into your home folder.

<details><summary><b>macOS / Linux</b> (open Terminal)</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then close and reopen Terminal so `uv` is on your `PATH`.
</details>

<details><summary><b>Windows</b> (open PowerShell — search the Start menu for "PowerShell")</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close and reopen PowerShell.
</details>

Verify (any OS):
```bash
uv --version
```
You should see something like `uv 0.5.x`. If you get *"command not found"*, close and reopen the terminal.

#### Step 2 — Get the code

If you have `git`:
```bash
git clone https://github.com/LydiaFrance/kinematic-morphospace.git
cd kinematic-morphospace
```

<details><summary>No git? Download the ZIP instead</summary>

On the GitHub page, click the green **Code** button → **Download ZIP** → unzip it, then `cd` into the folder:

```bash
# macOS / Linux
cd ~/Downloads/kinematic-morphospace-main

# Windows (PowerShell)
cd $env:USERPROFILE\Downloads\kinematic-morphospace-main
```
</details>

#### Step 3 — Install dependencies (all OS)

```bash
uv sync
```

Takes ~2 min. Reads `uv.lock` so the environment is bit-identical to the one used for the manuscript.

#### Step 4 — Download the dataset (all OS)

```bash
uv run python scripts/download_figshare_data.py
```

You'll be prompted for the share token from the manuscript's *Data Availability* section. The script accepts either the bare token or the full `https://figshare.com/s/…` URL. Downloads ~2.3 GB into `./data/`. The script can be re-run safely — already-downloaded files are skipped.

#### Step 5 — Launch the notebooks (all OS)

```bash
uv run jupyter lab
```

Your browser opens JupyterLab. Open `examples/hawks/00_ExperimentalSetup.ipynb` first.

---

### Where to find the share token

Open the manuscript PDF → **Data Availability** section. The share URL ends with a 20-character hex string, e.g.

```
https://figshare.com/s/abcdef0123456789abcd
                       └────────┬─────────┘
                       this is the token
```

Either the bare token or the whole URL works when the script prompts you.

---

### What to expect

| Notebook | Approximate wall time | Produces |
|---|---|---|
| `00_ExperimentalSetup` | <1 min | Dataset overview, sanity checks |
| `01_MarkerReconstructionTrajectories` | ~2 min | Trajectory visualisations |
| `02_BilateralShapePCA` | ~2 min | Bilateral PCA results |
| `03_RotationCorrection` | ~2 min | Body-rotation correction |
| `04_MorphingShapeModes` | ~5 min | Main morphing-shape PCA + permutation tests |
| `05_BeforeAfterRotation` | ~1 min | Rotation visualisation |
| `06_RobustnessValidation` | ~15–25 min | Permutation + bootstrap robustness checks |
| `07_MissingnessAndSamplingBias` | ~5 min | Missingness diagnostics |
| `08_IndividualVsSharedModes` | ~10–20 min | Bird-level vs shared modes (n=2000 perm/boot) |
| `09_VisualisingModes` | ~2 min | Mode visualisation figures |
| `10_MorphingSymmetry` | ~5 min | Bilateral symmetry tests |
| `11_MorphingScoresOverTime` | ~3 min | Temporal score profiles |
| `12_FlightBehaviourContinuum` | ~5 min | Behaviour-continuum analysis |
| `13_ExperimentalEffects` | ~2 min | Experimental-design effects |
| `14_CrossSpeciesGeneralisation` | ~3 min | Cross-species comparison |
| `15_AlternativeMethods` | ~5 min | Comparison with alternative methods |

Wall times are approximate, measured on a 4-core Codespace. A modern laptop is roughly 1.5–2× faster.

Each notebook has a `PAPER_MODE` flag near the top of its config cell. Set it to `True` to run the full permutation/bootstrap counts used in the manuscript; `False` runs a faster, lower-resolution version (figures still render but p-values are coarser). Default is `True`.

---

### OS troubleshooting

| Symptom | Fix |
|---|---|
| Windows: *"running scripts is disabled on this system"* | Run the install command exactly as printed, including `-ExecutionPolicy ByPass`. |
| macOS: *"<binary> cannot be opened because the developer cannot be verified"* | System Settings → Privacy & Security → click *Allow anyway* next to the blocked binary. |
| Any OS: `uv: command not found` after install | Close and reopen the terminal — the installer adds uv to your `PATH`, but existing shells don't pick it up until restart. |
| `uv sync` hangs or times out | You may be behind a corporate proxy. The Codespaces route bypasses local network restrictions. |
| Token rejected by Figshare | Re-check there are no leading/trailing spaces. The token is exactly 20 hex characters. Re-run the download script and paste again. |
| Out of memory in a notebook | The default Codespace machine has 16 GB RAM. If a notebook runs out, in Codespaces switch to a 32 GB machine via *…* menu → *Change machine type*. |

---

## Library API (for code reuse)

If you only want to use the package as a library on your own data, install it
into a uv-managed project:

```bash
uv add kinematic-morphospace
# or, with plotting extras:
uv add "kinematic-morphospace[plot]"
```

```python
import kinematic_morphospace

data = kinematic_morphospace.load_data("path/to/markers.csv")
markers, frame_info = kinematic_morphospace.prepare_marker_data(data)
scaled, scaler = kinematic_morphospace.scale_data(markers)
rotated = kinematic_morphospace.undo_body_rotation(scaled)
pca_model, scores = kinematic_morphospace.run_PCA(rotated)
reconstructed = kinematic_morphospace.reconstruct(pca_model, scores, n_components=4)
kinematic_morphospace.plot_explained(pca_model)  # requires [plot] extra
```

Tests (from a clone):
```bash
uv sync --extra test
uv run pytest tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Distributed under the terms of the [MIT license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LydiaFrance/kinematic-morphospace/workflows/CI/badge.svg
[actions-link]:             https://github.com/LydiaFrance/kinematic-morphospace/actions
[pypi-link]:                https://pypi.org/project/kinematic-morphospace/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/kinematic-morphospace
[pypi-version]:             https://img.shields.io/pypi/v/kinematic-morphospace
<!-- prettier-ignore-end -->
