# Data

This project uses simulated ATLAS Open Data samples for the study of the Higgs boson dimuon decay channel.

The analysis treats Higgs boson production followed by \(H \rightarrow \mu^+\mu^-\) as the **signal**, while Drell-Yan \(Z/\gamma^* \rightarrow \mu^+\mu^-\) production constitutes the dominant **background**.

## ATLAS Open Data Samples

The original ROOT files are distributed through the CERN Open Data Portal. For convenience, the files were renamed locally to shorter filenames during the analysis. The local filenames listed below are the names expected by the data-preparation scripts in this repository.

### Signal Samples

| DSID | Process | Official ATLAS Open Data file | Local filename |
|---|---|---|---|
| 345097 | \(ggH, H\rightarrow\mu\mu\) | `ODEO_FEB2025_v0_2muons_mc_345097.PowhegPythia8EvtGen_NNLOPS_nnlo_30_ggH125_mumu.2muons.root` | `ggH.root` |
| 345098 | \(ggZH, H\rightarrow\mu\mu\) | `ODEO_FEB2025_v0_2muons_mc_345098.PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125_Hmumu_Zinc.2muons.root` | `ggZH.root` |
| 345106 | \(VBF, H\rightarrow\mu\mu\) | `ODEO_FEB2025_v0_2muons_mc_345106.PowhegPythia8EvtGen_NNPDF30_AZNLOCTEQ6L1_VBFH125_mumu.2muons.root` | `VBF.root` |

### Background Samples

| DSID | Process | Official ATLAS Open Data file | Local filename |
|---|---|---|---|
| 700323 | \(Z/\gamma^*\rightarrow\mu\mu\), BFilter | `ODEO_FEB2025_v0_2muons_mc_700323.Sh_2211_Zmumu_maxHTpTV2_BFilter.2muons.root` | `Zbfilter.root` |
| 700324 | \(Z/\gamma^*\rightarrow\mu\mu\), CFilterBVeto | `ODEO_FEB2025_v0_2muons_mc_700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto.2muons.root` | `ZCfilterBveto.root` |
| 700325 | \(Z/\gamma^*\rightarrow\mu\mu\), CVetoBVeto | `ODEO_FEB2025_v0_2muons_mc_700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto.2muons.root` | `ZCvetoBveto.root` |

### Download Links

The original samples are available directly from CERN Open Data:

**Signal**

- DSID 345097 — ggH:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_345097.PowhegPythia8EvtGen_NNLOPS_nnlo_30_ggH125_mumu.2muons.root

- DSID 345098 — ggZH:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_345098.PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125_Hmumu_Zinc.2muons.root

- DSID 345106 — VBF:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_345106.PowhegPythia8EvtGen_NNPDF30_AZNLOCTEQ6L1_VBFH125_mumu.2muons.root

**Background**

- DSID 700323 — Zmumu BFilter:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_700323.Sh_2211_Zmumu_maxHTpTV2_BFilter.2muons.root

- DSID 700324 — Zmumu CFilterBVeto:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto.2muons.root

- DSID 700325 — Zmumu CVetoBVeto:  
  https://opendata.cern.ch/eos/opendata/atlas/rucio/opendata/ODEO_FEB2025_v0_2muons_mc_700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto.2muons.root

## Data-Preparation Pipeline

The original ROOT samples are not included in this repository.

The three background samples are first combined and divided according to jet multiplicity using:

`src/data_preparation/skim_z_background.cpp`

This produces:

- `Zskimmed_nj0.root` for \(n_j = 0\)
- `Zskimmed_nj1.root` for \(n_j = 1\)
- `Zskimmed_nj2.root` for \(n_j \geq 2\)

The category-specific dataset-building scripts then apply the event selection, construct the relevant kinematic variables, calculate the normalized Monte Carlo event weights, and create the signal and background datasets used for machine-learning training.

| Event category | Dataset-building script | Signal output | Background output |
|---|---|---|---|
| \(n_j = 0\) | `build_nj0_dataset.cpp` | `nj0_signal.root` | `nj0_background.root` |
| \(n_j = 1\) | `build_nj1_dataset.cpp` | `nj1_signal.root` | `nj1_background.root` |
| \(n_j \geq 2\) | `build_nj2plus_dataset.cpp` | `nj2_signal.root` | `nj2_background.root` |

## Repository Policy

ROOT files are intentionally excluded from version control because of their size.

This includes both the original ATLAS Open Data samples and the ROOT files generated during preprocessing and dataset construction. These files are excluded through the repository's `.gitignore`.

The repository instead provides the source code and documentation required to reconstruct the machine-learning datasets from the corresponding ATLAS Open Data samples.