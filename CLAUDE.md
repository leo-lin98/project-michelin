# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDA and analysis on the Michelin Guide restaurants dataset (2021, sourced from Kaggle via `kagglehub`). The dataset covers ~18,800 restaurants globally with columns: `Name`, `Address`, `Location`, `Price`, `Cuisine`, `Longitude`, `Latitude`, `PhoneNumber`, `Url`, `WebsiteUrl`, `Award`, `GreenStar`, `FacilitiesAndServices`, `Description`.

`Award` values: `"1 Star"`, `"2 Stars"`, `"3 Stars"`, `"Bib Gourmand"`, `"Selected Restaurants"`.

Dataset is downloaded automatically by `kagglehub` to `~/.cache/kagglehub/` — do not commit data files.

## Environment & Commands

This project uses `uv` for dependency management with Python 3.14.

```bash
# Install dependencies
uv sync

# Run a script
uv run python eda.py

# Add a dependency
uv add <package>
```

## Architecture

`eda.py` is the main analysis script — it loads the dataset via `kagglehub`, parses the `Location` column to extract a `country` field, and filters by `Award`. `main.py` is an unused boilerplate placeholder.

As the project grows, keep analysis in `eda.py` (or notebooks via `uv run jupyter lab`) and extract reusable logic (data loading, feature engineering) into `.py` modules under a `src/` directory.
