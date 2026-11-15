# 🧠 Data Preprocessing Studio

A modern, single-file **desktop application** for exploring, cleaning, and preparing tabular datasets (CSV / Excel) for machine learning — built with **Python**, **PySide6**, **pandas**, **scikit-learn**, and **matplotlib**.

> Explore → Clean → Encode → Scale → Handle Outliers → Split → Export — all inside one polished dark-themed GUI.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-green)
![pandas](https://img.shields.io/badge/pandas-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow)

---

## ✨ Features

### 🔍 Explore
- Preview first / last 5 rows
- `df.info()` and `df.describe()` views
- Data types & uniqueness summary
- **Paginated distribution charts** (histograms or box plots) — **2 graphs per page** with ◀ ▶ navigation

### 🩹 Missing Values
- Per-column missing count and percentage
- 8 strategies: drop rows, drop columns, fill mean / median / mode / constant, forward-fill, backward-fill
- Automatically skips non-applicable columns (e.g. mean on a string column)

### 🧬 Duplicates
- Detect duplicate rows (full or on a subset)
- Remove duplicates with `keep = first / last / none`

### 🔢 Encoding
- Label Encoding, Ordinal Encoding, One-Hot Encoding, Frequency Encoding
- **Live data preview** — see encoded values immediately on the same page

### ⚖️ Scaling
- Standard (Z-score), Min-Max (0-1), Robust (median/IQR), Max-Abs
- Before-scaling statistics (`mean`, `std`, `min`, `max`) for reference
- **Live data preview** of scaled values

### 🎯 Outliers
- Detection: IQR (with multiplier) or Z-score (with threshold)
- Actions: Remove rows, Cap to bounds, Replace with median
- Box plot visualization
- **Live data preview** after applying

### ✂️ Split
- Optional target column
- Configurable test size, random state, shuffle, and stratification
- Caches train/test DataFrames for export

### 💾 Export
- Export cleaned dataset as **CSV** or **Excel**
- Export train/test splits as two CSVs

### 🎨 UI / UX
- **Dark-navy theme** with gradient cards, rounded corners, and soft drop shadows
- Sidebar + top-tab navigation (both in sync)
- **Stat cards** showing row count, columns, numeric features, missing %
- **Color-coded readiness banner** (grey → green → amber)
- Vector icons drawn with `QPainter` — no external assets required
- Embedded base64 robot mascot artwork (with QPainter fallback)
- **Ctrl + / Ctrl −** to zoom the interface (80% – 160%)

---

## 🚀 Installation & Setup

### Requirements
- Python **3.9 or newer**
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/4103awi-tech/Data-Preprocessing.git
cd Data-Preprocessing
