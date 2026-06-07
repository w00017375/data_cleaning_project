# Data Analysis Studio refactor

streamlit link: https://datacleaningproject-pbebitrdhqjggrngwiptxh.streamlit.app/

This folder contains the same Streamlit application split into modules.

- `app.py` — Streamlit entry point. It configures the page and calls each tab renderer.
- `utils/app_setup.py` — page configuration and CSS.
- `utils/dependencies.py` — optional dependency imports and availability flags.
- `utils/state.py` — session state, flash messages, and recipe logging helpers.
- `utils/loaders.py` — uploaded file and Google Sheets loading.
- `utils/dataframe_utils.py` — dataframe profiling, recipe replay, export cache helpers, dtype helpers.
- `utils/validation_utils.py` — validation rule execution.
- `utils/visualization_utils.py` — chart suggestions, chart rendering, visual constants.
- `utils/tabs.py` — tab-level UI functions.

Run with:

```
pip install -r requirements.txt
streamlit run app.py
```


## Error handling

The app catches unsupported files, incompatible column/data selections, and unexpected pandas/Streamlit edge cases at the tab level. Instead of showing a raw traceback, it displays a user-facing message with a short technical detail section.

## Update: duplicate handling and chart previews

This version adds duplicate-row tools for both full-row matching and subset-column matching. Duplicate counting now displays the exact rows included in the count, and duplicate removal supports keeping either the first or last occurrence.

Manual chart-type previews were also simplified: the preview wireframes are independent from the data, use dark elements on a white background, and avoid small axis text.

## Update: missing values and visualization controls

- The Upload & Overview tab is read-only: duplicate information there is limited to full-row duplicate counting and preview.
- Duplicate removal and subset-based duplicate checks stay in Cleaning & Prep.
- Missing-value tools now include null-percent column dropping, time-series fill from previous/next rows after date sorting, categorical most-frequent fill, and grouped numeric fills by multiple categorical columns.
- Visualization controls now restrict incompatible column choices more strictly and support aggregated measures such as mean, sum, max, min, median, count, and distinct count where relevant.
- Created charts include a **Show numbers** option.


Demo video of the project usage
https://youtu.be/yVH9bTe2cE0