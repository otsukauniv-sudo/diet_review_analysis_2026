Diet review replication package

English version. For Japanese, open Readme_JA.txt.

Analyze mismatches between self-reported weight loss and star ratings.
The package produces group summaries, text-expression comparisons, three product fixed-effects regressions (N1, N1b and N2), and publication figures.

1. Add the data

Extract this package. Put your licensed Diet Review Dataset files directly in Maindata:

  Maindata/review.csv   - review data
  Maindata/product.csv  - product data

Extract the dataset ZIP first. Keep the original CSV filenames and column names.
The dataset is not included.

2. Run the analysis (Windows / VS Code)

Use a PC with Python installed. Open this folder in VS Code, where Analysis.py and requirements.txt are visible.
Open Terminal > New Terminal and use PowerShell.

Install the required packages once:

  python -m pip install -r requirements.txt

Run the analysis:

  python -u .\Analysis.py

The script reads both CSVs from Maindata and runs the summaries, regressions and figures.
Original CSVs are left unchanged. Processing is local; the code does not send review data externally.

3. Output folders and files

Each run creates a new folder under outputs_private/run_YYYYMMDD_HHMMSS_microseconds/.
The paths below are relative to that run folder. Start with:

  summary_tables/group_counts_and_weight_change.csv
  summary_tables/mismatch_counts_and_rates.csv
  regression_results/low_rating_associations_5_percent_loss.csv

Open CSV files in Excel or another CSV reader, PNG files in an image viewer, and JSON files in a text editor.

Run-folder root - Run folder

  run_status.json
    Run status (COMPLETED, PARTIAL or FAILED), analysis settings, Python and package versions, and model status.

data_checks/ - Input checks

  input_data_summary.json
    Input counts, duplicates, product-ID matches, missing values and warnings.
  input_columns.json
    Original CSV columns selected for the analysis.
  sample_selection_counts.csv
    Counts after input checks, product matching, category selection, and text/rating requirements.
  numeric_conversion_counts.csv
    Counts of valid, missing and unparsed weight, height and age values.
  PRIVATE_unparsed_numeric_formats.csv
    Unparsed input formats and their counts; digits are replaced by <num>.
  PRIVATE_rating_conversion.csv
    Original rating labels, their counts, converted star ratings, and missing/invalid classifications.
  product_category_mapping.csv
    Mapping from parent/child product categories to analysis categories, with counts.
  otsuka_sample_count_comparison.csv
    Differences between this run and the four category counts reported in Otsuka’s paper.

summary_tables/ - Group summaries

  group_counts_and_weight_change.csv
    Counts, shares, reported weight change and star-rating summaries for groups A–D and the two neutral groups.
  mismatch_counts_and_rates.csv
    Mismatch counts and rates for groups B and C, with their denominators.
  weight_rating_sample_selection.csv
    Sample losses due to missing/invalid ratings, weights and provisional range restrictions.
  weight_loss_threshold_comparison.csv
    Counts and rates under the fixed 5%, 3%, 1 kg and 3 kg weight-loss criteria.
  group_usage_duration.csv
    Usage-duration counts and shares within each group.
  group_product_categories.csv
    Product-category counts and shares within each group.
  group_text_expression_comparison.csv
    Matched-expression counts, rates and percentage-point differences for B versus A and C versus D.
  category_mismatch_rates.csv
    Category-level mismatch counts, rates and denominators for groups B and C.
  weight_change_breakdown.csv
    Breakdown of groups C and D into small positive loss, no change and weight gain.
  font_check.json
    Japanese-font detection by Analysis.py; Figures.py checks plotting fonts separately.

regression_results/ - Product fixed-effects regressions

  low_rating_associations_5_percent_loss.csv
    Main model (N1): reviews with loss of at least 5%; low rating (1–2 stars) = 1 and 3–5 stars = 0.
  low_vs_high_ratings_5_percent_loss.csv
    Supplementary model (N1b): group B versus group A, excluding 3-star reviews.
  rating_weight_loss_cancellation_interaction.csv
    Model N2: star ratings and the weight-loss-success × cancellation-mention interaction, using reviews with usable weights, ratings and text.
  low_rating_associations_5_percent_loss_covariance.csv
    Covariance matrix of N1 coefficients, used for uncertainty in coefficient combinations.
  low_vs_high_ratings_5_percent_loss_covariance.csv
    Covariance matrix of N1b coefficients.
  rating_weight_loss_cancellation_interaction_covariance.csv
    Covariance matrix of N2 coefficients, used for cancellation associations by weight-loss group.
  cancellation_associations_by_weight_loss.csv
    N2 cancellation associations for loss of at least 5% and loss below 5%, plus their difference.
  regression_status.csv
    Model status, sample size, product count and singleton removals.
  regression_details.json
    Detailed model information: omitted predictors, reference categories, warnings and reasons for non-estimability.

figures/en/ - English journal figures

  figure1_analysis_flow.png / figure1_analysis_flow.svg
    Sample selection, text processing, group comparisons and regressions.
  figure2_text_classification_rules.png / figure2_text_classification_rules.svg
    Text rules, invented examples and the seven regression indicators.
  figure3_weight_loss_rating_groups.png / figure3_weight_loss_rating_groups.svg
    Cross-tabulation of weight-loss status and low, neutral or high star ratings.
  figure4_weight_change_comparison.png / figure4_weight_change_comparison.svg
    Mean loss in A/B and the weight-change breakdown in C/D.
  figure5_text_expression_comparison.png / figure5_text_expression_comparison.svg
    Matched-expression rates for B versus A and C versus D.
  figure6_regression_coefficients.png / figure6_regression_coefficients.svg
    Main-model text coefficients and 95% confidence intervals.

figures/ja/ - Japanese conference figures

  figure1_analysis_flow.png / figure1_analysis_flow.svg
    Conference figure: sample selection and the analysis workflow.
  figure2_text_classification_rules.png / figure2_text_classification_rules.svg
    Conference figure: invented examples of mentions and specific cancellation/symptom wording.
  figure3_weight_change_comparison.png / figure3_weight_change_comparison.svg
    Conference figure: mean loss in A/B and the weight-change breakdown in C/D.
  figure4_text_expression_comparison.png / figure4_text_expression_comparison.svg
    Conference figure: matched-expression rates for B versus A and C versus D.

Coefficient tables include coefficients, standard errors, 95% confidence intervals, p-values, review counts and product counts.
N1 and N1b coefficients are probability differences; multiply by 100 for percentage points.
N2 and cancellation-association coefficients are differences in star points.
All models adjust for product fixed effects, posting year, usage duration and text length, with standard errors clustered by product.
If a model cannot be estimated, its coefficient tables may be absent. Check regression_status.csv or regression_details.json.

Each figure is saved as a 300-dpi PNG and an SVG for scalable graphics.
Figures.py is called automatically and reads saved aggregate CSVs.
The six English journal figures and four Japanese conference figures have different selections and layouts.
Japanese fonts such as Meiryo are detected automatically.
Files beginning PRIVATE_ contain original input labels or formats and are intended for local checking.
This package does not output human validation results for text classification.

4. Package files

  Analysis.py
    Main entry point for the analysis.
  Figures.py
    Creates publication figures from saved aggregates; called automatically by Analysis.py.
  requirements.txt
    Required Python packages.
  README.md
    English guide displayed on GitHub.
  Readme_EN.txt
    English plain-text guide.
  Readme_JA.txt
    Japanese plain-text guide.
  Maindata/README.md
    Bilingual instructions for placing the CSV files.
  LICENSE
    MIT license for the code.
  .gitignore
    Excludes local data and results from Git tracking.

5. Optional commands

For Shift-JIS CSVs:
  python -u .\Analysis.py --encoding cp932

For CSVs stored elsewhere (replace the quoted path):
  python -u .\Analysis.py --data-dir "D:\data\dietreview_data_20231026"

To try the pipeline without the dataset, using automatically generated fictional data:
  python -u .\Analysis.py --demo

To regenerate figures from a saved run (replace run_... with the actual folder name):
  python .\Figures.py ".\outputs_private\run_..."

If the script reports missing CSVs, check that both review.csv and product.csv are directly inside Maindata.
Standard Japanese dataset columns are supported. For other column names, edit DEFAULT_CONFIG near the top of Analysis.py.
Weights and duration use the columns 使用前体重, 使用後体重 and 使用期間.

Scope and license

Reproducing the paper’s numbers requires the same dataset version.
Demo runs produce results for fictional data.
The primary provisional criterion is reported weight loss of at least 5%; low ratings are 1–2 stars and high ratings are 4–5 stars.
A = loss ≥5% / high rating; B = loss ≥5% / low rating; C = loss <5% / high rating; D = loss <5% / low rating.
Three-star reviews remain in two neutral groups. Loss below 5% includes small positive loss, no change and weight gain.
Text-indicator accuracy on real reviews remains unvalidated. Regression coefficients describe conditional associations.
The pipeline has been checked with fictional data. The main counts and regression coefficients were also checked against supplied real-data output from Windows and Python 3.13.2.

The code is licensed under MIT. The dataset is subject to its provider’s terms.
Dataset information: https://www.nii.ac.jp/dsc/idr/diet-review/
Local data and results are excluded from Git tracking.
