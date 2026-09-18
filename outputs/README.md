# Evaluation Outputs Directory

This directory stores evaluation plots, test set metrics, and scientific reports generated during the verification and benchmarking phases.

## Generated Artifacts

- `final_test_evaluation.json`: Final unbiased evaluation on the untouched test set (218 samples) comparing the original classical baseline, the selected `HistGradientBoosting` model (calibrated threshold 0.25), and the `ResNet-18` visual transfer learning model (calibrated threshold 0.48).
- `classical_experiments_report.json`: Comprehensive benchmarking and threshold calibration results across classical classifiers (LogisticRegression, SVM, RandomForest, ExtraTrees, HistGradientBoosting).
- `feature_diagnosis_report.json`: Statistical audit of the 37 global tabular features, documenting univariate ROC-AUCs, Cohen's $d$ effect sizes, Spearman multicollinearity pairs, and Mann-Whitney $U$ test statistics.
- `visual_model_val_report.json`: Validation performance and threshold tuning report for the ResNet-18 visual transfer learning model.
- `metrics.json`: Baseline candidate validation comparison and test metrics.
- `confusion_matrix.png`: Test set confusion matrix visualization.
- `roc_curve.png`: Test set ROC curve visualization.
- `feature_importance.png`: Feature weight and attribution visualization.
- `classification_report.txt`: Detailed per-class precision, recall, and F1 text report.
- `test_cam.png`: Sample Grad-CAM attention heatmap overlay generated during verification.
