# ============================================================
# rules/XAI.smk — Explainability rules (opt-in)
# ============================================================
# Not included in rule all. Trigger via:
#   snakemake all_xai --cores 8 --resources gpu=1
#
# Currently implemented:
#   shap_analysis — GPU SHAP values + beeswarm/dependence plots (XGBoost)
# ============================================================


rule shap_analysis:
    """
    GPU SHAP values + beeswarm/dependence plots for one (study, region).
    Depends on both the trained model and the evaluate results CSV so it
    always runs after evaluation is confirmed complete.
    A flag file is used as output because the real outputs are a directory
    of PNGs whose names aren't known to Snakemake ahead of time.
    """
    input:
        model   = "models/xgboost/{study}/{region}.json",
        results = "report/tables/{study}/{region}_results.csv"
    output:
        flag = ".snakemake/flags/{study}/{region}_shap.done"
    log:
        "logs/{study}/{region}_shap.log"
    params:
        run_cfg   = lambda wc: _run_for_study(wc.study)
    resources:
        gpu = 1
    script:
        "../scripts/shap_analysis.py"
