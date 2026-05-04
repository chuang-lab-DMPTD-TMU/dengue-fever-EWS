# ============================================================
# rules/IG.smk — Integrated Gradients attribution for STGAT
# ============================================================
# Computes IG attributions and writes three visualisation figures:
#   ig_feature_importance.png  — global |IG| per input feature (bar chart)
#   ig_temporal_heatmap.png    — mean |IG| over nodes: lookback step × feature
#   ig_node_importance.png     — mean |IG| over time & features per node
#
# Depends on the trained checkpoint produced by train_stgat (stgat.smk).
#
# Run:
#   snakemake report/stgat/sea_baseline/ig_feature_importance.png \
#       --cores 4 --resources gpu=1
#
#   Or trigger all IG outputs at once via rule all_ig below.
# ============================================================


rule viz_ig:
    """
    Compute Integrated Gradients attributions for one trained STGAT run
    and produce three summary figures.
    """
    input:
        checkpoint  = "models/stgat/{run_name}/best.pt",
        test_data   = "data/processed/stgat/{run_name}_test.pt",
        config_file = "config/stgat/{run_name}.yaml"
    output:
        feat_plot    = "report/stgat/{run_name}/ig_feature_importance.png",
        temporal_plot= "report/stgat/{run_name}/ig_temporal_heatmap.png",
        node_plot    = "report/stgat/{run_name}/ig_node_importance.png",
        raw_attrs    = "report/stgat/{run_name}/ig_attributions.pt"
    log:
        "logs/stgat/{run_name}/viz_ig.log"
    params:
        out_dir     = "report/stgat/{run_name}",
        ig_steps    = config.get("ig_steps", 50),
        top_k_feat  = config.get("ig_top_k_feat",  20),
        top_k_nodes = config.get("ig_top_k_nodes", 30),
        cuda_device = config.get("cuda_visible_devices", "0"),
        device      = "cuda" if config.get("use_gpu", True) else "cpu"
    threads: 2
    resources:
        gpu    = 1,
        mem_mb = 16000
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.out_dir} $(dirname {log})

        export CUDA_VISIBLE_DEVICES="{params.cuda_device}"

        python scripts/viz_ig.py \
            --config      {input.config_file}  \
            --checkpoint  {input.checkpoint}   \
            --test-data   {input.test_data}    \
            --out-dir     {params.out_dir}     \
            --steps       {params.ig_steps}    \
            --top-k-feat  {params.top_k_feat}  \
            --top-k-nodes {params.top_k_nodes} \
            --device      {params.device}      \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Convenience target — run IG for all trained STGAT runs
# ---------------------------------------------------------------------------

def _all_ig_outputs(wildcards):
    """Collect IG figure paths for every STGAT run_name found in config/stgat/."""
    import glob as _glob
    run_names = [
        Path(p).stem
        for p in _glob.glob("config/stgat/*.yaml")
    ]
    return [
        f"report/stgat/{rn}/ig_feature_importance.png"
        for rn in run_names
    ]


rule all_ig:
    input:
        _all_ig_outputs
