###############################################################################
# Rule: Merge CSVs → One NetCDF per (clim_factor, AR6)
###############################################################################
# Helper function to collect CSVs safely
def collect_AR6_csvs(wc):
    folder = Path(AR6_RAW_DIR) / wc.clim_factor / wc.ar6
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder.resolve()}")
    return csv_files


rule AR6_convert_and_merge_across_years:
    input:
        lambda wc: collect_AR6_csvs(wc)
    output:
        AR6_NETCDF_DIR / "{clim_factor}" / "{ar6}.nc"
    threads: 12
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "AR6_convert_and_merge_across_years.py"

###############################################################################
# Rule: Combine all AR6 NetCDFs for each climate factor
###############################################################################

rule AR6_merge_across_models:
    input:
        lambda wc: expand(
            AR6_NETCDF_DIR / wc.clim_factor / "{AR6}.nc",
            AR6=ssp_scenarios
        )
    output:
        AR6_NETCDF_DIR / "combined_{clim_factor}.nc"
    threads: 6
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "AR6_merge_across_models.py"

###############################################################################
# Rule: Merge all climate factors into a single AR6_all.nc
###############################################################################

rule AR6_merge_across_clim_factors:
    input:
        expand(AR6_NETCDF_DIR / "combined_{clim_factor}.nc", clim_factor=ar6_clim_factors)
    output:
        AR6_NETCDF_DIR / "AR6_all.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "merge_across_clim_factors.py"


###############################################################################
# Rule: Slice AR6 all.nc depending on provided configuration
###############################################################################

rule AR6_get_slice_netcdf:
    input:
        AR6_NETCDF_DIR / "AR6_all.nc"
    output:
        AR6_NETCDF_DIR / "AR6_all_sliced.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "get_slice_netcdf.py"


###############################################################################
# Rule: Convert sliced NetCDF to TSV
###############################################################################
rule AR6_netcdf_slice_to_tsv:
    input:
        AR6_NETCDF_DIR / "AR6_all_sliced.nc"
    output:
        AR6_NETCDF_DIR / "AR6_all_sliced.tsv"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "netcdf_slice_to_tsv.py"

