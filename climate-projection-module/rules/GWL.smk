###############################################################################
# Rule: Merge CSVs → One NetCDF per (clim_factor, gwl)
###############################################################################
# Helper function to collect CSVs safely
def collect_GWL_csvs(wc):
    folder = Path(GWL_RAW_DIR) / wc.clim_factor / wc.gwl
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder.resolve()}")
    return csv_files


rule GWL_convert_and_merge_across_years:
    input:
        lambda wc: collect_GWL_csvs(wc)
    output:
        GWL_NETCDF_DIR / "{clim_factor}" / "{gwl}.nc"
    threads: 10
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_convert_and_merge_across_years.py"

###############################################################################
# Rule: Combine all GWL NetCDFs for each climate factor
###############################################################################

rule GWL_merge_across_models:
    input:
        lambda wc: expand(
            GWL_NETCDF_DIR / wc.clim_factor / "{gwl}.nc",
            gwl=gwl_scenarios
        )
    output:
        GWL_NETCDF_DIR / "combined_{clim_factor}.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_merge_across_models.py"

###############################################################################
# Rule: Merge all climate factors into a single gwl_all.nc
###############################################################################

rule GWL_merge_across_clim_factors:
    input:
        expand(GWL_NETCDF_DIR / "combined_{clim_factor}.nc", clim_factor=ar6_clim_factors)
    output:
        GWL_NETCDF_DIR / "gwl_all.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_merge_across_clim_factors.py"


###############################################################################
# Rule: Slice GWL all.nc depending on provided configuration
###############################################################################

rule GWL_get_slice_netcdf:
    input:
        GWL_NETCDF_DIR / "gwl_all.nc"
    output:
        GWL_NETCDF_DIR / "gwl_all_sliced.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "get_slice_netcdf.py"


###############################################################################
# Rule: Convert sliced NetCDF to TSV
###############################################################################
rule GWL_netcdf_slice_to_tsv:
    input:
        GWL_NETCDF_DIR / "gwl_all_sliced.nc"
    output:
        GWL_NETCDF_DIR / "gwl_all_sliced.tsv"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "netcdf_slice_to_tsv.py"

