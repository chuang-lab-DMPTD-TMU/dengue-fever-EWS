###############################################################################
# Rule: Merge CSVs → One NetCDF per (clim_factor, TREAD)
###############################################################################
# Helper function to collect CSVs safely
def collect_TREAD_csvs(wc):
    folder = Path(TREAD_RAW_DIR) / wc.clim_factor 
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder.resolve()}")
    return csv_files


rule TREAD_convert_and_merge_across_years:
    input:
        lambda wc: collect_TREAD_csvs(wc)
    output:
        TREAD_NETCDF_DIR / "combined_{clim_factor}.nc"
    threads: 12
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "TREAD_convert_and_merge_across_years.py"


###############################################################################
# Rule: Merge all climate factors into a single TREAD_all.nc
###############################################################################

rule TREAD_merge_across_clim_factors:
    input:
        expand(TREAD_NETCDF_DIR / "combined_{clim_factor}.nc", clim_factor=tread_clim_factors)
    output:
        TREAD_NETCDF_DIR / "TREAD_all.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "merge_across_clim_factors.py"


##############################################################################
# Rule: Slice TREAD all.nc depending on provided configuration
##############################################################################

rule TREAD_get_slice_netcdf:
    input:
        TREAD_NETCDF_DIR / "TREAD_all.nc"
    output:
        TREAD_NETCDF_DIR / "TREAD_all_sliced.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "get_slice_netcdf.py"


###############################################################################
# Rule: Convert sliced NetCDF to TSV
###############################################################################
rule TREAD_netcdf_slice_to_tsv:
    input:
        TREAD_NETCDF_DIR / "TREAD_all_sliced.nc"
    output:
        TREAD_NETCDF_DIR / "TREAD_all_sliced.tsv"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "netcdf_slice_to_tsv.py"

