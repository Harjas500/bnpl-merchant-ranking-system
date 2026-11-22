import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import matplotlib.ticker as mticker
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import contextily as ctx
import os
import sys

from pyspark.sql import functions as F
from pyspark.sql.functions import trim
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

def start_spark(
    app_name: str = "ADSA_Assignment2",
    master: str = "local[*]",
    driver_memory: str = "4g",
    executor_memory: str = "2g",
    shuffle_partitions: int = 200,
    enable_aqe: bool = True
):
    """
    Create and return a configured SparkSession.

    Example:
        spark = start_spark()
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master(master).appName(app_name)

    # Performance / robustness defaults similar to the notebook
    if enable_aqe:
        builder = builder.config("spark.sql.adaptive.enabled", "true") \
                         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    builder = builder \
        .config("spark.sql.parquet.filterPushdown", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.files.ignoreCorruptFiles", "true") \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.sql.repl.eagerEval.enabled", "true") \
        .config("spark.sql.parquet.cacheMetadata", "true") \
        .config("spark.sql.session.timeZone", "Etc/UTC") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory)

    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.caseSensitive", "false")
    return spark

spark = start_spark()

def plot_histogram(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Creates a histogram from a DataFrame using the specified x and y columns.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
    
    Returns:
        matplotlib.axes.Axes: The histogram plot
    """
    fig, ax = plt.subplots()
    ax.hist(df[x_col], weights=df[y_col], bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel(x_col)
    ax.set_ylabel("Frequency (weighted by " + y_col + ")")
    ax.set_title(f"Histogram of {x_col} weighted by {y_col}")
    plt.show()
    return ax


def plot_line(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Creates a line graph from a DataFrame using the specified x and y columns.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
    
    Returns:
        matplotlib.axes.Axes: The line plot
    """
    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col], marker='o', linestyle='-')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Line graph of {y_col} vs {x_col}")
    plt.show()
    return ax


def _find_lat_lon_cols(df):
    lat_candidates = ['lat_precise', 'latprecise', 'lat', 'latitude', 'y']
    lon_candidates = ['long_precise', 'longprecise', 'lon', 'long', 'lng', 'longitude', 'x']
    lat_col = next((c for c in df.columns if c.lower() in lat_candidates), None)
    lon_col = next((c for c in df.columns if c.lower() in lon_candidates), None)
    return lat_col, lon_col

def _find_postcode_col(df):
    candidates = ['postcode', 'post_code', 'postcodes', 'pcd', 'zip']
    cols = [c.lower() for c in df.columns.to_list()]
    for cand in candidates:
        if cand in cols:
            # return original-cased column name
            return df.columns.to_list()[cols.index(cand)]
    return None

def get_postcode_list_from_csv(csv_path: str, keep_unique: bool = True, zfill_width: int = 4) -> list:
    """
    Read CSV (outside the plotting function), find the postcode column using
    .columns.to_list(), and return a list of postcode strings suitable to pass
    into plot_australian_postcodes(..., postcode_filter=...).

    - preserves leading zeros by zfilling to zfill_width
    - returns unique list by default
    """
    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns.to_list()  # user-requested use of .columns.to_list()
    postcode_col = _find_postcode_col(df)
    if postcode_col is None:
        raise ValueError(f"No postcode column found in CSV columns: {cols}")
    pcs = df[postcode_col].dropna().astype(str).astype(str).str.zfill(zfill_width)
    if keep_unique:
        return pcs.unique().tolist()
    return pcs.tolist()

def plot_points_map(
    csv_path,
    lat_col=None,
    lon_col=None,
    state_col=None,
    filter_state=None,
    postcode_col=None,
    postcode_filter=None,
    figsize=(10, 10),
    cmap='tab20',
    point_size=10,
    alpha=0.5,
    basemap=True,
    save_path=None,
    left_trim_fraction=-0.1,
    color: str = None,
    state_colors: dict = None,
    show_labels: bool = False,      # NEW
    label_col: str = None           # NEW: column to use for point labels
):
    """
    Static map plot of points from a CSV or DataFrame using GeoPandas + Contextily.
    Accepts either a path (str) or a pandas.DataFrame for csv_path.
    ...
    """
    # accept either DataFrame or path
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
    else:
        df = pd.read_csv(csv_path)

    # auto-detect postcode column if not provided
    if postcode_col is None:
        postcode_col = _find_postcode_col(df)

    # If postcode_filter given, filter early (supports ints or strings)
    if postcode_filter is not None and postcode_col is not None:
        pf_set = set(str(p).zfill(0) if isinstance(p, str) and p.isdigit() and p.startswith('0') else str(p) for p in postcode_filter)
        df = df[df[postcode_col].astype(str).isin(pf_set)].copy()


    if lat_col is None or lon_col is None:
        auto_lat, auto_lon = _find_lat_lon_cols(df)
        if lat_col is None:
            lat_col = auto_lat
        if lon_col is None:
            lon_col = auto_lon

    if lat_col is None or lon_col is None:
        raise ValueError("Could not determine latitude/longitude columns. "
                         "Pass lat_col and lon_col explicitly.")

    # Optionally filter by state
    if filter_state is not None and state_col is not None:
        if isinstance(filter_state, (list, tuple, set)):
            df = df[df[state_col].isin(filter_state)].copy()
        else:
            df = df[df[state_col] == filter_state].copy()

    

    # Drop invalid rows
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    # Convert to numeric, coercing errors
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])

    # Create GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs="EPSG:4326")

    # Reproject to web mercator for basemap overlay
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if state_col and state_col in gdf_web.columns:
        # preserve reference for labeling
        gdf_plot = gdf_web.copy()
        if state_colors:
            # map each state's colour; ensure keys compare as strings
            gdf_web['_plot_color'] = gdf_web[state_col].astype(str).map(lambda s: state_colors.get(s, None))
            mapped = gdf_web[~gdf_web['_plot_color'].isna()].copy()
            unmapped = gdf_web[gdf_web['_plot_color'].isna()].copy()
            # plot mapped rows using their explicit colours
            if not mapped.empty:
                mapped.plot(ax=ax, color=list(mapped['_plot_color'].values), markersize=point_size, alpha=alpha)
            # plot unmapped rows by categorical state color (fallback) without adding its own legend
            if not unmapped.empty:
                unmapped.plot(ax=ax, column=state_col, categorical=True, legend=False,
                              markersize=point_size, alpha=alpha, cmap=cmap)
            # for labeling use the union of both
            gdf_plot = pd.concat([mapped, unmapped]) if (not mapped.empty and not unmapped.empty) else (mapped if not mapped.empty else unmapped)

            # Build a legend from state_colors for states present in the plotted data
            present = set(gdf_web[state_col].astype(str).unique())
            legend_patches = []
            for st, col in state_colors.items():
                if str(st) in present:
                    legend_patches.append(mpatches.Patch(color=col, label=str(st)))
            if legend_patches:
                ax.legend(handles=legend_patches, title=str(state_col))
        elif color:
            gdf_plot = gdf_web
            gdf_web.plot(ax=ax, color=color, markersize=point_size, alpha=alpha)
        else:
            gdf_plot = gdf_web
            gdf_web.plot(ax=ax, column=state_col, categorical=True, legend=True,
                         markersize=point_size, alpha=alpha, cmap=cmap)
            # ensure legend has a title
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, title=str(state_col))

    else:
        gdf_plot = gdf_web
        plot_color = color if color else 'red'
        gdf_web.plot(ax=ax, color=plot_color, markersize=point_size, alpha=alpha)

    if show_labels and label_col and (label_col in gdf_plot.columns):
        for _, row in gdf_plot.iterrows():
            x, y = row.geometry.x, row.geometry.y
            txt = str(row[label_col])
            ax.text(x, y, txt, fontsize=8, ha='left', va='bottom', color='black', alpha=0.9)

    # Focus the view on the data: set x/y limits from the data bounds with small padding
    minx, miny, maxx, maxy = gdf_web.total_bounds
    if not any(pd.isna([minx, miny, maxx, maxy])):
        # if bounds are tiny (single point) add a small absolute pad as well
        xpad = max((maxx - minx) * 0.05, 1000)
        ypad = max((maxy - miny) * 0.05, 1000)
        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)

    minx, miny, maxx, maxy = gdf_web.total_bounds
    if not any(pd.isna([minx, miny, maxx, maxy])):
        width = maxx - minx
        xpad = max(width * 0.05, 1000)
        ypad = max((maxy - miny) * 0.05, 1000)
        ax.set_xlim(minx - xpad, maxx + xpad)
        # move left limit inward by a fraction to "cut off" extra left space
        left_limit = minx + width * float(left_trim_fraction)
        # ensure left_limit is not greater than right
        left_limit = min(left_limit, maxx - 1)
        ax.set_xlim(left_limit - xpad*0.2, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)
    
    # Add basemap tiles (pass the plotted CRS so tiles align with limits)
    if basemap:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=gdf_web.crs.to_string())
        except Exception:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_web.crs.to_string())

    # hide axis after basemap added
    ax.set_axis_off()

    # Optionally save
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return gdf

def plot_australian_postcodes(
    csv_path="./data/raw/australian_postcodes.csv",
    state_filter=None,
    figsize=(10, 10),
    point_size=15,
    alpha=0.5,
    basemap=True,
    save_path=None,
    use_state_centroids: bool = False,
    state_centroid_postcodes: dict = None,
    left_trim_fraction: float = 0.7,
    color: str = None,
    state_colors: dict = None,
    cmap: str = 'tab20',
    postcode_filter: list = None       
):
    """
    Convenience wrapper to plot australian_postcodes.csv as a static map.

    If use_state_centroids=True, plots one point per state. You can optionally
    pass state_centroid_postcodes mapping like {'VIC':3000} to pick a specific
    postcode row as the centroid for that state. If a mapping entry is missing
    or the postcode row cannot be found, the state's median lat/lon is used.
    """
    df = pd.read_csv(csv_path)
    lat_col, lon_col = _find_lat_lon_cols(df)
    if lat_col is None or lon_col is None:
        raise ValueError("Could not detect latitude/longitude columns in the CSV. "
                         "Pass explicit lat_col/lon_col or verify column names.")

    # if not asking for centroids, delegate to existing function
    if not use_state_centroids:
        return plot_points_map(
            csv_path=csv_path,
            lat_col=lat_col,
            lon_col=lon_col,
            state_col='state',
            filter_state=state_filter,
            postcode_col=_find_postcode_col(df),
            postcode_filter=postcode_filter,   # pass through
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            basemap=basemap,
            save_path=save_path,
            left_trim_fraction=left_trim_fraction,
            color=color,
            state_colors=state_colors,
            cmap=cmap
        )

    # Build per-state centroid dataframe
    postcode_col = _find_postcode_col(df)
    states = state_filter if state_filter is not None else sorted(df['state'].dropna().unique())
    centroids = []
    for st in states:
        sub = df[df['state'] == st].copy()
        if sub.empty:
            continue
        chosen_row = None
        if state_centroid_postcodes and st in state_centroid_postcodes and postcode_col:
            target_pc = str(state_centroid_postcodes[st])
            match = sub[sub[postcode_col].astype(str) == target_pc]
            if not match.empty:
                chosen_row = match.iloc[[0]]
        if chosen_row is None:
            # fallback to median coords for the state
            lat_m = pd.to_numeric(sub[lat_col], errors='coerce').median()
            lon_m = pd.to_numeric(sub[lon_col], errors='coerce').median()
            chosen_row = pd.DataFrame([{lat_col: lat_m, lon_col: lon_m, 'state': st, postcode_col: None}])
        centroids.append(chosen_row.assign(state=st))

    
    if not centroids:
        raise RuntimeError("No states available to plot centroids.")

    cent_df = pd.concat(centroids, ignore_index=True)

    # If postcode_filter provided, keep only centroid rows that have a matching postcode
    if postcode_filter is not None and postcode_col is not None:
        pf_set = set(str(p) for p in postcode_filter)
        # keep rows where postcode matches the filter; note rows with postcode_col == None will be dropped
        cent_df = cent_df[cent_df[postcode_col].astype(str).isin(pf_set)].copy()
        if cent_df.empty:
            raise RuntimeError("No centroids match the provided postcode_filter.")

    # delegate to plot_points_map with DataFrame input
    return plot_points_map(
        csv_path=cent_df,
        lat_col=lat_col,
        lon_col=lon_col,
        state_col='state',
        filter_state=None,
        postcode_col=postcode_col,
        postcode_filter=None,   # already filtered
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        basemap=basemap,
        save_path=save_path,
        left_trim_fraction=left_trim_fraction,
        color=color,
        state_colors=state_colors,
        cmap=cmap
    )

def plot_postcode_density(
    counts: dict,
    postcodes_csv: str = "./data/raw/australian_postcodes.csv",
    postcode_col: str = None,
    zfill_width: int = 4,
    cmap: str = "OrRd",
    figsize: tuple = (10, 10),
    basemap: bool = True,
    save_path: str = None,
    marker_scale: float = 1.0,
    alpha: float = 0.5,
    log_scale: bool = False,
    left_trim_fraction: float = 0.7,
    point_size: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm = None,
    use_quantiles: bool = False,
    q_bins: int = 5,
):

    counts_norm = {str(k).zfill(zfill_width): float(v) for k, v in counts.items()}

    df = pd.read_csv(postcodes_csv)
    if postcode_col is None:
        postcode_col = _find_postcode_col(df)
    if postcode_col is None:
        raise ValueError("Could not find postcode column in postcodes CSV.")

    # normalize postcode column in CSV
    df[postcode_col] = df[postcode_col].astype(str).str.zfill(zfill_width)

    # build counts dataframe and merge
    df_counts = pd.DataFrame([{postcode_col: k, "count": v} for k, v in counts_norm.items()])
    merged = df.merge(df_counts, on=postcode_col, how="inner")
    if merged.empty:
        raise RuntimeError("No matching postcodes found between counts dict and postcodes CSV.")

    # detect lat/lon columns
    lat_col, lon_col = _find_lat_lon_cols(merged)
    if lat_col is None or lon_col is None:
        raise ValueError("Could not detect latitude/longitude columns in postcodes CSV.")

    # ensure numeric coords
    merged[lat_col] = pd.to_numeric(merged[lat_col], errors="coerce")
    merged[lon_col] = pd.to_numeric(merged[lon_col], errors="coerce")
    merged = merged.dropna(subset=[lat_col, lon_col, "count"]).copy()

    # build GeoDataFrame and project for plotting
    gdf = gpd.GeoDataFrame(merged, geometry=[Point(xy) for xy in zip(merged[lon_col], merged[lat_col])], crs="EPSG:4326")
    gdf_web = gdf.to_crs(epsg=3857)

    # compute marker sizes (sqrt scaling) and color normalization
    counts_arr = gdf_web["count"].values.astype(float)
    if point_size is None:
        # scaled size based on counts (sensible defaults)
        marker_sizes = (np.sqrt(counts_arr) * marker_scale * 10).clip(min=5)
    else:
        # fixed point size (user-supplied)
        marker_sizes = np.full_like(counts_arr, float(point_size))
    
    if norm is None:
        if use_quantiles:
            # create boundary bins from quantiles to emphasize lower counts
            try:
                bins = np.quantile(counts_arr, np.linspace(0.0, 1.0, q_bins + 1))
                bins = np.unique(bins)
                if len(bins) > 1:
                    norm = mcolors.BoundaryNorm(bins, ncolors=plt.cm.get_cmap(cmap).N, clip=True)
                else:
                    norm = mcolors.Normalize(vmin=counts_arr.min(), vmax=counts_arr.max())
            except Exception:
                norm = mcolors.Normalize(vmin=counts_arr.min(), vmax=counts_arr.max())
        elif log_scale:
            norm = mcolors.LogNorm(vmin=max(counts_arr.min(), 1e-6), vmax=counts_arr.max())
        else:
            norm = mcolors.Normalize(vmin=(vmin if vmin is not None else counts_arr.min()),
                                    vmax=(vmax if vmax is not None else counts_arr.max()))


    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf_web.plot(
        ax=ax,
        column="count",
        cmap=cmap,
        markersize=marker_sizes,
        alpha=alpha,
        legend=True,
        norm=norm
    )

    sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    sm.set_array(counts_arr)
    # remove automatically created colorbars (if any) and add our own
    try:
        # add colorbar with integer formatter (change format string if not integer)
        cbar = fig.colorbar(sm, ax=ax, format=mticker.FormatStrFormatter("%d"))
    except Exception:
        # fallback: use scalar formatter without scientific notation
        sf = mticker.ScalarFormatter(useMathText=False)
        sf.set_scientific(False)
        cbar = fig.colorbar(sm, ax=ax, format=sf)

    # add basemap
    if basemap:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=gdf_web.crs.to_string())
        except Exception:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_web.crs.to_string())

    ax.set_axis_off()
    minx, miny, maxx, maxy = gdf_web.total_bounds
    if not any(pd.isna([minx, miny, maxx, maxy])):
        width = maxx - minx
        xpad = max(width * 0.05, 1000)
        ypad = max((maxy - miny) * 0.05, 1000)
        left_limit = minx + width * float(left_trim_fraction)
        left_limit = min(left_limit, maxx - 1)
        ax.set_xlim(left_limit - xpad * 0.2, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return gdf_web

def spark_count_customers_by_postcode(
    sdf,
    postcode_col: str = "postcode",
    user_col: str = "user_id",
    zfill_width: int = 4
):
    """
    Count distinct customers per postcode from a Spark DataFrame.

    Returns a Spark DataFrame with columns: postcode (zfilled string), unique_customers, tx_count.
    """
    from pyspark.sql import functions as F

    # normalise postcode to string and zfill
    sdf2 = (
        sdf.withColumn(postcode_col, F.trim(F.col(postcode_col).cast("string")))
           .withColumn("postcode4", F.when(F.col(postcode_col).isNotNull(),
                                           F.lpad(F.col(postcode_col), zfill_width, "0"))
                                  .otherwise(None))
    )

    result = (
        sdf2.filter(F.col("postcode4").isNotNull())
            .groupBy("postcode4")
            .agg(
                F.countDistinct(F.col(user_col)).alias("unique_customers"),
                F.count(F.lit(1)).alias("tx_count")
            )
            .withColumnRenamed("postcode4", "postcode")
    )
    return result

def _normalize_pc_series(s: pd.Series, zfill_width: int = 4) -> pd.Series:
    # keep digits only, strip whitespace, zfill
    s2 = s.astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    s2 = s2.replace({"": None}).dropna()
    return s2.str.zfill(zfill_width)


def plot_merchant_density(
    merchant_csv: str = "./data/curated/merchant_feature_engineering.csv",
    postcode_col: str = "primary_postcode",
    merchant_col: str = "merchant_abn",
    zfill_width: int = 4,
    postcodes_csv: str = "./data/raw/australian_postcodes.csv",
    point_size: float | None = None,
    marker_scale: float = 1.0,
    cmap: str = "Blues",
    figsize: tuple = (10, 10),
    basemap: bool = True,
    save_path: str | None = None,
    alpha: float = 0.9,
    log_scale: bool = False,
    left_trim_fraction: float = 0.7
):
    """
    Count merchants per postcode from merchant_csv and plot density.
    Includes diagnostics to show postcode mismatches.
    """
    df = pd.read_csv(merchant_csv, low_memory=False)
    if postcode_col not in df.columns:
        pc = _find_postcode_col(df)
        if pc is None:
            raise ValueError(f"postcode column '{postcode_col}' not found and automatic detection failed. Columns: {df.columns.tolist()}")
        postcode_col = pc

    if merchant_col not in df.columns:
        raise ValueError(f"merchant column '{merchant_col}' not found. Columns: {df.columns.tolist()}")

    df = df.dropna(subset=[postcode_col, merchant_col]).copy()

    pc_num = pd.to_numeric(df[postcode_col], errors="coerce")
    df = df.loc[pc_num.notna()].copy()
    df[postcode_col] = pc_num.astype(int).astype(str).str.zfill(zfill_width)

    # count distinct merchants per postcode
    counts_df = df.groupby(postcode_col)[merchant_col].nunique().reset_index(name="count")
    counts_df = counts_df.dropna(subset=[postcode_col])
    counts_dict = dict(zip(counts_df[postcode_col].astype(str), counts_df["count"].astype(float)))

    # load postcodes CSV and normalise its postcode column
    pcs_df = pd.read_csv(postcodes_csv, low_memory=False)
    pc_col = _find_postcode_col(pcs_df)
    if pc_col is None:
        raise ValueError("Could not find postcode column in postcodes CSV.")

    # proceed to plot (plot_postcode_density will raise if no matches)
    return plot_postcode_density(
        counts=counts_dict,
        postcodes_csv=postcodes_csv,
        postcode_col=pc_col,
        zfill_width=zfill_width,
        cmap=cmap,
        figsize=figsize,
        basemap=basemap,
        save_path=save_path,
        marker_scale=marker_scale,
        alpha=alpha,
        log_scale=log_scale,
        left_trim_fraction=left_trim_fraction,
        point_size=point_size
    )

def plot_selected_merchants_from_rankings(
    rankings_csv: str,
    rankings_abn_col: str = "merchant_abn",
    merchant_csv: str = "./data/curated/merchant_overall.csv",
    merchant_abn_col: str = "merchant_abn",
    postcode_col: str = "primary_postcode",
    zfill_width: int = 4,
    postcodes_csv: str = "./data/raw/australian_postcodes.csv",
    point_size: float | None = None,
    marker_scale: float = 1.0,
    cmap: str = "OrRd",
    figsize: tuple = (10, 10),
    basemap: bool = True,
    save_path: str | None = None,
    alpha: float = 0.5,
    log_scale: bool = False,
    left_trim_fraction: float = -0.1
):
    """
    Plot density for merchants listed in rankings_csv. Returns GeoDataFrame plotted.
    """
    # load rankings ABNs (treat as strings)
    ranks = pd.read_csv(rankings_csv, usecols=[rankings_abn_col], dtype=str)
    selected_abns = set(ranks[rankings_abn_col].dropna().astype(str).str.strip().tolist())
    if not selected_abns:
        raise ValueError("No merchant ABNs found in rankings CSV")

    # load merchant table and filter
    mdf = pd.read_csv(merchant_csv, low_memory=False, dtype=str)
    if merchant_abn_col not in mdf.columns:
        raise ValueError(f"merchant column '{merchant_abn_col}' not found in {merchant_csv}")
    if postcode_col not in mdf.columns:
        raise ValueError(f"postcode column '{postcode_col}' not found in {merchant_csv}")

    mdf = mdf[mdf[merchant_abn_col].astype(str).str.strip().isin(selected_abns)].copy()
    if mdf.empty:
        raise RuntimeError("No matching merchants found in merchant CSV for provided rankings ABNs")

    # postcode formatting (keep as strings, pad to 4 for matching postcode file)
    mdf[postcode_col] = mdf[postcode_col].astype(str).str.strip().replace({"nan": None})
    tmp_pc = pd.to_numeric(mdf[postcode_col], errors="coerce")
    mdf = mdf.loc[tmp_pc.notna()].copy()
    mdf[postcode_col] = tmp_pc.astype(int).astype(str).str.zfill(zfill_width)

    # count merchants per postcode (distinct merchant_abn)
    counts_df = mdf.groupby(postcode_col)[merchant_abn_col].nunique().reset_index(name="count")
    counts_dict = dict(zip(counts_df[postcode_col].astype(str), counts_df["count"].astype(float)))

    # delegate to existing plotting function
    return plot_postcode_density(
        counts=counts_dict,
        postcodes_csv=postcodes_csv,
        postcode_col=None,
        zfill_width=zfill_width,
        cmap=cmap,
        figsize=figsize,
        basemap=basemap,
        save_path=save_path,
        marker_scale=marker_scale,
        alpha=alpha,
        log_scale=log_scale,
        left_trim_fraction=left_trim_fraction,
        point_size=point_size
    )


state_centroids = {
    'VIC': '3000', 'NSW': '2000', 'QLD': '4000', 'SA': '5000',
    'WA': '6000', 'TAS': '7000', 'NT': '0800', 'ACT': '2600'
}
state_colors = {
    'VIC': '#1f78b4', 'NSW': '#33a02c', 'QLD': '#e31a1c', 'WA': '#ff7f00',
    'SA': '#6a3d9a', 'TAS': '#b15928', 'ACT': '#a6cee3', 'NT': '#b2df8a'
}

csv_path = "./data/raw/australian_postcodes.csv"
cols = pd.read_csv(csv_path, nrows=0).columns.to_list()  # inspect column names quickly
postcode_filter = get_postcode_list_from_csv(csv_path)

tx_path = "./data/raw/transactions_cleaned_parquet"
tx_sdf = spark.read.parquet(tx_path)
 
consumer_csv = "./data/curated/consumer_curated_with_postcode.csv"
consumer_sdf = (
    spark.read.option("header", True).option("inferSchema", True).csv(consumer_csv)
    .withColumn("postcode", trim(F.col("postcode").cast("string")))
)

sdf_for_count = tx_sdf.join(broadcast(consumer_sdf.select("user_id", "postcode")), on="user_id", how="left")
sdf_for_count = sdf_for_count.filter(F.col("postcode").isNotNull() & F.col("user_id").isNotNull())
counts_sdf = spark_count_customers_by_postcode(sdf_for_count, postcode_col="postcode", user_col="user_id", zfill_width=4)

# 5) Collect to driver and build dict {postcode: unique_customers}
counts_pd = counts_sdf.toPandas()   # small-ish result; if huge write to file instead
counts_dict = dict(zip(counts_pd["postcode"].astype(str), counts_pd["unique_customers"].astype(float)))

gdf_web = plot_postcode_density(
    counts=counts_dict,
    postcodes_csv="./data/raw/australian_postcodes.csv",
    zfill_width=4,
    cmap="OrRd",
    figsize=(10, 10),
    basemap=True,
    save_path="./data/out/consumer_postcode_density.png",
    marker_scale=0.15,
    alpha=0.85,
    log_scale=False
)

gdf = plot_merchant_density(
    merchant_csv="./data/curated/merchant_overall.csv",
    postcode_col="primary_postcode",
    merchant_col="merchant_abn",
    zfill_width=4,
    postcodes_csv="./data/raw/australian_postcodes.csv",
    point_size=8,                     
    marker_scale=0.2,
    save_path="./data/out/merchant_density.png",
    log_scale=True,
    left_trim_fraction=-0.1,
)

gdf = plot_merchant_density(
    merchant_csv="./data/curated/top10_with_postcode.csv",
    postcode_col="primary_postcode",
    merchant_col="merchant_abn",
    zfill_width=4,
    postcodes_csv="./data/raw/australian_postcodes.csv",
    point_size=80,                     
    marker_scale=0.2,
    save_path="./data/out/merchant_density.png",
    log_scale=True,
    left_trim_fraction=-0.1,
)

gdf = plot_selected_merchants_from_rankings(
    rankings_csv="./data/out/rankings_top100_overall.csv",
    merchant_csv="./data/curated/merchant_overall.csv",
    save_path="./data/out/ranked_merchants_density.png",
    log_scale=False,
    point_size=8
)

# Call plotting with the externally created postcode list
plot_australian_postcodes(
    csv_path=csv_path,
    use_state_centroids=False,         # plot exact postcode rows (not one-per-state centroid)
    postcode_filter=postcode_filter,   # list built outside the function
    state_filter=["ACT", "NT", "VIC", "NSW", "WA", "QLD", "SA", "TAS"],
    state_colors=state_colors,
    cmap="tab20",
    figsize=(8, 8),
    point_size=8,
    alpha=0.9,
    basemap=True,
    save_path="./data/out/customer_locations.png"
)


spark.stop()