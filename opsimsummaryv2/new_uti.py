try:
    import geopandas as gpd
    import shapely.geometry as shp_geo
    import shapely.ops as shp_ops

    use_geopandas = True
except ImportError:
    use_geopandas = False



def def_host_joiner(survey_fields, host):
    """Use geopandas to match hosts in survey fields and assign GROUPID from LIBID.

    Parameters
    ----------
    survey_fields : geopandas.GeoDataFrame
        Geodataframe describing survey fields. It must include a 'LIBID' column.
    host : pandas.DataFrame
        Dataframe that contains host information with columns 'ra' and 'dec' (in degrees).

    Returns
    -------
    pandas.DataFrame
        Dataframe containing hosts that fall inside a survey field, with a new column 'GROUPID'
        set to the corresponding LIBID from survey_fields.
    """
    from geopandas import GeoDataFrame, points_from_xy

    if not use_geopandas:
        raise ModuleNotFoundError("Install geopandas library to use host_joiner.")

    host_pos = GeoDataFrame(
        host.copy(),
        geometry=gpd.points_from_xy(host.ra.values, host.dec.values))

    joined = host_pos.sjoin(survey_fields[['LIBID', 'geometry']], how="inner", predicate="intersects")

    grouped = joined.groupby(joined.index).first()
    grouped = grouped.rename(columns={'LIBID': 'GROUPID'})

    result = host.loc[grouped.index].copy()
    result['GROUPID'] = grouped['GROUPID']
    return result
