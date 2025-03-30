"""Main module to read OpSim output database."""

import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
    import shapely.geometry as shp_geo
    import shapely.affinity as shp_aff

    use_geopandas = True
except ImportError:
    use_geopandas = False

import healpy as hp
import numpy as np
import sqlalchemy as sqla

from sklearn.neighbors import BallTree
from astropy.time import Time
from . import utils as ut
from .new_uti import def_host_joiner


class OpSimSurvey:
    """
    A class to manipulate OpSim db data and turn them into simulation inputs.

    Attributes
    ----------
    db_path : pathlib.Path
        Path to the database file
    sql_engine : sqlalchemy.engine.base.Engine
        The sqlalchemy engine link to the db.
    opsimdf : pandas.DataFrame
        Dataframe of the simulated observations.
    tree : sklearn.neighbors.BallTree
        A BallTree used to select observations on the healpy representation of the survey.
    host : pandas.DataFrame
        Dataframe of the hosts.
    hp_rep : pandas.DataFrame
        The healpy representation oif the survey
    survey : pandas.DataFrame
        The healpy representation oif the survey
    survey_hosts : pandas.DataFrame
        The hosts that are inside the survey.
    """

    __DEBASS_FIELD_RADIUS__ = np.radians(1.1)  # DEBASS Field Radius in radians
    __DEBASS_pixelSize__ = 0.263  # DECAM Pixel size in arcsec^-1

    def __init__(
        self,
        db_path,
        table_name="observations",
        MJDrange=None,
        host_file=None,
        host_config={},
    ):
        """Construct an OpSimSurvey object from an OpSim DataBase.

        Parameters
        ----------
        db_path : str
            The path to the Opsim db file
        table_name : str, optional
            Name of the observations table in the OpSIm db file, by default "observations"
        MJDrange : (int, int) or (str,str), optional
            Min and Max date to query if float assumed to be mjd, by default None
        host_file : str, optional
            Path to a parquet file containg hosts, by default None
        host_config : dict, optional
            Configuration for reading host file, by default {}
        """
        self.db_path = Path(db_path)
        self.sql_engine = self._get_sql_engine(db_path)
        self.opsimdf = self._get_df_from_sql(self.sql_engine, MJDrange=MJDrange)
        self.opsimdf.attrs["OpSimFile"] = self.db_path.name

        self.tree = BallTree(
            self.opsimdf[["_dec", "_ra"]].values, leaf_size=50, metric="haversine"
        )

        self.host = self._read_host_file(host_file, **host_config)

        # Instead of a healpy representation, we build a field representation
        self._field_rep = None  # DataFrame of unique DEBASS field centers
        self._survey = None


    @staticmethod
    def _get_sql_engine(dbname):
        """Read a sql db Opsim output file.

        Parameters
        ----------
        dbname : str
            Path to sql file.

        Returns
        -------
        sqlalchemy.engine.base.Engine
            The sqlalchemy engine link to the db.
        """
        if not Path(dbname).exists():
            raise ValueError(f"{dbname} does not exists.")
        # Prepend the abs path with sqlite for use with sqlalchemy
        if not dbname.startswith("sqlite"):
            dbname = "sqlite:///" + dbname
        print("Reading from database {}".format(dbname))
        engine = sqla.create_engine(dbname, echo=False)
        return engine

    @staticmethod
    def _get_df_from_sql(sql_engine, MJDrange=None):
        tstart = time.time()
        query = """
        SELECT 
          o.id AS observationId, 
          o.LIBID, 
          o.MJD, 
          o.IDEXPT, 
          o.BAND, 
          o.GAIN, 
          o.RDNOISE, 
          o.SKYSIG, 
          o.PSF1, 
          o.PSF2, 
          o.PSFRAT, 
          o.ZP, 
          o.ZPERR, 
          o.MAG,
          h.RA AS fieldRA,
          h.DEC AS fieldDec,
          h.REDSHIFT AS REDSHIFT,
          h.PEAKMJD AS PEAKMJD,
          h.TEMPLATE_ZPT AS TEMPLATE_ZPT,
          h.TEMPLATE_SKYSIG AS TEMPLATE_SKYSIG
        FROM observations o
        JOIN lib_header h ON o.LIBID = h.LIBID
        """
        if MJDrange is not None:
            if isinstance(MJDrange, str):
                time_format = None
            else:
                time_format = "mjd"
            MJDrange = Time(MJDrange, format=time_format)
            query += f" WHERE o.MJD > {MJDrange.mjd[0]} AND o.MJD < {MJDrange[1].mjd}"

        df = pd.read_sql(query, con=sql_engine)
        df["_ra"] = np.radians(df.fieldRA.astype(float))
        df["_dec"] = np.radians(df.fieldDec.astype(float))
        df.set_index("observationId", inplace=True)
        print(f"Read N = {len(df)} observations in {time.time() - tstart:.2f} seconds.")
        return df.sort_values(by="MJD")

    def _read_host_file(
        self,
        host_file,
        col_ra="ra",
        col_dec="dec",
        ra_dec_unit="radians",
        wgt_map=None,
        add_SNMAGSHIFT=False,
    ):
        """Read a parquet file containing hosts.

        Parameters
        ----------
        host_file : str
            Path to the parquet file
        col_ra : str, optional
            Key of column containing RA, by default 'ra'
        col_dec : str, optional
            Key of column containing Dec, by default 'dec'
        ra_dec_unit : str, optional
            Unit of ra_dec (radians or degrees), by default 'radians'

        Returns
        -------
        pandas.DataFrame
            Dataframe of the hosts.
        """
        if host_file is None:
            print("No host file.")
            return None

        print("Reading host from {}".format(host_file))
        hostdf = pd.read_parquet(host_file)

        if wgt_map is not None:
            print(f"Reading and applying HOST WGT MAP from {wgt_map}")
            var_names, wgt_map = ut.read_SNANA_WGTMAP(wgt_map)
            if len(var_names) > 1:
                raise NotImplementedError("HOST RESAMPLING ONLY WORK FOR 1 VARIABLES")
            keep_index = ut.host_resampler(
                wgt_map[var_names[0]],
                wgt_map["WGT"],
                hostdf.index.values,
                hostdf[var_names[0]].values,
            )

            hostdf = hostdf.loc[keep_index]

            if add_SNMAGSHIFT and "SNMAGSHIFT" in wgt_map:
                snmagshift = np.zeros(len(hostdf))
                for i in range(len(wgt_map["WGT"]) - 1):
                    mask = np.ones(len(hostdf), dtype=bool)
                    for v in var_names:
                        mask &= hostdf[v].between(wgt_map[v][i], wgt_map[v][i + 1])
                    snmagshift[mask] = wgt_map["SNMAGSHIFT"][i]
                hostdf["SNMAGSHIFT"] = snmagshift

        if ra_dec_unit == "degrees":
            hostdf[col_ra] += 360 * (hostdf[col_ra] < 0)
            hostdf.rename(columns={col_ra: "RA_GAL", col_dec: "DEC_GAL"}, inplace=True)
            hostdf["ra"] = np.radians(hostdf["RA_GAL"])
            hostdf["dec"] = np.radians(hostdf["DEC_GAL"])
        elif ra_dec_unit == "radians":
            hostdf[col_ra] += 2 * np.pi * (hostdf[col_ra] < 0)
            hostdf.rename(columns={col_ra: "ra", col_dec: "dec"}, inplace=True)
            hostdf["RA_GAL"] = np.degrees(hostdf["ra"])
            hostdf["DEC_GAL"] = np.degrees(hostdf["dec"])
        hostdf.attrs["file"] = host_file
        return hostdf
    def compute_field_rep(self, minVisits=1):
        """
        Compute a field representation for DEBASS by grouping observations
        by their field centers (fieldRA and fieldDec) and counting the number
        of visits per field.
        """
        field_rep = self.opsimdf.groupby(["fieldRA", "fieldDec"]).size().reset_index(name="n_visits")
        # Optionally, filter out fields with very few observations
        field_rep = field_rep[field_rep["n_visits"] >= minVisits]
        # Add radians columns for spatial queries
        field_rep["_ra"] = np.radians(field_rep["fieldRA"].astype(float))
        field_rep["_dec"] = np.radians(field_rep["fieldDec"].astype(float))
        self._field_rep = field_rep
        print(f"Computed {len(self._field_rep)} unique fields from DEBASS observations.")

    def sample_survey(self, N_fields, random_seed=None, nworkers=10):
        """Sample Nfields inside the survey's healpy representation.

        Parameters
        ----------
        N_fields : int
            Number of fields to sample
        random_seed : int or numpy.random.SeedSquence, optional
            The random  seed used to sample the fields, by default None
        nworkers : int, optional
            Number of cores used to run multiprocessing on host matching, by default 10

        Notes
        ------
        Random seed only apply on field sampling.
        """
        # Define the base columns you always need
        header_columns = ['LIBID', 'fieldRA', 'fieldDec', 'REDSHIFT', 'PEAKMJD', 'TEMPLATE_ZPT', 'TEMPLATE_SKYSIG']
        unique_fields = self.opsimdf[header_columns].drop_duplicates().reset_index(drop=True)

        if N_fields > len(unique_fields):
            print(f"N_fields ({N_fields}) > available fields ({len(unique_fields)}); sampling all fields.")
            N_fields = len(unique_fields)
        rng = np.random.default_rng(random_seed)
        self._survey = unique_fields.sample(n=N_fields, replace=False, random_state=rng)

        if self.host is not None:
            print("Compute survey hosts")
            self._survey_hosts = self.get_survey_hosts(nworkers=nworkers)


    def get_obs_from_coords(self, ra, dec, is_deg=True, formatobs=False, keep_keys=[]):
        """Get observations at ra, dec coordinates.

        Parameters
        ----------
        ra : numpy.ndarray(float)
            RA coordinate
        dec : numpy.ndarray(float)
            Dec coordinate
        is_deg : bool, optional
            is RA, Dec given in degrees, by default True
        formatobs : bool, optional
            format obs for simulation, by default False
        keep_keys : list(str)
            List of keys to keep in addition to formatted quantities

        Yields
        ------
        pandas.DatFrame
            Dataframes of observations.
        """
        if is_deg:
            ra = np.radians(np.array(ra, dtype=float))
            dec = np.radians(np.array(dec, dtype=float))

        obs_idx = self.tree.query_radius(
            np.array([dec, ra]).T,
            r=self.__DEBASS_FIELD_RADIUS__,
            count_only=False,
            return_distance=False,
        )
        for idx in obs_idx:
            if formatobs:
                yield self.formatObs(self.opsimdf.iloc[idx], keep_keys=keep_keys)
            else:
                yield self.opsimdf.iloc[idx]

    def get_survey_obs(self, formatobs=True, keep_keys=[]):
        """Get survey observations.

        Parameters
        ----------
        formatobs : bool, optional
            Format the observation to get only quantities of interest for simulation, by default True
        keep_keys : list(str)
            List of keys to keep in addition to formatted quantities

        Yields
        ------
        pandas.DatFrame
            Dataframes of observations.
        """
        return self.get_obs_from_coords(
            *self.survey[["fieldRA", "fieldDec"]].values.T,
            is_deg=True,
            formatobs=formatobs,
            keep_keys=keep_keys,
        )

    def get_libid_field(self):
        # Return a tuple (LIBID, (fieldRA, fieldDec)) for each survey field.
        return zip(self.survey["LIBID"].values, self.survey[["fieldRA", "fieldDec"]].values)

    def get_survey_hosts(self, nworkers=10):
        """Get survey hosts.

        Parameters
        ----------
        nworkers : int, optional
            Number of cores used to run multiprocessing, by default 10

        Returns
        -------
        pandas.DataFrame
            Dataframe of host inside the survey, matched to their field indicated by GROUPID.
        """
        if not use_geopandas:
            raise ModuleNotFoundError("Install geopandas library to use host matching.")

        if self.host is None:
            raise ValueError("No host file set.")

        _RA = pd.to_numeric(self.survey.fieldRA, errors="coerce")
        _Dec = pd.to_numeric(self.survey.fieldDec, errors="coerce")
        libid = pd.to_numeric(self.survey.LIBID, errors="coerce")
        _RA_adjusted = _RA - 180
        rad_ra = np.radians(_RA_adjusted)
        rad_dec = np.radians(_Dec)

        survey_fields = gpd.GeoDataFrame(
            self.survey.copy(),
            geometry=gpd.points_from_xy(
                rad_ra, rad_dec
            ).buffer(self.__DEBASS_FIELD_RADIUS__)
        )

        # Use the same host joiner and subdivide methods as before.
        host_joiner_func = partial(def_host_joiner, survey_fields)

        nsub = 5 if nworkers == 1 else nworkers
        sdfs = ut.df_subdiviser(self.host, Nsub=nsub)

        # Process sequentially
        res = []
        for sdf in sdfs:
            res.append(host_joiner_func(sdf))

        survey_host = pd.concat(res)
        return survey_host

    def formatObs(self, OpSimObs, keep_keys=[]):
        """Format DEBASS SIMLIB observation for simulation.

        For DEBASS SIMLIB, the necessary quantities are provided directly:
        - 'MJD' for the exposure time,
        - 'PSF1' as the PSF,
        - 'ZP' as the zero point,
        - 'SKYSIG' as the sky noise,
        - 'BAND' as the filter.

        Additional keys in keep_keys will be preserved.
        """
        formatobs_df = pd.DataFrame(
            {
                "expMJD": OpSimObs["MJD"],
                "PSF": OpSimObs["PSF1"],
                "ZPT": OpSimObs["ZP"],
                "SKYSIG": OpSimObs["SKYSIG"],
                "BAND": OpSimObs["BAND"],
                "RDNOISE": OpSimObs["RDNOISE"],
                "GAIN": OpSimObs["GAIN"],
                "IDEXPT": OpSimObs["IDEXPT"],  # ID of the exposure
                **{k: OpSimObs[k] for k in keep_keys},
            }
        ).reset_index(names="ObsID")
        return formatobs_df

    @property
    def survey(self):
        if self._survey is None:
            raise ValueError("survey not set. Please run sample_survey() before.")
        return self._survey

    @property
    def survey_hosts(self):
        if self.host is None:
            return None
        return self._survey_hosts