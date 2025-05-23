#!/usr/bin/env python
# coding: utf-8

# ## Preprocess Data for VAE

# The aim of this notebook is to translate NetCDF files (.nc)
# of a daily climate variable (e.g. maximum temperature)
# to a numpy 3D-array. This output array can be read for training
# and evaluating the Convolutional Variational AutoEncoder model.


# #### 0. Libraries

import os
from typing import Dict, List

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from itwinai.components import DataGetter, monitor_exec


def get_extrema(histo_dataset: np.ndarray) -> np.array:
    """
    Computes global extrema over past data.

    Parameters:
    histo_dataset: historical data
    """
    global_min = np.min(histo_dataset)
    global_max = np.max(histo_dataset)
    return np.array([global_min, global_max])


class PreprocessData(DataGetter):
    def __init__(
        self,
        dataset_root: str,
        input_path: str,
        output_path: str,
        histo_extr: List[str],
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        scenarios: List[int],
        scenario_extr: Dict[int, List[str]],
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.input_path = input_path
        self.output_path = output_path
        self.histo_extr = histo_extr
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.scenarios = scenarios
        self.scenario_extr = scenario_extr

        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

    def xr_to_ndarray(
        self, xr_dset: xr.Dataset, sq_coords: dict
    ) -> (np.ndarray, np.array, str):
        """
        Converts an xarray dataset it to a cropped square ndarray,
        after ajusting the longitudes from [0,360] to [-180,180].

        Parameters:
        xr_dset: data set of a climate variable
        sq_coords: spatial coordinates of the crop
        """
        # adjust the longitudes to keep a continuum over Europe
        xr_dset.coords["lon"] = (xr_dset.coords["lon"] + 180) % 360 - 180
        # re-organize data
        xr_dset = xr_dset.sortby(xr_dset.lon)
        # crop to the right square
        xr_dset = xr_dset.sel(
            lon=slice(sq_coords["min_lon"], sq_coords["max_lon"]),
            lat=slice(sq_coords["min_lat"], sq_coords["max_lat"]),
        )
        time_list = np.array(xr_dset["time"])
        n_t = len(time_list)
        n_lat = len(xr_dset.coords["lat"])
        n_lon = len(xr_dset.coords["lon"])
        nd_dset = np.ndarray((n_t, n_lat, n_lon, 1), dtype="float32")
        climate_variable = xr_dset.attrs["variable_id"]
        nd_dset[:, :, :, 0] = xr_dset[climate_variable][:, :, :]
        nd_dset = np.flip(nd_dset, axis=1)

        return nd_dset, time_list

    def sftlf_to_ndarray(
        self, xr_dset: xr.Dataset, sq_coords: dict
    ) -> (np.ndarray, np.array, str):
        """
        Converts and normalizes the land-sea mask data set
        to a cropped square ndarray,
        after ajusting the longitudes from [0,360] to [-180,180].

        Parameters:
        xr_dset: land-sea mask data set
        sq_coords: spatial coordinates of the crop
        """
        xr_dset.coords["lon"] = (xr_dset.coords["lon"] + 180) % 360 - 180
        xr_dset = xr_dset.sortby(xr_dset.lon)
        xr_dset = xr_dset.sel(
            lon=slice(sq_coords["min_lon"], sq_coords["max_lon"]),
            lat=slice(sq_coords["min_lat"], sq_coords["max_lat"]),
        )
        lat_list = xr_dset.coords["lat"]
        print(lat_list.shape, lat_list)
        lon_list = xr_dset.coords["lon"]
        n_lat = len(lat_list)
        n_lon = len(lon_list)
        land_prop = np.ndarray((n_lat, n_lon, 1), dtype="float32")
        climate_variable = xr_dset.attrs["variable_id"]
        land_prop[:, :, 0] = xr_dset[climate_variable][:, :]
        # flip upside down to have North up
        land_prop = np.flipud(land_prop)
        # normalize data (originally in [0,100])
        land_prop = land_prop / 100
        print(land_prop.shape, land_prop)

        return land_prop, lat_list, lon_list

    def normalize(self, nd_dset: np.ndarray, extrema: np.array, lat_list, lon_list) -> np.ndarray:
        """
        Normalizes a data set with given extrema.

        Parameters:
        nd_dset: data set of a climate variable
        extrema: extrema of the climate variable ([min, max])
        """
        norm_dset = (nd_dset - extrema[0]) / (extrema[1] - extrema[0])
        # create a latitude square array
        norm_lat = (lat_list-min(lat_list))/(max(lat_list)-min(lat_list))
        norm_lat = np.repeat(np.array(norm_lat), len(lon_list), axis=0)
        norm_lat = norm_lat.reshape((32,32,1))
        return norm_dset, norm_lat

    # #### 4. Split Historical Data into Train and Test Datasets
    # Train the network on most of the historical data,
    # but keep some to test the model performance on new data points.
    def split_train_test(
        self, nd_dset: np.ndarray, time_list: np.array, train_proportion: float = 0.8
    ) -> (np.ndarray, np.ndarray, np.array, np.array):
        """
        Splits a data set into train and test data (and time).

        Parameters:
        nd_dset: data set to be split
        time_list: time list to be split
        train_proportion: proportion of train data
        """
        len_train = int(len(nd_dset) * train_proportion)
        train_data = nd_dset[:len_train]
        test_data = nd_dset[len_train:]
        train_time = time_list[:len_train]
        test_time = time_list[len_train:]
        return train_data, test_data, train_time, test_time

    def split_date(
        self, nd_dset: np.ndarray, time_list: np.array, year: int = 1950
    ) -> (np.ndarray, np.ndarray, np.array, np.array):
        """
        Splits a data set into train and test data (and time),
        if the previous train_proportion splits data in the middle of a year.

        Parameters:
        nd_dset: data set to be split
        time_list: time list to be split
        year: year where the data is to be split
        """
        split_index = np.where(
            time_list == cftime.DatetimeNoLeap(year, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        )[0][0]
        train_data = nd_dset[:split_index]
        test_data = nd_dset[split_index:]
        train_time = time_list[:split_index]
        test_time = time_list[split_index:]
        return train_data, test_data, train_time, test_time

    # #### 5. Combine into a 2D-Array
    def ndarray_to_2d(self, atmosfield_dset: np.ndarray, land_prop: np.ndarray) -> np.ndarray:
        """
        Combines climate variable and land-sea mask data sets.

        Parameters:
        atmosfield_dset: climate variable data
        land_prop: land-sea mask data
        """
        n_t = np.shape(atmosfield_dset)[0]
        n_lat = np.shape(atmosfield_dset)[1]
        n_lon = np.shape(atmosfield_dset)[2]

        # combine all variables on a same period to a new 2D-array
        total_dset = np.zeros((n_t, n_lat, n_lon, 2), dtype="float32")
        total_dset[:, :, :, 0] = atmosfield_dset.reshape(n_t, n_lat, n_lon)
        total_dset[:, :, :, 1] = np.transpose(
            np.repeat(land_prop, n_t, axis=2), axes=[2, 0, 1]
        )

        return total_dset
    

    # #### 5. Combine into a 3D-Array
    def ndarray_to_3d(self, atmosfield_dset: np.ndarray, land_prop: np.ndarray, norm_lat) -> np.ndarray:
        """
        Combines climate variable and land-sea mask data sets.

        Parameters:
        atmosfield_dset: climate variable data
        land_prop: land-sea mask data
        """
        n_t = np.shape(atmosfield_dset)[0]
        n_lat = np.shape(atmosfield_dset)[1]
        n_lon = np.shape(atmosfield_dset)[2]

        # combine all variables on a same period to a new 2D-array
        total_dset = np.zeros((n_t, n_lat, n_lon, 3), dtype="float32")
        total_dset[:, :, :, 0] = atmosfield_dset.reshape(n_t, n_lat, n_lon)
        total_dset[:, :, :, 1] = np.transpose(
            np.repeat(land_prop, n_t, axis=2), axes=[2, 0, 1]
        )
        total_dset[:,:,:,2] = np.transpose(np.repeat(norm_lat, n_t, axis=2), axes=[2, 0, 1])
    
        return total_dset

    @monitor_exec
    def execute(self):
        # #### 1. Load Data to xarrays

        atmosfield = []
        for f in self.histo_extr:
            print(f)
            # Historical Datasets
            # regrouped by climate variable
            atmosfield.append(xr.open_dataset(self.dataset_root + "/" + f))
        
        atmosfield_histo = xr.concat(atmosfield, "time")

        # Load land-sea mask data
        sftlf = xr.open_dataset(
            f"{self.dataset_root}/sftlf_fx_CESM2_historical_r9i1p1f1_gn.nc",
            chunks={"time": 10},
        )

        # #### 2. Restrict to a Geospatial Square
        sq32_world_region = {
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
        }

        land_prop, lat_list, lon_list = self.sftlf_to_ndarray(sftlf, sq32_world_region)

        # Crop original data to chosen region
        atmosfield_histo_nd, time_list = self.xr_to_ndarray(
            atmosfield_histo, sq32_world_region
        )

        # Compute the variable extrema over history
        atmosfield_extrema = get_extrema(atmosfield_histo_nd)

        # Normalize data with the above extrema
        atmosfield_histo_norm, norm_lat = self.normalize(atmosfield_histo_nd, atmosfield_extrema, lat_list, lon_list)

        # Split data into train and test data sets
        train_atmosfield, test_atmosfield, train_time, test_time = self.split_train_test(
            atmosfield_histo_norm, time_list
        )

        # Add up split data and land-sea mask
        total_train = self.ndarray_to_3d(train_atmosfield, land_prop, norm_lat)
        total_test = self.ndarray_to_3d(test_atmosfield, land_prop, norm_lat)

        # Save train and test data sets
        np.save(self.input_path + "/preprocessed_3d_train_data_allssp_mod1.npy", total_train)
        np.save(self.input_path + "/preprocessed_3d_test_data_allssp_mod1.npy", total_test)
        pd.DataFrame(train_time).to_csv(self.input_path + "/dates_train_data.csv")
        pd.DataFrame(test_time).to_csv(self.input_path + "/dates_test_data.csv")

        # Projection Datasets
        # regrouped by climate variable
        # IPCC scenarios: SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
        # choose among "126", "245", "370", "585"
        # scenario = self.scenario

        for scenario in self.scenarios:
            datasets_histo = self.scenario_extr[scenario]

            atmosfield = []
            # for f in datasets_proj:
            for f in datasets_histo:
                # SSP Datasets
                # regrouped by climate variable
                atmosfield.append(xr.open_dataset(self.dataset_root + "/" + f))

            atmosfield_proj = xr.concat(atmosfield, "time")

            # #### 6. Step-by-Step Preprocessing

            # Crop original data to chosen region
            atmosfield_proj_nd, time_proj = self.xr_to_ndarray(
                atmosfield_proj, sq32_world_region
            )

            # Here are the results for CMCC-ESM2 (all scenarios)
            # to save time
            # atmosfield_extrema = np.array([234.8754, 327.64])
            # ssp585 array([234.8754, 327.64  ], dtype=float32)
            # ssp370 array([234.8754 , 325.43323], dtype=float32)
            # ssp245 array([234.8754, 324.8263], dtype=float32)
            # ssp126 array([234.8754, 323.6651], dtype=float32)

            # Normalize data with the above extrema
            atmosfield_proj_norm, norm_lat = self.normalize(atmosfield_proj_nd, atmosfield_extrema, lat_list, lon_list)

            # Add up land-sea mask
            total_proj = self.ndarray_to_3d(atmosfield_proj_norm, land_prop, norm_lat)

            # #### 7. Save Results

            # Save projection data for one scenario
            np.save(
                self.input_path + f"/preprocessed_3d_proj{scenario}_data_allssp_mod1.npy",
                total_proj,
            )
            pd.DataFrame(time_proj).to_csv(self.input_path + "/dates_proj{scenario}_data.csv")
