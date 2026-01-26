from datetime import datetime, timedelta , date
import logging
import os
import glob
import shutil

import pandas as pd

from tkinter import Tk, filedialog
from IPython.display import display, clear_output
import ipywidgets as widgets
import traitlets
import shapefile

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select Files"
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.files = filedialog.askopenfilename(multiple=True,filetypes=[('shp','.shp')])

        b.description = "Files Selected"
        b.icon = "check-square-o"
        b.style.button_color = "lightgreen"

def _last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + timedelta(days=4)  # this will never fail
    return next_month - timedelta(days=next_month.day)

def _rename_columns(df, column_name, suffix):
    if column_name in df.columns:
        df.rename(columns={column_name: f"{column_name}_{suffix}"}, inplace=True)

def monthlist(begin,end):
    # begin = datetime.strptime(begin, "%Y-%m-%d")
    # end = datetime.strptime(end, "%Y-%m-%d")

    result = []
    while True:
        if begin.month == 12:
            next_month = begin.replace(year=begin.year+1,month=1, day=1)
        else:
            next_month = begin.replace(month=begin.month+1, day=1)
        if next_month > end:
            break
        result.append ([begin.strftime("%Y-%m-%d"),_last_day_of_month(begin).strftime("%Y-%m-%d")])
        begin = next_month
    result.append ([begin.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d")])
    return result


def date_format_concersion(date, output_format='%Y/%m/%d'):

    # Fool-proof: check if the input date is None
    if date is None:
        return None

    try: 
        parsed_date = datetime.strptime(date, output_format)
        
    except ValueError as e:

        try:
            # Try to parse the input date
            parsed_date = datetime.strptime(date, '%Y%m%d')
        except ValueError as e:
        
            try:
                parsed_date = datetime.strptime(date, '%Y_%m_%d')
            except ValueError as e:

                try:
                    parsed_date = datetime.strptime(date, '%Y-%m-%d')
                except ValueError as e:

                    return logging.error(f'Unparsable date format {e}')

    output_date = parsed_date.strftime(output_format)

    return output_date

def read_shp_date_data(my_button):
    shp = shapefile.Reader("".join(my_button.files))
    b = []
    for i in range(len(shp.fields)):
        a = shp.fields[i][0]
        b.append(a)

    # give the shapefile name

    file_name = widgets.Dropdown(
    options=b,
    description='Regional category')

    # give the star date and end date

    star = widgets.DatePicker(
        description='Pick a Star Date',
        disabled=False
    )
    end = widgets.DatePicker(
        description='Pick a End Date',
        disabled=False
    )

    return file_name, star, end

def read_bands_statics(setting_band_config : list):
    band_name=widgets.SelectMultiple(
    options=setting_band_config,
    description='Band',
    )

    statics =widgets.Dropdown(
        options=['MEAN','MAXIMUM', 'MINIMUM', 'MEDIAN', 'STD', 'VARIANCE', 'SUM'],
        value='MEAN',
        description='Statistics')

    return band_name, statics

def make_temp_file(folder_name: str):
    if not os.path.exists('./' + folder_name):
        os.makedirs('./' + folder_name)
        logging.info('create temp folder')
    else:
        logging.info('temp folder already exists')

    return folder_name


def cbind_era5(statics, folder_name, era5_band_list, folder):

    all_files = glob.glob(os.path.join(folder_name, "Era5_{}*.csv".format(statics)))
    
    df_from_each_file = (pd.read_csv(f, sep = ",") for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index = True)

    for column_name in df_merged.columns:
        if column_name in era5_band_list:
            _rename_columns(df_merged, column_name, str(statics))

    df_merged.to_csv(folder.selected + '.csv',index=False)

    shutil.rmtree(folder_name, ignore_errors=True)

def cbind_era5(statics, folder_name, era5_band_list, folder):

    all_files = glob.glob(os.path.join(folder_name, "Era5_{}*.csv".format(statics)))
    
    df_from_each_file = (pd.read_csv(f, sep = ",") for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index = True)

    for column_name in df_merged.columns:
        if column_name in era5_band_list:
            _rename_columns(df_merged, column_name, str(statics))

    df_merged.to_csv(folder.selected + '.csv',index=False)

    shutil.rmtree(folder_name, ignore_errors=True)

def cbind_modis_ndvi_evi(statics, folder_name, modis_band_list, folder):

    all_files = glob.glob(os.path.join(folder_name, "Modis_NDVI_EVI_{}.csv".format(statics)))
    
    df_from_each_file = (pd.read_csv(f, sep = ",") for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index = True)

    for column_name in df_merged.columns:
        if column_name in modis_band_list:
            _rename_columns(df_merged, column_name, str(statics))

    df_merged.to_csv(folder.selected + '.csv',index=False)

    shutil.rmtree(folder_name, ignore_errors=True)

def cbind_chirsp(statics, folder_name, modis_band_list, folder):

    all_files = glob.glob(os.path.join(folder_name, "chirsp_{}.csv".format(statics)))
    
    df_from_each_file = (pd.read_csv(f, sep = ",") for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index = True)

    for column_name in df_merged.columns:
        if column_name in modis_band_list:
            _rename_columns(df_merged, column_name, str(statics))

    df_merged.to_csv(folder.selected + '.csv',index=False)

    shutil.rmtree(folder_name, ignore_errors=True)

def cbind_modis_lst(statics, folder_name, modis_band_list, folder):

    all_files = glob.glob(os.path.join(folder_name, "Modis_LST_{}.csv".format(statics)))
    
    df_from_each_file = (pd.read_csv(f, sep = ",") for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index = True)

    for column_name in df_merged.columns:
        if column_name in modis_band_list:
            _rename_columns(df_merged, column_name, str(statics))

    df_merged.to_csv(folder.selected + '.csv',index=False)

    shutil.rmtree(folder_name, ignore_errors=True)