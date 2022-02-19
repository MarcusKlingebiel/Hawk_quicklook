
#Get arguments from command line
import sys

hawk_file 		 = sys.argv[1]
calibration_file = sys.argv[2]
folder_output    = sys.argv[3]

print("********************")
print(hawk_file)
print(calibration_file)
print(folder_output)
print("********************")

# Settings -------------------------------------
Create_Images = True
image_wavelength = 2100 #nm

Create_Radiance_Plots = True
selected_wavelengths = [1200, 1600, 2200] #nm

Create_NetCDF_File = True
Create_Spectral_Center_line_file = True

#Import modules --------------------------------

from spectral import *
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates
import pandas as pd
import os
import xarray
import gc




#Functions -------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_hawk_raw(file):
    img = open_image(file)
    wvls= img.bands.centers
    arr = img.asarray()
    date = img.metadata['acquisition date'][-10:]
    date = date[-4:]+"-"+date[-7:-5]+"-"+date[-10:-8]
    start_time = img.metadata['gps start time'][-13:-1]
    starttime = datetime.fromisoformat(date+" "+start_time)
    time_ls  = []
    for i in range(0,len(arr)):
        time_ls.append(starttime + i* timedelta(seconds=0.05))
    time = np.asarray(time_ls)
    autodarkstartline = int(img.metadata['autodarkstartline'][:])
    tint = float(img.metadata['tint'][:])
    return arr, wvls, time, autodarkstartline, tint
    del img, arr
    gc.collect()

def read_hawk_calibration(file):
    print(file[:-3]+"cal")
    img = envi.open(file, image = file[:-3]+"cal")
    wvls= img.bands.centers
    arr = img.asarray()
    arr = arr / 100
    tint = float(img.metadata['tint'][:])
    return arr, wvls, tint
    del img, arr
    gc.collect()

def averaging_dark_current(array, autodarkstartline):
    arr = array[autodarkstartline:,:,:]
    #print(np.shape(arr))
    arr_dark = np.zeros(shape=np.shape(arr[1]))
    for i in (range((np.shape(arr)[1]))):
        for j in range((np.shape(arr)[2])):
            arr_dark[i,j]  = np.nanmean(arr[:,i,j])
    return arr_dark
    del array, arr_dark
    gc.collect()

def plot_image(array, time, wavelength, array_wavelengths, filename):
    fs=16
    wvl_idx = find_nearest(array_wavelengths,wavelength)
    _, ax = plt.subplots(figsize=(12,4))
    ax.contourf(time[:],np.arange(0,384),array[:,:,wvl_idx].T,25,cmap="Greys_r")
    ax.set_ylabel("across-track-pixel", fontsize=fs)
    ax.set_xlabel("UTC time", fontsize=fs)
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    #print(filename)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    del array
    gc.collect()
    
def plot_radiance(time, array,array_wavelengths, selected_wavelengths, across_track_pixel_position, filename):
    fs=16
    _, ax = plt.subplots(figsize=(12,4))
    for i in range(len(selected_wavelengths)):
        wvl_idx = find_nearest(array_wavelengths,selected_wavelengths[i])
        ax.plot(time, array[:,across_track_pixel_position,wvl_idx].T, label = str(selected_wavelengths[i]) + "nm")
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_ylabel("Radiance ($W~m^{-2}~sr^{-1}$)", fontsize=fs)
    ax.set_xlabel("UTC time", fontsize=fs)
    plt.legend(fontsize=fs-6)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    del array
    gc.collect()
      
def save_radiances_as_netcdf(time, array,array_wavelengths, selected_wavelengths, across_track_pixel_position, filename):
    df = pd.DataFrame(time_meas, columns=["utc_time"])
    for i in range(len(selected_wavelengths)):
        wvl_idx = find_nearest(array_wavelengths,selected_wavelengths[i])
        df["rad_"+str(selected_wavelengths[i])+"nm"] = array[:,across_track_pixel_position,wvl_idx].T
    df.index = df["utc_time"]
    df = df.drop('utc_time', 1)
    ds = df.to_xarray()
    ds.to_netcdf(filename)
    del array
    del ds
    gc.collect()
    
    
def save_spectral_center_line(array,time, wavelengths, filename):
    arr_meas_exp = array[:,192-5:192+5,:]
    arr_meas_exp_mean = np.nanmean(arr_meas_exp, axis = 1)
    np.savez(filename,array = arr_meas_exp_mean, wvl = wavelengths, time=time)
    del array
    gc.collect()




# Main Programm

arr_cal, wvls_cal, tint_cal = read_hawk_calibration(calibration_file)

arr_meas, wvls_meas, time_meas, autodarkstartline_meas, tint_meas = read_hawk_raw(hawk_file)
    
#Print file name
print("*******************************************************************")
print("File: "+ os.path.basename(hawk_file))

#Averaging the dark current
dark_current = averaging_dark_current(arr_meas,autodarkstartline_meas)
print("1. Averaging the dark current done")

#Subtracting the dark current from the measurement
arr_meas_dk = arr_meas - dark_current
print("2. Substraction of the dark current from the measurement done")
del dark_current, arr_meas
gc.collect()

#Correcting with the calibration file and the integration time
Result_Hawk = arr_meas_dk * arr_cal / tint_meas
print("3. Correction with calibration file and integration time done")
del arr_meas_dk
gc.collect()
  
#Plot and save image
if Create_Images:
    plot_image(Result_Hawk, time_meas, 1800, wvls_meas, folder_output + os.path.basename(hawk_file)[:-4]+".png")
    print("4. Image plotted and saved")
    print(folder_output + os.path.basename(hawk_file)[:-4]+".png")

#Plot radiance and save plot
if Create_Radiance_Plots:
    plot_radiance(time_meas,Result_Hawk,wvls_meas, selected_wavelengths, int(384/2),folder_output + os.path.basename(hawk_file)[:-4]+"_radiances.png")
    print("5. Radiances plotted and saved")
    print(folder_output + os.path.basename(hawk_file)[:-4]+"_wvls.png")
    
#Save radiances as NetCDF file
if Create_NetCDF_File:
    save_radiances_as_netcdf(time_meas,Result_Hawk,wvls_meas, selected_wavelengths, int(384/2),folder_output + os.path.basename(hawk_file)[:-4]+"_radiances.nc")
    print("6. NetCDF created")
    print(os.path.basename(hawk_file)[:-4]+"_radiances.nc")
    
#Save spectra of 10 averaged center lines
if Create_Spectral_Center_line_file:
    save_spectral_center_line(Result_Hawk,time_meas,wvls_meas,folder_output + os.path.basename(hawk_file)[:-4]+"_spectra.npz")
    print("7. Spectra of 10 average center lines saved")
    print(folder_output + os.path.basename(hawk_file)[:-4]+"_spectra.npz")
    
del Result_Hawk
gc.collect()

