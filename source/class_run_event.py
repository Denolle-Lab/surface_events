import sys
sys.path.append('/data/wsd01/pnwstore/')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np
import pandas as pd
import obspy
from obspy.core import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.geodetics import *
from obspy.signal.cross_correlation import *
from obspy.signal.trigger import classic_sta_lta
from obspy.core.utcdatetime import UTCDateTime
import requests
import glob
from pnwstore.mseed import WaveformClient
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from geopy import distance
import datetime
import rasterio as rio
from rasterio.plot import show
from rasterio.merge import merge
import richdem as rd
from pathlib import Path
from pyproj import Proj,transform,Geod
import os 
from scipy.interpolate import RectBivariateSpline
import json
import matplotlib

class run_event:
    
    def __init__(self, associated_volcano, event_id, time, fs):
        self.time = time
        self.event_id = event_id
        self.associated_volcano = associated_volcano
        self.fs = fs
        
    def download_volc_data(self,df):
        self.stations = df[df['Volcano_Name'] == self.associated_volcano]['Station'].values.tolist()
        self.networks = df[df['Volcano_Name'] == self.associated_volcano]['Network'].values.tolist()
        self.latitudes = df[df['Volcano_Name'] == self.associated_volcano]['Latitude'].values.tolist()
        self.longitudes = df[df['Volcano_Name'] == self.associated_volcano]['Longitude'].values.tolist()
        self.elevations = df[df['Volcano_Name']== self.associated_volcano]['Elevation'].values.tolist()

        if self.stations.count("LON")>0 and self.stations.count("LO2")>0:
            index = self.stations.index("LO2")
            del self.stations[index]
            del self.networks[index]
            del self.latitudes[index]
            del self.longitudes[index]
            del self.elevations[index]
            
    def download_waveform_data(self, t_before, t_after, client, fs, window, thr, smooth_length, pr):
        # resamples the data
        def resample(st, fs):
            for i in st:
                i.detrend(type='demean')
                i.taper(0.05)
                i.resample(fs)   
            return st
        
        bulk = [] 
        for m in range(0, len(self.networks)):
            bulk.append([self.networks[m], self.stations[m], '*', '*', self.time-t_before, self.time+t_after])
        self.st = client.get_waveforms_bulk(bulk)

        #remove unwanted data
        for tr in self.st:
            cha = tr.stats.channel
            if cha[0:2] != 'BH' and cha[0:2] != 'EH' and cha[0:2] != 'HH':
                self.st.remove(tr)
            try:
                if len(tr.data)/tr.stats.sampling_rate < 239.9:
                    self.st.remove(tr)
            except:
                pass

        # resampling the data to 40Hz for each trace
        self.st = resample(self.st,fs) 
        
        #Plotting all traces for one event with channel z, SNR>10, and bandpasses between 2-12Hz
        self.SNR = []
        self.SNR_weight = []
        self.no_weight = []
        self.stas = []
        self.nets = []
        max_amp_times = []
        self.durations = []
        self.data_env_dict = {}
        self.t_diff = {}
        
        for i,ii in enumerate(self.st):
            self.network = ii.stats.network
            self.station = ii.stats.station
            ii.detrend(type = 'demean')
            ii.filter('bandpass',freqmin=2.0,freqmax=12.0,corners=2,zerophase=True)
            self.cha = ii.stats.channel
            self.starttime = ii.stats.starttime
            max_amp_time = np.argmax(ii.data)/fs
            signal_window = ii.copy()
            noise_window = ii.copy()
            signal_window.trim(self.starttime + t_before - 20, self.starttime + t_before - 20 + window)
            noise_window.trim(self.starttime + t_before - window -10, self.starttime + t_before - 10)
            self.snr = (20 * np.log(np.percentile(np.abs(signal_window.data),pr) 
                           / np.percentile(np.abs(noise_window.data),pr))/np.log(10))

            if self.cha[-1] == 'Z' and self.snr>thr and 100<max_amp_time<200:
                self.t_diff[self.network+'.'+self.station] = self.starttime-self.time 
                # enveloping the data 
                data_envelope = obspy.signal.filter.envelope(ii.data[115*fs:150*fs])
                data_envelope /= np.max(data_envelope)
                # finding the time of max amplitude of each event
                max_amp_times.append(max_amp_time)
                max_amp = np.max(ii.data)      
                # creating envelope data dictionary to calculate picktimes
                data_envelope = obspy.signal.util.smooth(data_envelope, smooth_length)
                self.data_env_dict[self.network+'.'+self.station]= data_envelope
                self.stas.append(ii.stats.station)
                self.nets.append(ii.stats.network)
                self.SNR.append(self.snr)
                self.SNR_weight.append(int(self.snr))
                self.no_weight.append(1)
            else:
                self.st.remove(ii)

#         if len(self.st)<4:  
#             continue

        # get peak frequency of each event
        # read and preprocess data
        self.st.taper(max_percentage=0.01,max_length=20)
        self.st.trim(starttime=self.time-20,endtime=self.time+30)
        
    def plot_waveforms(fs,thr):
        fig = plt.figure(figsize = (11,8), dpi=200)
        fig.suptitle('evtID:UW'+ str(self.event_id)+sef.associated_volcano)
        plt.rcParams.update({'font.size': 20})
        ax1 = plt.subplot(1,1,1)
        iplot = 0
        for i,ii in enumerate(self.st):
            self.network = ii.stats.network
            self.station = ii.stats.station
            self.cha = ii.stats.channel
            self.starttime = ii.stats.starttime

            if self.SNR[i]>thr and 100<self.max_amp_times[i]<200:
                t = ii.times()
                # enveloping the data 
                data_envelope = self.data_env_dict[self.network+'.'+self.station]
                data_envelope += iplot*1.5     
                b,e = 115,150
                ax1.plot(t[b*fs:e*fs],ii.data[b*fs:e*fs]/np.max(np.abs(ii.data))+iplot*1.5)
                ax1.plot(t[115*fs:150*fs], data_envelope, color = 'k')
                ax1.set_xlabel('time (seconds)')
                ax1.set_xlim([b,e])
                ax1.set_yticks([])
                plt.text(t[e*fs], iplot*1.5, 'SNR:'+str(int(self.snr)))
                plt.text(t[b*fs], iplot*1.5, self.station)
                iplot = iplot+1
                    
                    
    def phase_picking_env_cc(ref_env):
        def pick_time(time, ref_env, data_env_dict, st, t_diff, t_before, fs):
            # time; picktime from the PNSN
            # ref_env; reference envelope
            # data_env_dict; dictionary of the envelope data, key is the network+station
            # st; stream of traces of the waveforms
            # t_diff; 120
            # t_before; 120
            # fs sample rate
            pick_times,offsets, starttimes = [],[],[]
            for i,key in enumerate(data_env_dict):
                starttimes.append(st[i].stats.starttime)
                xcor = correlate(data_env_dict[key],ref_env,int(50*fs))
                index = np.argmax(xcor)
                cc = round(xcor[index],9) #correlation coefficient
                shift = 50*fs-index #how much it is shifted from the reference envelope
                offset_time = time - shift/fs # shift from one envelope to the reference envelope in seconds
                offsets.append(offset_time) # number of seconds from the beginning of the trace
                pick_times.append(offset_time + 120)
            return pick_times, offsets, starttimes

    
        def shift(offsets, starttimes, t_diff):
            shifts, vals =[],[]
            for i,ii in enumerate(t_diff):
                t_shift = offsets[i]-min(offsets)
                vals.append((-1*t_diff[ii])+t_shift)
                shifts.append(t_shift)
            return shifts, vals
        
        
        
        # calculating the picktimes and shift in arrival times using envelope cross_correlation
        self.pick_times, self.offsets, self.starttimes = pick_time(self.time, ref_env, data_env_dict,st,t_diff, t_before, fs) #calculate picktimes
        self.shifts, self.vals = shift(pick_times, offsets, starttimes, t_diff)
        
    
    def event_location():
        
        # predict synthetic arrival times
        def travel_time(t0, x, y, vs, sta_x, sta_y):
            dist = np.sqrt((sta_x - x)**2 + (sta_y - y)**2)
            tt = t0 + dist/vs
            return tt

        # compute residual sum of squares
        def error(synth_arrivals,arrivals, weight):
            res = (arrivals - synth_arrivals)* weight 
            res_sqr = res**2
            mse = np.mean(res_sqr)
            rmse = np.sqrt(mse)
            return rmse

        # iterate through grid and calculate travel time residuals
        def gridsearch(t0,x_vect,y_vect,sta_x,sta_y,vs,arrivals, weight):
            rss_mat = np.zeros((len(t0),len(x_vect),len(y_vect)))
            rss_mat[:,:,:] = np.nan
            for i in range(len(t0)):
                for j in range(len(x_vect)):
                    for k in range(len(y_vect)):
                        synth_arrivals = []
                        for h in range(len(sta_x)):
                            tt = travel_time(t0[i],x_vect[j],y_vect[k],vs,sta_x[h],sta_y[h])
                            synth_arrivals.append(tt)
                        rss = error(np.array(synth_arrivals),np.array(arrivals), np.array(weight))
                        rss_mat[i,j,k] = rss
            return rss_mat

        # find lower-left corner of grid and grid size based on height of volcano
        def start_latlon(elevation, ratio, center_lat, center_lon):
            side_length = elevation * ratio
            l = side_length/2
            hypotenuse = l*np.sqrt(2)
            d = distance.geodesic(meters = hypotenuse)
            start_lat = d.destination(point=[center_lat,center_lon], bearing=225)[0]
            start_lon = d.destination(point=[center_lat,center_lon], bearing=225)[1]
            return start_lat, start_lon, side_length

        # convert the location index into latitude and longitude
        def location(x_dist, y_dist, start_lat, start_lon):
            bearing = 90-np.rad2deg(np.arctan(y_dist/x_dist)) 
            dist = np.sqrt((x_dist)**2 + (y_dist)**2)
            d = distance.geodesic(meters = dist)
            loc_lat = d.destination(point=[start_lat,start_lon], bearing=bearing)[0]
            loc_lon = d.destination(point=[start_lat,start_lon], bearing=bearing)[1]
            return loc_lat, loc_lon, d

        # find diameter in meters of the error on the location
        def error_diameter(new_array):
            min_idx = np.min(new_array[:,1]) # get the left most index 
            max_idx = np.max(new_array[:,1]) # get the right most index
            difference = max_idx-min_idx # take the difference
            diameter_m = difference*1000 # convert to meters
            return diameter_m 
        
        
        self.arrivals = self.shifts
        self.sta_lats = self.lats
        self.sta_lons= self.lons

        # define grid origin in lat,lon and grid dimensions in m
        lat_start = volc_grid[associated_volcano][0]
        lon_start = volc_grid[associated_volcano][1]
        side_length = volc_grid[associated_volcano][2]

        # create the grid of locations
        self.sta_x = []
        self.sta_y = []
        for i in range(len(sta_lats)):
            self.x_dist = distance.distance([lat_start,lon_start],[lat_start,sta_lons[i]]).m
            self.y_dist = distance.distance([lat_start,lon_start],[sta_lats[i],lon_start]).m
            self.sta_x.append(self.x_dist)
            self.sta_y.append(self.y_dist)
        x_vect = np.arange(0, side_length, step)
        y_vect = np.arange(0, side_length, step)
        t0 = np.arange(0,np.max(arrivals),t_step)

        # gridsearch with no weight
        weight = [1 for i in range(len(SNR_weight))]
        rss_mat = gridsearch(t0,x_vect,y_vect,sta_x,sta_y,1000,arrivals,weight)
        loc_idx = np.unravel_index([np.argmin(rss_mat)], rss_mat.shape)
        # find the latitude and longitude of the location index 
        loc_lat, loc_lon, d = location(x_vect[loc_idx[1]], y_vect[loc_idx[2]], lat_start, lon_start)
        err_thr = np.min(np.log10(rss_mat))+.05
        thr_array = np.argwhere(np.log10(rss_mat)<err_thr)
        self.diameter = error_diameter(thr_array)

        # gridsearch weighted by SNR
        weight = np.array(SNR_weight)/np.max(SNR_weight)
        rss_mat_snr = gridsearch(t0,x_vect,y_vect,sta_x,sta_y,1000,arrivals,weight)
        loc_idx_snr = np.unravel_index([np.argmin(rss_mat_snr)], rss_mat_snr.shape)
        self.loc_lat_snr, self.loc_lon_snr, test_d = location(x_vect[loc_idx_snr[1]], y_vect[loc_idx_snr[2]], lat_start, lon_start)

        # gridsearch weighted with SNR and Slope
        # gives the left right, bottom, top of the grid
        left, right = r_dem_data_dict[associated_volcano]['left'],r_dem_data_dict[associated_volcano]['right']
        bottom, top = r_dem_data_dict[associated_volcano]['bottom'],r_dem_data_dict[associated_volcano]['top']

        a = int((left_x-left)/10)
        b = a+2500
        c = (slope.shape[0] - int((bottom_y-bottom)/10))-2500
        d = slope.shape[0] - int((bottom_y-bottom)/10)

        x = np.arange(a,b,1)
        y = np.arange(c,d,1)

        x2 = np.arange(a,b,10) # every 100m
        y2 = np.arange(c,d,10) # every 100m

        slope_data = np.array(slope[c:d,a:b])

        slope_data[slope_data < 1] = 1
        slope_data[slope_data > 90] = 80

        slope_norm = 1/slope_data

        slope_interp_mat = RectBivariateSpline(y,x,slope_norm, s = 0)
        interp = (slope_interp_mat(x2,y2)/np.max(slope_interp_mat(x2,y2)))*0.1+.9

        # gridsearch weighted with slope
        rss_mat_slope = np.multiply(rss_mat[loc_idx[0],:,:],(interp))
        loc_idx_slope = np.unravel_index([np.argmin(rss_mat_slope)], rss_mat_slope.shape)
        loc_lat_slope, loc_lon_slope, test_d = location(x_vect[loc_idx_slope[1]], y_vect[loc_idx_slope[2]], lat_start, lon_start)

        # gridsearch weighted with snr and slope
        rss_mat_slopesnr = np.multiply(rss_mat_snr[loc_idx[0],:,:],(interp))
        loc_idx_slopesnr = np.unravel_index([np.argmin(rss_mat_slopesnr)], rss_mat_slopesnr.shape)
        self.loc_lat_slopesnr, self.loc_lon_slopesnr, test_d = location(x_vect[loc_idx_slopesnr[1]], y_vect[loc_idx_slopesnr[2]], lat_start, lon_start)
            
        
    
    def doppler_shift_algorithm():
        # fit data to a cosine curve
        def test_func(theta, a,theta0, c):
            return a * np.cos(theta-theta0)+c
        
        # calculating azimuth for each station with respect to the location of the event
        for i in range(len(stas)):
            u,b,c = gps2dist_azimuth(loc_lat_slope, loc_lon_slope, lats[i], lons[i], a=6378137.0, f=0.0033528106647474805)
            r.append(u)
            theta.append(b)
            
        # ensuring that there is sufficient station symmetry around the event
        bin1,bin2,bin3 = [],[],[]
        for i in theta:
            if 0<=i<=120:
                bin1.append(i)
            if 121<=i<=240:
                bin2.append(i)
            if 241<=i<=360:
                bin3.append(i)

#         if bin1 == [] or bin2 == [] or bin3 == []:
#             continue

        #manipulating the data
        data = {'azimuth_deg':theta, 'freq':char_freq, 'station':stas, 'distance_m':r, 
                'weight':sharp_weight, 'SNR':SNR, 'colors':colors[0:len(stas)]}
        DF = pd.DataFrame(data, index = None)
        DF2 = DF.sort_values('azimuth_deg')

        #Taking out stations that are too close to the location when looking at azimuth 
        drops = []
        for i in range(len(DF2)):
            value = DF2.loc[i,'distance_m']
            if value < az_thr:
                drops.append(i)
                
        self.DF3 = DF2.drop(drops)
        self.y_data =  self.DF3["freq"].values.tolist()
        self.Sta2 = self.DF3["station"].values.tolist()
        self.dist2 = self.DF3["distance_m"].values.tolist()
        self.self.spike_weight = self.DF3["weight"].values.tolist()
        self.SNR2 = self.DF3['SNR'].values.tolist()
        self.colors2 = self.DF3['colors'].values.tolist()
        self.x_data =  np.asarray(self.DF3["azimuth_deg"].values.tolist())
        self.x_points = np.linspace(0,360, 100)

        #optimizing parameters to fit data to test_function
        params, params_covariance = optimize.curve_fit(test_func, np.deg2rad(self.x_data), self.y_data, p0=None)
        perr = np.sqrt(np.diag(params_covariance))
        self.std_deviation = str(round(perr[0],9))+','+str(round(perr[1],9))+','+str(round(perr[2],9))
        self.d = test_func(np.deg2rad(self.x_points), params[0], params[1], params[2])
        self.len_r = int(max(self.r))

        if params[0]<0:
            self.direction = params[1]+pi 
        else:
            self.direction = params[1]
        
        # estimate the velocity based off of minimum and maximum frequencies
        fmax = max(self.d)
        fmin = min(self.d)
        self.v = v_s*((fmax-fmin)/(fmax+fmin))










