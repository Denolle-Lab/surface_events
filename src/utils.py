# functions used in the surface event analysis
import numpy as np
from obspy.signal.cross_correlation import correlate
from obspy.core import UTCDateTime
from geopy import distance
from joblib import Parallel, delayed
import pyproj

#define a function that calculates picktimes at each station

def pick_time(time, ref_env, data_env_dict, st, t_diff, t_before, fs):
    pick_times,offsets, starttimes = [],[],[]
    for i,key in enumerate(data_env_dict):
        starttimes.append(st[i].stats.starttime)
        xcor = correlate(data_env_dict[key], ref_env,int(50*fs))
        index = np.argmax(xcor)
        cc = round(xcor[index],9) #correlation coefficient
        shift = 50*fs-index #how much it is shifted from the reference envelope
        offset_time = time - shift/fs
        p = time - shift/fs  # p is the new phase pick for each station
        pick_times.append(p+t_diff[key])
        offsets.append(offset_time + t_diff[key])
    return pick_times, offsets, starttimes

def shift(pick_times, offsets, starttimes, t_diff):
    shifts, vals =[],[]
    for i,ii in enumerate(t_diff):
        t_shift = offsets[i]-min(offsets)
        vals.append((-1*t_diff[ii])+t_shift)
        shifts.append(t_shift)
        #plt.vlines(val, ymin = iplot*1.5-.5, ymax = iplot*1.5+.5, color = colors[i])
    return shifts, vals

# define functon that resamples the data

def resample(st, fs):
    for i in st:
        i.detrend(type='demean')
        i.taper(0.05)
        i.resample(fs)   
    return st

# define function to calculate number of surface events per month
def events_per_month(starttimes, events):
    num_events = {}
    for year in range(2001, 2021):
        for month in range(1, 13):
            Nevt = []
            period = str(year)+"_"+str(month)
            t0 = UTCDateTime(year, month, 1)
            t1 = t0+3600*24*30
            for i in range(0, len(starttimes)):
                if t0<starttimes[i]<t1:
                    Nevt.append(events[i])
            if len(Nevt) != 0:
                num_events[period]=len(Nevt)
            if len(Nevt) == 0:
                num_events[period] = 0

    periods = list(num_events.keys())
    num_of_events = list(num_events.values())
    return periods, num_of_events

# define function to fit data to

def test_func(theta, a,theta0, c):
    return a * np.cos(theta-theta0)+c

# define a function to make plots of weighted data

def weight_data(x_data,y_data,weight,test_func,v_s,stas):    
    #weighting the data
    tempx, tempy = [],[]
    for i,ii in enumerate(x_data):
        tempx.append([])
        tempx[i].append([ii for l in range(0,weight[i])])
        tempy.append([])
        tempy[i].append([y_data[i] for l in range(0,weight[i])])   
    weighted_x = sum(sum(tempx, []),[])
    weighted_y = sum(sum(tempy, []),[])
   
    #optimizing parameters to fit weighted data to test_function
    params, params_covariance = optimize.curve_fit(test_func, np.deg2rad(weighted_x), weighted_y, p0=None)
    d = test_func(np.deg2rad(x_points), params[0], params[1], params[2])
    if params[0]<0:
        direction = params[1]+pi 
    else:
        direction = params[1]   
    fmax = max(d)
    fmin = min(d)
    v = v_s*((fmax-fmin)/(fmax+fmin))
    return v, direction, d

# define function to predict synthetic arrival times
def travel_time(t0, x, y, vs, sta_x, sta_y):
    dist = np.sqrt((sta_x - x)**2 + (sta_y - y)**2)
    tt = t0 + dist/vs
    return tt

# # define function to compute residual sum of squares
# def error(synth_arrivals,arrivals, weight):
#     res = (arrivals - synth_arrivals)* weight 
#     res_sqr = res**2
#     mse = np.mean(res_sqr)
#     rmse = np.sqrt(mse)
#     return rmse

#define function to iterate through grid and calculate travel time residuals
def gridsearch(lat_start, lon_start, lat_end, lon_end, sta_lat, sta_lon,\
               arrivals, x_step = 100 , t_step = 0.5, vs = 1000, weight=None):
    '''
    gridsearch(t0,X,Y,sta_x,sta_y,vs,arrivals, weight)
    lat_start, lon_start= lat, lon of source bottom left corner
    lat_end,lon_end= lat, lon of source top right corner
    sta_lat,sta_lon = lat, lon coordinates of stations
    arrivals = array of arrival times of each station

    x_step = spatial increment in meters, default is 100 m
    t_step = temporal increment in seconds, default is 0.1 s
    vs = velocity of shear wave in m/s, default is 1000 m/s
    weight = array of weights for each station measurements



    '''

    proj = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')

    if lon_start<0: 
        lon_start+=360
        lon_end+=360

    # Convert lat/long to Cartesian in meters
    xsta, ysta = proj(sta_lon, sta_lat)
    x1,y1=proj(lon_start,lat_start)
    x2,y2=proj(lon_end,lat_end)
    # Generate the x and y coordinates for the grid
    x_coords = np.arange(x1, x2, x_step)
    y_coords = np.arange(y1, y2, x_step)
    t0 = np.arange(-3,3,t_step)

    tpick =arrivals-np.min(arrivals)
    rss_mat = np.zeros((len(t0),len(x_coords),len(y_coords)))
    rss_mat[:,:,:] = np.nan
    for i in range(len(t0)):
        for j in range(len(x_coords)):
            for k in range(len(y_coords)):
                synth_arrivals = np.zeros(len(xsta))
                for h in range(len(xsta)):
                    tt = travel_time(t0[i],x_coords[j],y_coords[k],vs,xsta[h],ysta[h])
                    synth_arrivals[h]=tt
                if weight is None:
                    rss = np.sqrt(np.mean(((tpick - synth_arrivals) ))**2)
                else:
                    rss = np.sqrt(np.mean(((tpick - synth_arrivals) *weight))**2)


                # giant residual matri
                rss_mat[i,j,k] = rss

        # I confirm that this grid search is working
        # Create a new figure with a specific size (in inches) and DPI
        # fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        # cax = ax.imshow(rss_mat[i,:,:], cmap='hot', interpolation='nearest')
        # fig.colorbar(cax)
        # plt.show()
    
    oc_idx = np.unravel_index([np.argmin(rss_mat)], rss_mat.shape)
    t_best = t0[oc_idx[0]]
    x_best = x_coords[oc_idx[1]]
    y_best = y_coords[oc_idx[2]]

    if oc_idx[0]==len(t0) or oc_idx[0]==0: print("grid search failed to converge in time")
    if oc_idx[1]==len(x_coords) or oc_idx[1]==0: print("grid search failed to converge in longitude")
    if oc_idx[2]==len(y_coords) or oc_idx[2]==0: print("grid search failed to converge in latitude")

    # Convert Cartesian back to lat/long
    lon_best, lat_best = proj(x_best, y_best, inverse=True)

    return rss_mat, t_best, lon_best, lat_best, oc_idx

#define function to iterate through grid and calculate travel time residuals

def gridsearch_parallel(lat_start,lon_start,lat_end,lon_end,sta_lat,sta_lon,\
               arrivals, x_step=100,t_step=0.25,vs=1000,weight=None):
    '''
    gridsearch(t0,X,Y,sta_x,sta_y,vs,arrivals, weight)
    lat_start, lon_start= lat, lon of source bottom left corner
    lat_end,lon_end= lat, lon of source top right corner
    sta_lat,sta_lon = lat, lon coordinates of stations
    arrivals = array of arrival times of each station

    x_step = spatial increment in meters, default is 100 m
    t_step = temporal increment in seconds, default is 0.1 s
    vs = velocity of shear wave in m/s, default is 1000 m/s
    weight = array of weights for each station measurements



    '''

    proj = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')

    if lon_start<0: 
        lon_start+=360
        lon_end+=360

    # Convert lat/long to Cartesian in meters
    xsta, ysta = proj(sta_lon, sta_lat)
    x1,y1=proj(lon_start,lat_start)
    x2,y2=proj(lon_end,lat_end)
    # Generate the x and y coordinates for the grid
    x_coords = np.arange(x1, x2, x_step)
    y_coords = np.arange(y1, y2, x_step)
    t0 = np.arange(-5,5,t_step)

    tpick =arrivals-np.min(arrivals)
    def rss_calc(tt0):
        resmin=np.inf
        idx=[0,0]
        for j in range(len(x_coords)):
            for k in range(len(y_coords)):
                for h in range(len(xsta)):
                    tt = travel_time(tt0,x_coords[j],y_coords[k],vs,xsta[h],ysta[h])
                if weight is None:
                    rss = np.sqrt(np.mean(((tpick - tt) ))**2)
                else:
                    rss = np.sqrt(np.mean(((tpick - tt) *weight))**2)
                if rss<resmin:
                    resmin=rss
                    idx = [j,k]
        return resmin, idx[0],idx[1]
    
    # Create a pool of workers to execute the rss_calc function in parallel
    results = np.array(Parallel(n_jobs=8)(delayed(rss_calc)(tt0) for tt0 in t0))
    imin = np.argmin(results[:,0],axis=0)
    print(imin)
    t_best = t0[imin]
    x_best = x_coords[int(results[imin, 1])] 
    y_best = y_coords[int(results[imin, 2])] 
    # Convert Cartesian back to lat/long
    lon_best, lat_best = proj(x_best, y_best, inverse=True)

    return t_best, lon_best, lat_best #,oc_idx


# define function to find lower-left corner of grid and grid size based on height of volcano
def start_latlon(elevation, ratio, center_lat, center_lon):
    side_length = elevation * ratio
    l = side_length/2
    hypotenuse = l*np.sqrt(2)
    d = distance.geodesic(meters = hypotenuse)
    start_lat = d.destination(point=[center_lat,center_lon], bearing=225)[0]
    start_lon = d.destination(point=[center_lat,center_lon], bearing=225)[1]
    return start_lat, start_lon, side_length

# define function to convert the location index into latitude and longitude
def location(x_dist, y_dist, start_lat, start_lon):
    bearing = 90-np.rad2deg(np.arctan(y_dist/x_dist))
    dist = np.sqrt((x_dist)**2 + (y_dist)**2)
    d = distance.geodesic(meters = dist)
    loc_lat = d.destination(point=[start_lat,start_lon], bearing=bearing)[0]
    loc_lon = d.destination(point=[start_lat,start_lon], bearing=bearing)[1]
    return loc_lat, loc_lon, d

# define function to find diameter in meters of the error on the location
def error_diameter(new_array):
    min_idx = np.min(new_array[:,1])
    max_idx = np.max(new_array[:,1])
    difference = max_idx-min_idx
    diameter_m = difference*1000
    return diameter_m 