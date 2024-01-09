# Surface Events
Surface events such as Avalanches, rockfalls, debris flows, etc. are prevalent on the Cascade Volcanoes in the Pacific Northwest and are hazardous to the surrounding area. They are frequent yet do not have the same automated analysis as earthquakes. Using seismic data from the Pacific Northwest Seismic Network (PNSN), we aim to locate these events with a grid search model and derive the flow velocities and directions of these events using a Doppler shift algorithm. 



This project is licensed under the MIT Public License.

## Dependencies
Environment: 
```sh
conda env create environment.yml
```

Data: 
- Surface Event Seismic Data is from the Pacific Northwest Seismic Network. The catalog can be found [here](https://seismica.library.mcgill.ca/article/view/368) and the station and waveform data can be downloaded using pnwstore Client. 
- DEM data is obtained from the University of Washington and can be found [here](https://gis.ess.washington.edu/data/raster/tenmeter/). It is processed and prepared for the workflow in Data/DEM_data/Volc_Dem.ipynb
- Labels to certain events are obtained from Wes Thelen of the Cascade Volcano Observatory
https://docs.google.com/spreadsheets/d/1tickhlEZjjYVUwvWrW2tbsCKyLKZY1oLjJuoKuWfRGg/edit#gid=619751142
  

For each surface event, the workflow analysis workflow consists in:
1. Waveform download for each event on each volcano given the PNSN pick times of "su" events.
2. Data pre-processing to trim the data within 2-12 Hz and remove outliers.
3. phase picking using transfer-learned model (Ni et al, 2023)
4. centroid picking estimate on the envelope.
5. event location using 1D grid search
6. directivity measurements (velocity and direction) using Doppler effects and 
7. gathering of the data into a CSV data frame.

```
├── src
│   ├── old
│   │   ├── all old scripts from Francesca' 2022, 2023 work
│   ├── RedPy_Clustering
|   |   ├── Scripts for @Nicholas Smoczyk to pick
│   ├── mbf_elep_func.py
│   ├── utils.py
│   ├── ML_Picker_Benchmark.ipynb
├── data
│   ├── Redpy  
|   |   ├── CSV files for @Nicholas Smoczyk to pick
│   ├── events  
|   |   ├── event_ids_r.json
|   |   ├── event_ids_st.json
|   |   ├── labels_rainier.json
|   |   ├── labels_st_helens.json
|   |   ├── wes_event_ids_r.json
|   |   ├── wes_event_ids_st.json
│   ├── bb_elep_picks_su.csv
│   ├── mbf_elep_picks_su.csv
│   ├── geospatial  
|   |   ├── Mt_Adams/
|   |   ├── Mt_Baker/
|   |   ├── Mt_Hood/
|   |   ├── Mt_Rainier/
|   |   ├── Mt_St_Helens/
|   |   ├── Volc_Dem.ipynb
├── plot
│   ├── *pdf
│   ├── *png
├── READNE.md
├── .vscode
├── environment.yml
├── LICENSE.md
└── .gitignore
```


<h2>Understanding the Repo</h2>
<table>
  <tr>
    <th>Folder Filename(s) or Notebook</th>
    <th>Directory (if applicable)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>wes_events_r.json,wes_events_st.json</td>
    <td>Data/</td>
    <td>Starttimes of labeled events</td>
  </tr>
  <tr>
    <td>Equake_test_AK.ipynb,Equake_test_deep_local.ipynb</td>
    <td>Equake_Tests/</td>
    <td>Analysis run on earthquake to ensure that it fails</td>
  </tr>
  <tr>
    <td>Grid_Search_Model.ipynb</td>
    <td> </td>
    <td>Simplest model of a grid search such as the one used in this workflow</td>
  </tr>
  <tr>  
    <td>Poster_Figs/</td>
    <td> </td>
    <td>Figures made for AGU 2022 poster</td>
  </tr>
  <tr>
    <td>Surface_Event_Directivity_Updated_Rainier_fig_8.ipynb, 
    Surface_Event_Directivity_Updated_St_Helens_figs_8.ipynb, 
    Surface_event_Directivity_Updated_Hood.ipynb</td>
    <td> </td>
    <td> creates Wes_Labeled_FinaL_Figs </td>
  </tr>
  <tr>
    <td> Event_Analysis_figs_245.ipynb</td>
    <td> </td>
    <td> Creates time series of events, velocity distribution, 
      and location distribution at each volcano </td>
  </tr>
  <tr>  
    <td>Wes_Labeled_FinaL_Figs/</td>
    <td>Analysis_Data</td>
    <td>Figures like fig 8 for all labeled events consisting of waveforms, label, and directivity</td>
  </tr>
  <tr>  
    <td>Wes_Events_Rainier_Figs, Wes_Events_St_Figs/</td>
    <td>Analysis_Data</td>
    <td>Analysis Figures of the labeled events</td>
  </tr>
  <tr>  
    <td>All_Events_Rainier_Figs, All_Event_St_Helens_Figs, All_Events_Hood_Figs</td>
    <td>Analysis_Data</td>
    <td>Cosine curve fit the frequency versus azimuth plot for all events at the respective volcano</td>
  </tr>
  
  
  
  


