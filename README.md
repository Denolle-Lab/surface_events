# Surface Events
Surface events such as Avalanches, rockfalls, debris flows, etc. are prevalent on the Cascade Volcanoes in the Pacific Northwest and are hazardous to the surrounding area. They are frequent, yet do not have the same automated analysis that earthquakes do. Using seismic data from the Pacific Northwest Seismic Network (PNSN), we aim to locate these events with a grid search model and derive the flow velocities and directions of these events using a doppler shift algorithm.

This project is licensed under the MIT Public License.

## Dependencies
Environment: 
conda env create environment.yml

Data: 
- Surface Event Seismic Data is from the Pacific Northwest Seismic Network. The catalog can be found [here](https://seismica.library.mcgill.ca/article/view/368) and the station and waveform data can be downloaded using pnwstore Client. 
- DEM data is obtained from the University of Washington and can be found [here](https://gis.ess.washington.edu/data/raster/tenmeter/). It is processed and prepared for the workflow in Data/DEM_data/Volc_Dem.ipynb
- Labels to certain events are obtained from Wes Thelen of the Cascade Volcano Observatory
https://docs.google.com/spreadsheets/d/1tickhlEZjjYVUwvWrW2tbsCKyLKZY1oLjJuoKuWfRGg/edit#gid=619751142
  

<h2>Understanding the Repo</h2>
<table>
  <tr>
    <th>Folder or Filename(s)</th>
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
    <td> runs workflow for labeled events on respective volcano and produces figures like Figure 8</td>
  </tr>
  <tr>
    <td> Event_Analysis_figs_2,4,5.ipynb</td>
    <td> </td>
    <td> Creates time series of events, velocity distribution, 
      and location distribution at each volcano </td>
  </tr>
  <tr>  
    <td>Wes_Labeled_FinaL_Figs/</td>
    <td>Analysis_Data</td>
    <td>Figures like fig 8 for all labeled events consisting of wwaveforms, label, and directivity</td>
  </tr>
  <tr>  
    <td>Wes_Events_Rainier_Figs,Wes_Events_St_Figs/</td>
    <td>Analysis_Data</td>
    <td>Analysis Figures of the labeled events</td>
  </tr>
  <tr>  
    <td>All_Events_Rainier_Figs,All_Event_St_Helens_Figs,All_Events_Hood_Figs</td>
    <td>Analysis_Data</td>
    <td>Cosine curve fit to the frequency versus azimuth plot for all events at the respesctive volcano</td>
  </tr>
  
  
  
  


