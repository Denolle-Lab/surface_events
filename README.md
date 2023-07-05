# Surface Events
Surface events such as Avalanches, rockfalls, debris flows, etc. are prevalent on the Cascade Volcanoes in the Pacific Northwest and are hazardous to the surrounding area. They are frequent, yet do not have the same automated analysis that earthquakes do. Using seismic data from the Pacific Northwest Seismic Network (PNSN), we aim to locate these events with a grid search model and derive the flow velocities and directions of these events using a doppler shift algorithm.

This project is licensed under the MIT Public License.

## Dependencies
Environment: 
conda env create environment.yml

Data: Data is from the Pacific Northwest Seismic Network. The catalog can be found here and the station and waveform data can be downloaded using pnwstore Client. 

<h2>Understanding the Repo</h2>
<table>
  <tr>
    <th>Filename(s)</th>
    <th>Directory</th>
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
    <td>NA/</td>
    <td>Simplest model of a grid search such as the one used in this workflow</td>
  </tr>
  <tr>  <td>curves_freq_data1262743Mt_Rainier.png,loc_direction3179368Mt_Rainier.png,wiggles1262743Mt_Rainier.png,curves_freq_data3179368Mt_Rainier.png,psd1262743Mt_Rainier.png,wiggles1267698Mt_Rainier.png,Event_Data.csv,velsMt_Hood.png,wiggles3179093Mt_St_Helens.png,heatmap1262743Mt_Rainier.png,velsMt_Rainier.png,wiggles3179368Mt_Rainier.png,loc_direction1262743Mt_Rainier.png,velsMt_St_Helens.png</td>
    <td>Poster_Figs/</td>
    <td>Figures made for AGU 2022 poster</td>
  </tr>
  <tr>
    <td>Surface_Event_Directivity_Updated_Rainier_fig_8.ipynb</td>
    <td> /</td>
    <td> runs workflow for Labeled Mount Rainier Events and makes figures like Figure 8</td>
  </tr>
  
  


