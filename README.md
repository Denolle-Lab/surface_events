# Surface Events
Surface events such as Avalanches, rockfalls, debris flows, etc. are prevalent on the Cascade Volcanoes in the Pacific Northwest and are hazardous to the surrounding area. They are frequent, yet do not have the same automated analysis that earthquakes do. Using seismic data from the Pacific Northwest Seismic Network (PNSN), we aim to locate these events with a grid search model and derive the flow velocities and directions of these events using a doppler shift algorithm.

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


This project is licensed under the MIT Public License.

## Dependencies
Environment: 
conda env create environment.yml

Data: Data is from the Pacific Northwest Seismic Network. The catalog can be found here and the station and waveform data can be downloaded using pnwstore Client. 

