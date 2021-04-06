<br />

<h1 align="center">Visual Analytics: Bokeh</h1>
 <h4 align="center">Interactive dashboard on Property Sales in New York.</h4>

<br />

Welcome to the visual-analytics-bokeh repo!

This project consists of an interactive dashboard built with Python, using packages like pandas, bokeh and more. The data used for this dashboard is the [NYC Citywide Annualized Calendar Sales Update](https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu) by NYC OpenData, and the geospatial data is from [nycehs/NYC_geography](https://github.com/nycehs/NYC_geography) repository.

This project is made by Karin Meijvogel and Lea van den Heuvel, for the Minor Data Science at the Amsterdam University of Applied Sciences.

<br />

## Table of contents 📜
* [Installation guide](#installation-guide-%EF%B8%8F)
* [Sources](#sources-)

<img src="https://user-images.githubusercontent.com/57796369/113760605-b5cb3400-9716-11eb-90de-372db032332d.png" width="100%">

<br />

## Installation guide 🖱️
### Requirements
To run this project, you'll need to have Python installed, as well as several packages. Also, any code editor is useful, if you want to look into the code and see our data preparation process for yourself. The packages used for this project project are:

* Numpy
* Pandas
* Geopandas
* Statsmodels
* Matplotlib
* Seaborn
* Bokeh

### Install
If you want to try the dashboard out for yourself, you can clone the repository to your desired location. You should be in the folder containing the project folder.

```
git clone https://github.com/imkarin/visual-analytics-bokeh.git
```

Install the required pip modules if you don't have them in your environment yet:

```
pip install numpy
pip install pandas
...
```

Finally, when you've made sure you're in the project's parent directory, start the bokeh server:
```
bokeh serve --show visual-analytics-bokeh
```


A browser window will open, leading to http://localhost:5006/visual-analytics-bokeh. In your console, you'll see that the code is running and preparing the dashboard. After several seconds the dashboard will appear in your browser. Enjoy!

<br />

## Sources 🔎
### Property Sales NYC data
NYC Citywide Annualized Calendar Sales Update | NYC Open Data. (2020, 3 maart). NYC OpenData. https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu

### Geojson data
NYC_geography. (2020). [Software]. https://github.com/nycehs/NYC_geography

