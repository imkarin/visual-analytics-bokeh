# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:49:48 2021

@author: Gebruiker
"""

# Packages
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import datetime
from datetime import date
from shapely import wkt
from math import pi
import math
from ast import literal_eval

import statsmodels.formula.api as smf
import statsmodels.api as sm

import scipy.special
from bokeh.layouts import gridplot, row, column, widgetbox
from bokeh.plotting import figure, output_file, show, gmap
from bokeh.io import curdoc
from bokeh.models import (CategoricalColorMapper, CheckboxGroup, ColumnDataSource, CustomJS, Dropdown, Select,
                          DatetimeTickFormatter, DateRangeSlider, DateSlider, Grid, LinearAxis, MultiPolygons, 
                          Plot, GeoJSONDataSource, GMapOptions, GeoJSONDataSource)
from scipy import interpolate
from bokeh.models.annotations import Title
from bokeh.palettes import Category20c, Set3
from bokeh.transform import cumsum
from bokeh.tile_providers import CARTODBPOSITRON, get_provider


#%%
#GEODATA
# Borough
geo_bor = gpd.read_file('https://raw.githubusercontent.com/nycehs/NYC_geography/master/borough.geo.json')
print(geo_bor.head())

# CD
geo_cd = gpd.read_file('https://raw.githubusercontent.com/nycehs/NYC_geography/master/CD.geo.json')
print(geo_cd.head())
print(geo_cd.GEONAME.value_counts()) # Duplicated GEONAME (SI en Queens) veranderen in Queens: dit lijkt eigenlijk een vd naamloze CDs in Queens te zijn.
geo_cd.loc[(geo_cd.GEONAME == 'St. George and Stapleton (CD1)') & (geo_cd.BOROUGH == 'Queens'), 'GEONAME'] = 'Queens (CD80)'

# UHF42
geo_uhf42 = gpd.read_file('https://raw.githubusercontent.com/nycehs/NYC_geography/master/UHF42.geo.json') 
print(geo_uhf42.head())

# UHF34
geo_uhf34 = gpd.read_file('https://raw.githubusercontent.com/nycehs/NYC_geography/master/UHF34.geo.json')
geo_uhf34.loc[0, 'GEONAME'] = 'Central Park' # 0 had geen naam
print(geo_uhf34.head())

# NTA
geo_nta = gpd.read_file('https://raw.githubusercontent.com/nycehs/NYC_geography/master/NTA.geo.json')
print(geo_nta.head())

#%%
# NY DATA: INLADEN, OPSCHONEN EN BEWERKEN
# Data requesten - Alleen properties waar 1 residential unit is verkocht (niet meerdere units tegelijk)
url = "https://data.cityofnewyork.us/resource/w2pb-icbu.json?$limit=350000&residential_units=1&commercial_units=0"
r  = requests.get(url)
json_data = r.json()
psales = pd.DataFrame(json_data) # Dataframe maken
print(psales.head(3))
print(psales.info())

# Omzetten naar juiste dtype
int_values = ['zip_code', 'residential_units', 'commercial_units', 'total_units', 'year_built', 'tax_class_at_time_of_sale',
              'community_board', 'council_district', 'census_tract', 'bin', 'bbl']
remove_comma_int_values = ['land_square_feet', 'gross_square_feet', 'sale_price']
float_values = ['latitude', 'longitude']

# Int values toch naar float geconverteerd omdat NaN values niet als int kunnen. Later cleanen en naar int.
psales[int_values] = psales[int_values].astype(float)
psales[float_values] = psales[float_values].astype(float)
# Komma's verwijderen en converteren naar int
psales[remove_comma_int_values] = psales[remove_comma_int_values].replace(',', '', regex=True)\
                                                                 .replace('- 0', '0', regex=True)\
                                                                 .astype(float)
# Date values
psales['sale_date'] = pd.to_datetime(psales['sale_date'])
# Drop apartment_number
psales = psales.drop(columns=['apartment_number'])

# Nieuwe df maken
woningen = psales.copy()
print(woningen.shape)

# Rijen verwijderen waar salesprijs mist/0 is en gekke waarden verwijderen
woningen = woningen.loc[woningen['sale_price'] > 0]

# building_class_at_time_of alleen letters eruit halen en nieuwe var maken
woningen['building_class_at_time_of'].unique()
woningen['building_class_at_time_of_num'] = woningen['building_class_at_time_of'].str[0]

# Rijen verwijderen waar latitude/longitude mist
woningen = woningen[woningen[['latitude', 'longitude']].notna().all('columns')]
woningen.shape

# Duplicates eruit halen
woningen = woningen.drop_duplicates(keep='first')
woningen.shape

# Kolom met maand van aankoop, eerste dag vd maand
woningen['sale_date_fixed'] = woningen.loc[:, 'sale_date'].apply(lambda x: x.replace(day=1))
print(woningen[['sale_date', 'sale_date_fixed']].head())

# Verschil in maanden kolom maken
today = datetime.datetime.today()
woningen['mon_diff'] = woningen['sale_date_fixed'].apply(lambda x: (today.year - x.year) * 12 + (today.month - x.month))

# Jaar van verkoop kolom maken
woningen['year_sold'] = woningen.sale_date.dt.year.astype(str)

# Bekijk hoe vaak elke woningklasse wordt genoemd
print(woningen['building_class_at_time_of_num'].value_counts())
# Alles wat niet A en R is samenvoegen tot Overig _____________________________________________________________________
woningen['class_fixed'] = np.where(woningen['building_class_at_time_of_num'].isin([ "A", "R"]), woningen['building_class_at_time_of_num'], "Other")
print(woningen.shape)

#Onderzoek doen naar de prijzen
filter_price = woningen['sale_price'] < woningen['sale_price'].median()
fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = woningen[filter_price],
            x = "sale_price")
plt.title("Histogram voor prijs waar de prijs lager is dan de mediaan")
plt.xlabel("Prijs")
#plt.show() # Er zijn veel lage, rare waarden, de halen we eruit

grens = 100000/9
woningen = woningen.loc[woningen['sale_price'] > grens]
print(woningen.shape)

# Andere, hoge outliers bekijken
fig, ax = plt.subplots(figsize = (4,8))
ax.boxplot(woningen["sale_price"])
plt.title("Boxplot\nHuizenprijzen")
#plt.show() # Waar komen al deze uitschieters vandaan?

# Die allerhoogste waarden verwijderen
woningen = woningen[woningen["sale_price"] < woningen["sale_price"].nlargest(3).iloc[-1]]
# Bovengrens outliers berekenen en histogram bekijken
Q1 = woningen.sale_price.quantile(0.25)
Q3 = woningen.sale_price.quantile(0.75)
IQR = Q3-Q1
grens = Q3 + IQR*1.5 

fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = woningen[woningen['sale_price'] < grens*2],
            x = "sale_price",
            kde=True)
plt.title("Histogram voor prijs waar de prijs lager is dan 2 maal de outlier grens")
plt.xlabel("Prijs")
#plt.show() # Naar rechts skewed, wat als we een transformatie toepassen?

# Wortel transformatie prijs
woningen['sale_price_sqrt'] = np.sqrt(woningen['sale_price'])
fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = woningen,
            x = "sale_price_sqrt",
            kde=True)
plt.title("Histogram voor de wortel van prijs")
plt.xlabel("Wortel van de prijs")
#plt.show()
# Nogsteeds redelijk skewed

# Log transformatie prijs
woningen['sale_price_log'] = np.log(woningen['sale_price'])
fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = woningen,
            x = "sale_price_log",
            kde=True)
plt.title("Histogram voor de log van prijs")
plt.xlabel("Log van de prijs")
#plt.show() # Ziet er beter uit, maar de vorm is nog een beetje gek

#%%
# Merge zodat elk huis zn borough naam heeft
geo_bor['BoroCode'] = geo_bor['BoroCode'].astype(str)
woningen = woningen.merge(geo_bor[['BoroCode', 'BoroName']], left_on='borough', right_on = 'BoroCode')\
                    .drop(columns=['BoroCode', 'borough'])\
                    .rename(columns={'BoroName' : 'bor_name'})
print(woningen.shape)
print(woningen.head())

#%%
# Bekijken of de rare normaalverdeling van prijs door boroughs komt
fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = woningen,
             x = "sale_price_log",
             hue = "bor_name",
             element ="step",
             kde=True)
plt.title("Histogram voor de log van de prijs\nPer borough")
plt.xlabel("Log prijs")
#plt.show()

# Hoe verloopt de prijs over de tijd per Borough?
fig, ax = plt.subplots(figsize = (8,8))
sns.lineplot(data = woningen,
             x = 'sale_date_fixed',
             y = 'sale_price_log',
             hue = "bor_name",
             alpha = 0.5)
plt.title("Lineplot voor log van de prijs over de tijd\nVerdeelt over Boroughs")
#plt.show()

#%%
# BOKEH: 1D visualisaties ---------------------------------------------------------------------------------------------------------------------
# PALETTE!
palette1 = ["#98dfea","#25283d","#3988c9","#8f3985","#efd9ce", "#eaac8b", "#d93057", "#b56576","#6d597a"]

# Histogram: log prijs
hist_prijs, edges_prijs = np.histogram(woningen['sale_price_log'], bins = 100) #Om de hist te maken
p_hist_prijs = figure(height=600,
                      title = "Histogram log sale price",
                      toolbar_location = None,
                      tools = "",
                      x_axis_label = "Log price ($)",
                      y_axis_label = "Count")
p_hist_prijs.quad(top=hist_prijs, bottom=0, left=edges_prijs[:-1], right=edges_prijs[1:], 
                  fill_color=palette1[2], line_color="white", alpha=0.7)

#%%
# Boxplots: prijs per borough
# find the quartiles and IQR for each category
groups = woningen[['sale_price', 'bor_name']].groupby('bor_name')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr
cats = q1.index.tolist() # Volgorde van de boxplots moet hetzelfde zijn als in de dataframe

# find the outliers for each category
def outliers(group):
    cat = group.name
    return group[(group.sale_price > upper.loc[cat]['sale_price']) | (group.sale_price < lower.loc[cat]['sale_price'])]['sale_price']
out = groups.apply(outliers).dropna()

# prepare outlier data for plotting, we need coordinates for every outlier.
if not out.empty:
    outx = list(out.index.get_level_values(0))
    outy = list(out.values)

p_boxplot = figure(height=600,
                   title = "Boxplot sale price per borough",
                   y_axis_label = "Sale price ($)",
                   tools="",
                   x_range=cats, 
                   toolbar_location=None)

# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.sale_price = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'sale_price']),upper.sale_price)]
lower.sale_price = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'sale_price']),lower.sale_price)]

# stems
p_boxplot.segment(cats, upper.sale_price, cats, q3.sale_price, line_color="black", line_alpha=0.6)
p_boxplot.segment(cats, lower.sale_price, cats, q1.sale_price, line_color="black", line_alpha=0.6)

# boxes
p_boxplot.vbar(cats, 0.7, q2.sale_price, q3.sale_price, fill_color=palette1[2], line_color="black", line_alpha=0.6)
p_boxplot.vbar(cats, 0.7, q1.sale_price, q2.sale_price, fill_color=palette1[-1], line_color="black", line_alpha=0.6)

# whiskers (almost-0 height rects simpler than segments)
p_boxplot.rect(cats, lower.sale_price, 0.2, 0.01, line_color="black", line_alpha=0.6)
p_boxplot.rect(cats, upper.sale_price, 0.2, 0.01, line_color="black", line_alpha=0.6)

# outliers
if not out.empty:
    p_boxplot.circle(outx, outy, size=6, color='#6ebdff', fill_alpha=0.2)

p_boxplot.xgrid.grid_line_color = None
p_boxplot.ygrid.grid_line_color = "white"
p_boxplot.grid.grid_line_width = 2




#%%
# Piechart: Hoe veel huizen er zijn gekocht in welke borough
# Tellen per borough
borough_count = pd.DataFrame(woningen['bor_name'].value_counts()).reset_index() # Krijg het aantal verkochte huizen per borough
borough_count.rename(columns = {"bor_name": "count", # Rename columns
                       "index": "bor_name"}, 
                     inplace = True)

borough_count['angle'] = borough_count['count']/borough_count['count'].sum() * 2*pi # Bereken hoe groot het stuk taart moet zijn

borough_count = borough_count.append({'bor_name': 'NYC Total', 'count': 0, 'angle':0}, ignore_index=True)
borough_count = borough_count.sort_values('bor_name')
borough_count['color'] = palette1[2:len(borough_count)+2] # Elke borough een kleur geven
print(borough_count.head(10))

# ColumnDataSource maken
source_pie = ColumnDataSource(data = {"x": borough_count['bor_name'],
                                      "count": borough_count['count'],
                                       "y": borough_count['angle'],
                                       "color": borough_count['color']})

p_pie = figure(plot_height=350, plot_width = 400, # Initialiseer figure
               title="Amount of sold houses per per borough", toolbar_location=None,
               tools="hover", tooltips="@x: @count", x_range=(-0.5, 0.7))

p_pie.wedge(source = source_pie, x=0, y=1, radius=0.4, # Maak piechart
        start_angle=cumsum('y', include_zero=True), end_angle=cumsum('y'),
        line_color="white", fill_color='color', legend_field='x', alpha=0.8)

p_pie.axis.axis_label=None # Geen lelijke achtergronden en assen
p_pie.axis.visible=False
p_pie.grid.grid_line_color = None

# Interactief maken, kies je borough
LABELS = list(woningen.bor_name.unique().astype(str)) # Tekst van checkboxen
checkbox_button_group = CheckboxGroup(labels=LABELS, active = [0, 1, 2, 3, 4]) # Initialiseer checkboxen, zet ze allemaal aan

#show(checkbox_button_group)

def callback_pie(attr, old, new):
    bor_active = [checkbox_button_group.labels[i] for i in checkbox_button_group.active] # Lijst met alle actieve knoppen
    borough_temp = borough_count[borough_count["bor_name"].isin(bor_active)][['bor_name', 'count', 'color']] # Filter de aangeklikte boroughs
    borough_temp['angle'] = borough_temp['count']/borough_temp['count'].sum() * 2*pi # Grootte moet opnieuw berekend worden
    
    # Nieuwe ColumnDataSource maken
    source_pie_new = ColumnDataSource(data = {"x": borough_temp['bor_name'],
                                          "count": borough_temp['count'],
                                      "y": borough_temp['angle'],
                                      "color": borough_temp['color']})
    source_pie.data = dict(source_pie_new.data)

checkbox_button_group.on_change('active', callback_pie) # Knoppem en callback koppelen

p_pie_col = row(checkbox_button_group, p_pie)


#%%
# Piechart: Hoe veel er van elke klasse is verkocht
# Tellen per klasse
class_count = pd.DataFrame(woningen['class_fixed'].value_counts()).reset_index() # Krijg het aantal verkochte huizen per klasse
class_count.rename(columns = {"class_fixed": "count", # Rename columns
                    "index": "class"},
                    inplace = True)
class_count['angle'] = class_count['count']/class_count['count'].sum() * 2*pi # Bereken hoe groot het stuk taart moet zijn
class_count['color'] = palette1[0:len(class_count)] # Elke klasse een kleur geven
print(class_count.head())


# ColumnDataSource maken
source_pie_class = ColumnDataSource(data = {"x": class_count['class'],
                                            "count": class_count['count'],
                                            "y": class_count['angle'],
                                            "color": class_count['color']})
# Initialiseer figure
p_pie_class = figure(plot_height=350, plot_width = 400,
title="Amount of sold houses per building class", toolbar_location=None,
tools="hover", tooltips="@x: @count", x_range=(-0.5, 0.7))
# Voeg taart toe
p_pie_class.wedge(source = source_pie_class, x=0, y=1, radius=0.4,
start_angle=cumsum('y', include_zero=True), end_angle=cumsum('y'),
line_color="white", fill_color='color', legend_field='x')


p_pie_class.axis.axis_label=None # Geen lelijke achtergronden en assen
p_pie_class.axis.visible=False
p_pie_class.grid.grid_line_color = None


# Interactief maken, kies je klasse
LABELS_pie_class = list(woningen['class_fixed'].unique().astype(str)) # Tekst van checkboxen
checkbox_pie_class = CheckboxGroup(labels=LABELS_pie_class, active = [*range(len(LABELS_pie_class))]) # Initialiseer checkboxen, zet ze allemaal aan

#show(checkbox_pie_class)

def callback_pie_class(attr, old, new):
    class_active = [checkbox_pie_class.labels[i] for i in checkbox_pie_class.active] # Lijst met alle actieve knoppen
    class_temp = class_count[class_count["class"].isin(class_active)][['class', 'count', 'color']] # Filter de aangeklikte boroughs
    class_temp['angle'] = class_temp['count']/class_temp['count'].sum() * 2*pi # Grootte moet opnieuw berekend worden

    # Nieuwe ColumnDataSource maken
    source_pie_class_new = ColumnDataSource(data = {"x": class_temp['class'],
                                                    "count": class_temp['count'],
                                                    "y": class_temp['angle'],
                                                    "color": class_temp['color']})
    source_pie_class.data = dict(source_pie_class_new.data)

checkbox_pie_class.on_change('active', callback_pie_class) # Knoppen en callback koppelen

column_pie_class = column(checkbox_pie_class, p_pie_class)

# Op basis van de piechart kiezen we ervoor om Overig te verwijderen
woningen = woningen[woningen['building_class_at_time_of_num'].isin([ "A", "R"])]
print(woningen.shape)



#%%
# 2D VISUALISATIES ---------------------------------------------------------------------------------------------------------------------------
# Lijnplot: Gemiddelde prijs per maand per klasse
# Dataframe maken voor plot
avg_bor_month = woningen.groupby(['bor_name', 'sale_date_fixed', 'building_class_at_time_of_num'])['sale_price_log'].mean()\
    .to_frame()\
        .reset_index()
# Voeg values toe voor NYC Total (gemiddelde over hele stad)
avg_month_nyc = woningen.groupby(['sale_date_fixed', 'building_class_at_time_of_num'])['sale_price_log'].mean().to_frame().reset_index()
avg_month_nyc['bor_name'] = 'NYC Total'
print(avg_month_nyc.head(10))
# Toevoegen aan de avg_bor_month
avg_bor_month = avg_bor_month.append(avg_month_nyc, ignore_index=True)
print(avg_bor_month.tail())
klasses = avg_bor_month['building_class_at_time_of_num'].unique()
print(klasses)

###################
# Plot maken
title = Title()
title.text = 'Average sale price per month, per building class in Manhattan'
p_line_prijs = figure(x_axis_label='Date', y_axis_label='Mean log sale price ($)',
            title=title, tools = "", toolbar_location = None, y_range = [11.2, 16.5])

# Datasource met coordinaten voor de lijnen
manhattan = (avg_bor_month['bor_name'] == 'Manhattan')

source1 = ColumnDataSource({
    'xs' : [avg_bor_month[manhattan & (avg_bor_month['building_class_at_time_of_num'] == klasse)]['sale_date_fixed'] 
           for klasse in klasses],
    'ys' : [avg_bor_month[manhattan & (avg_bor_month['building_class_at_time_of_num'] == klasse)]['sale_price_log'] 
           for klasse in klasses],
    'label' : klasses,
    'palette' : palette1[:len(klasses)]
})

# Plot de lijnen
p_line_prijs.multi_line('xs', 'ys', source=source1, line_width=2, line_color='palette', legend_field="label")

# Datums geroteerd op x-as
p_line_prijs.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"])
p_line_prijs.xaxis.major_label_orientation = 45


# Dropdownmenu maken
menu_options = list(woningen.bor_name.unique())
dropdown_line_prijs = Select(options = menu_options, value = menu_options[0], title = 'Select borough')

# Callback maken
def select_borough(attr, old, new):
    title.text = 'Gemiddelde verkoopprijs per maand per gebouwklasse in ' + new
    source1.data = {
        'xs' : [avg_bor_month[(avg_bor_month['bor_name'] == new)
                             & (avg_bor_month['building_class_at_time_of_num'] == klasse)]['sale_date_fixed']
                           for klasse in klasses],
        'ys' : [avg_bor_month[(avg_bor_month['bor_name'] == new)
                             & (avg_bor_month['building_class_at_time_of_num'] == klasse)]['sale_price_log']
                           for klasse in klasses],
        'label' : klasses,
        'palette' : palette1[:len(klasses)]
    }
dropdown_line_prijs.on_change('value', select_borough)

# Layout maken om later toevoegen aan de root
p_line_prijs_col = column(dropdown_line_prijs, p_line_prijs)

#%%
# Lineplot: Gemiddelde verkoopprijs per borough -----------------------------------------------------------
avg_bors = avg_bor_month.groupby(['bor_name', 'sale_date_fixed'])['sale_price_log'].mean().reset_index()
bors = avg_bors['bor_name'].unique()

# Plot maken
title=Title()
title.text = 'Average sale price per month per borough'
p_line_prijs_bor = figure(x_axis_label='Date', y_axis_label='Mean log price ($)',
                          title = title, tools = "", toolbar_location = None, y_range = [11.2, 16.5])

# Datasource met coordinaten voor de lijnen
manhattan = (avg_bor_month['bor_name'] == 'Manhattan')

source2 = ColumnDataSource({
    'xs' : [avg_bors[avg_bors['bor_name'] == bor]['sale_date_fixed'] for bor in bors],
    'ys' : [avg_bors[avg_bors['bor_name'] == bor]['sale_price_log'] for bor in bors],
    'label' : bors,
    'palette' : palette1[2:len(bors)+2]
})

# Plot de lijnen
p_line_prijs_bor.multi_line('xs', 'ys', source=source2, line_width=2, line_color='palette', legend_field='label')

# Datums geroteerd op x-as
p_line_prijs_bor.xaxis.formatter=DatetimeTickFormatter(
                    days=["%d %B %Y"],
                    months=["%d %B %Y"],
                    years=["%d %B %Y"])
p_line_prijs_bor.xaxis.major_label_orientation = 45

# Date range slider voor x-as
date_range_slider = DateRangeSlider(value=(date(2016, 1, 1), date(2019, 12, 31)),
                                    start=date(2016, 1, 1), end=date(2019, 12, 31))

def change_date(attr, old, new):
    p_line_prijs_bor.x_range.start = new[0]
    p_line_prijs_bor.x_range.end = new[1]
    
date_range_slider.on_change('value', change_date)
#show(p_line_prijs_bor)
# Layout maken en toevoegen aan de root
p_line_prijs_bor_col = column(date_range_slider, p_line_prijs_bor)

#%%
# Scatterplot: Prijs tegenover square feet ------------------------
# De dataframe
columns = ['sale_price', 'gross_square_feet', 'bor_name', 'sale_date_fixed']
woningen_sqfeet = woningen[(woningen['gross_square_feet'] > 0) & (woningen['gross_square_feet'] < 20000)
                              & (woningen['sale_price'] > 30000)][columns]
palette3 = ["#ef476f","#ffd166","#06d6a0","#118ab2","#073b4c"] # Kleurtjes

# Plot maken
p_scatter_prijs_sqf = figure(height=600, width=900,
                             x_axis_label='Gross square feet', 
                             y_range = [-500000, 10000000],
                             x_range= [0, 8000],
                             y_axis_label='Sale price ($)',
                             title='Saleprices and surface area per borough',
                             tools = "pan,wheel_zoom,box_zoom")

# Data source: de dataframe
source3 = ColumnDataSource(woningen_sqfeet)

# Color mapper/legenda
LABELS = woningen_sqfeet['bor_name'].unique().tolist()
color_mapper = CategoricalColorMapper(factors=LABELS, palette=palette1[0:len(LABELS)])

# Plot de lijnen
p_scatter_prijs_sqf.circle('gross_square_feet', 'sale_price', source=source3, alpha=0.6, size=3,
          color={'field':'bor_name', 'transform': color_mapper},
          legend_field='bor_name')

# Checkboxes boroughs
checkboxes = CheckboxGroup(labels=LABELS, active=[0, 1, 2, 3, 4])
checkbox_mappings = {}

for number, label in enumerate(LABELS):
    checkbox_mappings[number] = label

def update_scatter(attr, old, new):
    # Houd axes hetzelfde
    p_scatter_prijs_sqf.x_range.start = 0
    p_scatter_prijs_sqf.x_range.end = 8000
    
    p_scatter_prijs_sqf.y_range.start = -500000
    p_scatter_prijs_sqf.y_range.end = 10000000
    
    # monthslider values
    slider_month = pd.Timestamp(monthslider.value, unit='ms').month
    slider_year = pd.Timestamp(monthslider.value, unit='ms').year
    
    # Filter dataframe en gebruik deze als source
    source3.data = woningen_sqfeet[(woningen_sqfeet['bor_name'].isin([checkbox_mappings[active] for active in checkboxes.active]))
                                   & 
                                   (woningen_sqfeet['sale_date_fixed'].dt.month == slider_month)
                                   &
                                   (woningen_sqfeet['sale_date_fixed'].dt.year == slider_year)]
checkboxes.on_change('active', update_scatter)

# Slider voor maand
monthslider = DateSlider(title='Month', value=date(2016, 1, 1),
                    start=date(2016, 1, 1), end=date(2019, 12, 31))
monthslider.on_change('value', update_scatter)

# Layout maken en toevoegen aan de root
p_scatter_prijs_sqf_row = row(column(checkboxes, monthslider), p_scatter_prijs_sqf)



#%%
# DATA VOOR REGRESSIEMODEL ----------------------------------------------------------------------------
# We kiezen ervoor om 2 eruit te halen en daar regressie voor te doen
test = woningen[woningen['bor_name']=="Bronx"].copy()
test.isna().sum()

# Boxplot prijs bekijken
fig, ax = plt.subplots(figsize = (4,7))
sns.boxplot(data = test, y = "sale_price_log")
plt.title("Boxplot voor log price")
#plt.show()

# Is er verschil voor de woningklasses?
Q1 = test.sale_price_log.quantile(0.25)
Q3 = test.sale_price_log.quantile(0.75)
IQR = Q3-Q1
ondergrens = Q1 - IQR*1.5 #Grenzen voor outliers toevoegen aan plot
bovengrens = Q3 + IQR*1.5 

fig, ax = plt.subplots(figsize = (8,8))
sns.boxplot(data = test, y = "sale_price_log", x = 'building_class_at_time_of_num')
plt.title("Boxplot log sale price")
plt.axhline(ondergrens)
plt.axhline(bovengrens)
#plt.show()

test = test[(test['sale_price_log']> ondergrens) & (test['sale_price_log']<bovengrens)]
test.shape

fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(data = test.sample(frac = 0.3, random_state = 100),
             x = 'mon_diff',
             y = 'sale_price_log',
             hue = "building_class_at_time_of_num",
             alpha = 0.5)
plt.title("Scatterplot voor log van de prijs over de tijd\nVerdeelt over woningklasse")
#plt.show()

#%%
# BOKEH: Scatterplot: tijd vs log prijs (onderbouwing voor regressiemodel?)
# De data
columns = ['sale_date_fixed', 'sale_price_log', 'council_district']
df_scatter_regr = test[columns].copy()
df_scatter_regr['council_district'] = df_scatter_regr['council_district'].astype(str)
source_scatter_regr = ColumnDataSource(df_scatter_regr)

# Plot maken
p_scatter_regr = figure(tools = "", toolbar_location = None,
                        height=600, width=900,
                        x_axis_label='Date', 
                        y_axis_label='Log sale price ($)',
                        title='Scatterplot for sale prices over time, per council district')

# X-as datum
p_scatter_regr.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"])
p_scatter_regr.xaxis.major_label_orientation = 45

# Color mapper/legenda
labels = [i for i in df_scatter_regr['council_district'].unique()]
color_mapper_regr = CategoricalColorMapper(factors = labels, palette = Set3[len(labels)])

# Plot de lijnen
p_scatter_regr.circle(x = 'sale_date_fixed', y = 'sale_price_log', source = source_scatter_regr, alpha=0.6, size=3
                      , color = {"field": "council_district", "transform": color_mapper_regr},
                      legend='council_district'
                      )


# Layout maken en toevoegen aan de root
p_scatter_regr_row = row(p_scatter_regr)

#%%
# REGRESSIE ------------------------------------------------------------------

# Welke variabelen toevoegen aan regressiemodel?
print(test.columns)
# neighborhood
# zip_code
#! council_district
#
#%%
# Neighbourhood onderzoeken
print(test.neighborhood.unique())

#%%
# Zip code onderzoeken
#print(test.zip_code.unique())
test["zip_code_fixed"] = (test['zip_code']/10).astype(int).astype("category")
#print(test["zip_code_fixed"].unique())
print(test.value_counts("zip_code_fixed"))

fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(data = test.sample(frac = 0.3, random_state = 100),
             x = 'mon_diff',
             y = 'sale_price_log',
             hue = "zip_code_fixed",
             alpha = 0.5)
plt.title("Lineplot voor log van de prijs over de tijd\nVerdeelt over zipcode")
#plt.show() 

# Als ik dit toevoeg moet 1080 er wel uit

#%%
# Council district onderzoeken
print(test.council_district.unique())
print(test.council_district.dtype) 
# Veranderen naar category
test['council_district'] = test['council_district'].astype("category")
print(test.value_counts("council_district"))

fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(data = test.sample(frac = 0.3, random_state = 100),
             x = 'mon_diff',
             y = 'sale_price_log',
             hue = "council_district",
             alpha = 0.5)
plt.title("Lineplot voor log van de prijs over de tijd\nVerdeelt over council district")
#plt.show() 

# Dit ziet er beter uit dan zip_code

#%%

print(test[['mon_diff', 'council_district', 'building_class_at_time_of_num']].corr())

#%%
# Regressiemodel maken en bekijken
formula = "sale_price_log ~ mon_diff  + C(building_class_at_time_of_num) + council_district"
# Nummeriek:mon diff
# Categories: Building class en council district
results = smf.ols(formula, data = test).fit()
summary  = results.summary()
print(summary)

#%%
# Model valideren
# QQplot residu
res = results.resid # residuals
fig = sm.qqplot(res, line = "45")
plt.title("QQplot Y = log prijs")
#plt.show()

# Histogram residu
fig, ax = plt.subplots(figsize = (8,8))
sns.histplot(res, kde = True)
plt.title("Histogram Y = log prijs")
#plt.show()

# Res vs pred
fig, ax = plt.subplots(figsize = (8,8))
fitted_vals = results.predict()
sns.regplot(x=fitted_vals, y=res, lowess=True, line_kws={'color': 'red'})
plt.title("Fitted vals vs. residuals, Y = log prijs")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
#plt.show()

#%%
# BOKEH: Regressie visualisaties -----------------------------------------------------
# Histogram van residu
hist_res, edges_res = np.histogram(res, bins = 90) #Om de hist te maken
# Daatframe maken
source_hist = ColumnDataSource(data = {"count": hist_res,
                                       "left": edges_res[:-1],
                                       "right": edges_res[1:]})
# Plotten
p_hist_res = figure(align='end',
                    height=600,
                    title = "Histogram residuals",
                    toolbar_location = None,
                    tools = "", 
                    x_range=(-2, 2),
                    x_axis_label = "Residuals",
                    y_axis_label = "Count")

p_hist_res.quad(bottom=0, 
                top = 'count', 
                left = 'left', 
                right = 'right', 
                source = source_hist,
                fill_color=palette1[-1],
                alpha=0.7,
                #hover_fill_alpha=0.7,
                #hover_fill_color='blue',
                line_color='white')

#%%
# Scatter: Res vs fitted
source_res_fitted = ColumnDataSource(data = {"x":fitted_vals,
                                             "y":res})
p_scatter_res_fitted = figure(height=600,
                              align='end',
                              title = "Scatterplot fitted values vs. residuals",
                              toolbar_location = None,
                              tools = "",
                              x_axis_label = "Fitted values",
                              y_axis_label = "Residuals")

p_scatter_res_fitted.circle(source = source_res_fitted, x = "x", y = "y", color=palette1[0]) # Datapunten

#%%
# Dropdownmenu om Borough te kiezen waarvoor je een regressiemodel maakt
dropdown_regressie = Select(title = "Select the borough you want to fit the regression model on", options = list(woningen['bor_name'].unique()))

# Callback om Borough aan te passen
def callback_regressie(attr, old, new):
    borough_dropdown = dropdown_regressie.value
    test = woningen[woningen['bor_name']==borough_dropdown] # Filter op borough
    test = test.dropna()
    test = test[test['building_class_at_time_of_num'].isin([ "A", "R"])]
    Q1 = test.sale_price_log.quantile(0.25)
    Q3 = test.sale_price_log.quantile(0.75)
    IQR = Q3-Q1
    ondergrens = Q1 - IQR*1.5 #Grenzen voor outliers toevoegen aan plot
    bovengrens = Q3 + IQR*1.5
    test = test[(test['sale_price_log']> ondergrens) & (test['sale_price_log']<bovengrens)] #Outliers verwijderen
    #print(test.shape)
    
    formula = "sale_price_log ~ mon_diff  + C(building_class_at_time_of_num) + council_district"
    # Nummeriek:mon diff
    # Categories: Building class en council district
    results = smf.ols(formula, data = test).fit()
    #summary  = results.summary()
    #print(summary)
    
    # Res vs fitted
    res_new = results.resid # residuals
    fitted_vals_new = results.predict()
    
    source_res_fitted_new = ColumnDataSource(data = {"x": fitted_vals_new,
                                                     "y": res_new})
    source_res_fitted.data = dict(source_res_fitted_new.data)
    
    # Hist res
    hist_res_new, edges_res_new = np.histogram(res_new, bins = 90) #Om de hist te maken
    source_hist_new = ColumnDataSource(data = {"count": hist_res_new,
                                           "left": edges_res_new[:-1],
                                           "right": edges_res_new[1:]})
    source_hist.data = dict(source_hist_new.data)

dropdown_regressie.on_change("value", callback_regressie)

#%%
########### DATA WONINGEN MERGEN MET GEODATA VOOR KAART --------------------------------------------------------------------------
# NORMALE MERGES
# Merge (on nta) met geo_nta (on NTAName)
woningen_geo = woningen.copy()
woningen_geo = woningen_geo.drop(columns='bor_name')  # bestaande bor_name van eerder weghalen omdat deze anders dubbel is

# Merge (on nta) met geo_nta (on NTAName)
woningen_geo = woningen_geo.merge(geo_nta[['NTAName', 'BoroName', 'geometry']], left_on='nta', right_on='NTAName')\
                    .drop(columns=['NTAName'])\
                    .rename(columns={'geometry' : 'nta_geometry', 'BoroName': 'bor_name', 'nta': 'nta_name'})


#%%
# Merge (on borough_name) met geo_bor (on BoroName)
woningen_geo = woningen_geo .merge(geo_bor[['BoroName', 'geometry']], left_on='bor_name', right_on='BoroName')\
                    .drop(columns=['BoroName'])\
                    .rename(columns={'geometry' : 'bor_geometry'})

woningen_geo.head(2)

#%% 
# SPATIALE JOINS
# Geodataframe maken van woningen
woningen_geo = gpd.GeoDataFrame(woningen_geo,
                                geometry=gpd.points_from_xy(woningen_geo.longitude, woningen_geo.latitude),
                                crs='EPSG:4326')

woningen_geo.head(2)

#%% 
# sjoinen met geo_cd
woningen_geo = gpd.sjoin(woningen_geo, geo_cd, op='within', lsuffix='', rsuffix='cd')\
                  .drop(columns=['index_cd', 'id', 'BOROUGH'])\
                  .merge(geo_cd[['geometry', 'GEONAME']], on='GEONAME', suffixes=('', '_cd'))\
                  .rename(columns={'GEONAME': 'cd_name', 'geometry_cd': 'cd_geometry', 'GEOCODE': 'cd_code'})

woningen_geo.shape

#%%
# sjoinen met geo_uhf34
woningen_geo = gpd.sjoin(woningen_geo, geo_uhf34, op='within', lsuffix='', rsuffix='uhf34')\
                  .drop(columns=['index_uhf34', 'OBJECTID', 'GEOCODE', 'BOROUGH'])\
                  .merge(geo_uhf34[['geometry', 'GEONAME']], on='GEONAME', suffixes=('', '_uhf34'))\
                  .rename(columns={'GEONAME': 'uhf34_name', 'geometry_uhf34': 'uhf34_geometry', 'UHF': 'uhf34_code'})

#%%
woningen_geo.shape

# Plotseling zijn er meer rows. De sjoin heeft dus meerdere rijen gecreëerd voor sommige adressen,
# m.a.w. sommige adressen komen in meerdere uhf34 districten voor. Dit kan alleen als de uhf34 districten overlappen.
dup_address = woningen_geo[woningen_geo.duplicated(subset=['address', 'sale_date'], keep=False)]
diff_uhf34 = woningen_geo[woningen_geo.duplicated(subset=['address', 'sale_date', 'uhf34_name'], keep=False)]

# Aantal adressen dat in 2 uhf34 districten valt:
(dup_address.shape[0] - diff_uhf34.shape[0]) / 2

# Toon waardes met zelfde adres/sale-date maar verschillend uhf34_name
outer_join = dup_address.merge(diff_uhf34, how = 'outer', indicator = True)
same_address_diff_uhf34 = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)

# We droppen de laatste duplicate (laatste ufh34_name)
rows_to_drop = same_address_diff_uhf34[same_address_diff_uhf34.duplicated(subset=['address', 'sale_date'], keep='last')]\
                            .index.to_list()

woningen_geo = woningen_geo[~woningen_geo.index.isin(rows_to_drop)]
woningen_geo.shape
# Dit ziet er beter uit.
    
    
#%%
# sjoinen met geo_uhf42
woningen_geo = gpd.sjoin(woningen_geo, geo_uhf42, op='within', lsuffix='', rsuffix='uhf42')\
                  .drop(columns=['index_uhf42', 'id', 'BOROUGH'])\
                  .merge(geo_uhf42[['geometry', 'GEONAME']], on='GEONAME', suffixes=('', '_uhf42'))\
                  .rename(columns={'GEONAME': 'uhf42_name', 'geometry_uhf42': 'uhf42_geometry', 'GEOCODE': 'uhf42_code'})

woningen_geo.shape


#%%
# Uiteindelijke geodataframe met kolommen op juiste volgorde:
kolommen_georderd = ['neighborhood', 'building_class_category',
                    'tax_class_as_of_final_roll', 'block', 'lot',
                    'building_class_as_of_final', 'address', 'zip_code',
                    'residential_units', 'commercial_units', 'total_units',
                    'land_square_feet', 'gross_square_feet', 'year_built',
                    'tax_class_at_time_of_sale', 'building_class_at_time_of', 'building_class_at_time_of_num',
                    'sale_price', 'sale_price_sqrt', 'sale_price_log', 'sale_date_fixed',
                    'sale_date', 'sale_price_sqrt', 'latitude', 'longitude', 'community_board',
                    'council_district', 'census_tract', 'bin', 'bbl',
                    'geometry',
                    'nta_name', 'nta_geometry',                        # <--- Vanaf hier kolommen opnieuw georderd
                    'bor_name', 'bor_geometry',
                    'cd_code', 'cd_name', 'cd_geometry',
                    'uhf34_code', 'uhf34_name', 'uhf34_geometry',
                    'uhf42_code', 'uhf42_name', 'uhf42_geometry']

woningen_geo = woningen_geo[kolommen_georderd]

woningen_geo.head()


#%%
########### POLYGON KAART MAKEN BOKEH ---------------------------------------------------
from bokeh.models import (ColorBar,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider, CategoricalColorMapper)
from bokeh.palettes import brewer
from bokeh.plotting import figure

# DATAFRAMES VOOR ELK TYPE DISTRICT (borough, cd, nta, uhf34, uhf42)

#%%
cols = ['sale_price', 'bor_name', 'nta_name', 'cd_name', 'uhf34_name', 'uhf42_name']
# Default: gemiddelde prijs per borough in 2019

print('Making default geojson:')
# loc: [rows binnen geselecteerde datums, kolommen]
default_jaar = 2019
default_geojson = gpd.GeoDataFrame(
                                    woningen_geo.loc[woningen_geo['sale_date_fixed'].dt.year == default_jaar, 
                                                     [*cols, 'bor_geometry']]
                                                .rename(columns={'bor_geometry': 'geometry'})\
                                                .groupby(['bor_name']).agg({'sale_price':'mean', 'geometry': 'first'}).reset_index()
                                  )

#%%
print('Converting to json for bokeh source: ')
geosource = GeoJSONDataSource(geojson = default_geojson.to_json())

#%%
print('Plotting polygon map: ')
# Define color palettes
palette = ["#dcedfa","#85beea","#54a5e3","#3988c9","#1e6bae","#18568c","#11406a","#122a40"]

# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 6000000)

# Define custom tick labels for color bar.
tick_labels = {'0': '0', '6000000': '6000000'}

# Create color bar.
color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 8,
                     width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal',
                     major_label_overrides = tick_labels)

# Create figure object.
title_polymap = Title()
title_polymap.text = 'Average sale price of houses in NYC per borough, 2016 - 2019'
p_polymap = figure(title = title_polymap,
           toolbar_location = 'below',
           tools ='pan, wheel_zoom, box_zoom, reset')
p_polymap.xgrid.grid_line_color = None
p_polymap.ygrid.grid_line_color = None


# Add patch renderer to figure.
boroughs = p_polymap.patches('xs','ys', source = geosource,
                   fill_color = {'field' :'sale_price',
                                 'transform' : color_mapper},
                   line_color = 'gray', 
                   line_width = 0.25, 
                   fill_alpha = 1)
# Create hover tool
p_polymap.add_tools(HoverTool(renderers = [boroughs],
                      tooltips = [('Place name','@bor_name'),
                               ('Average sale price', '@sale_price')]))


# Dropdownmenu maken (soort district) --------------------------------------------
menu_options = ['Borough', 'Community District', 'NTA', 'UHF34', 'UHF42']
dropdown_polymap = Select(options = menu_options, value = menu_options[0], title = 'Select type of district division')

# Dict voor de callback/dropdown voor districtverdeling
districten = {'Borough' : ['bor_geometry', 'bor_name'],
               'NTA' : ['nta_geometry', 'nta_name'],
               'Community District' : ['cd_geometry', 'cd_name'],
               'UHF34' : ['uhf34_geometry', 'uhf34_name'],
               'UHF42' : ['uhf42_geometry', 'uhf42_name']}

# Callback maken
def callback_kaart(attr, old, new):
    jaar = yearslider_kaart.value
    # Stippen kaart
    new_stippen_dataframe = df_geo_stippen[df_geo_stippen['year_sold'] == str(jaar)] 
    new_stippen_source = ColumnDataSource(new_stippen_dataframe)
    source_stippen.data = dict(new_stippen_source.data)
    
    # Polygon kaart
    title_polymap.text = 'Average sale price of house in New York City per ' + dropdown_polymap.value
    verdeling = districten[dropdown_polymap.value]
    new_geojson = gpd.GeoDataFrame(woningen_geo.loc[woningen_geo['sale_date_fixed'].dt.year == jaar, [*cols, verdeling[0]]]
                                                .rename(columns={verdeling[0]: 'geometry'})\
                                                .groupby([verdeling[1]]).agg({'sale_price':'mean', 'geometry': 'first'}).reset_index())
    geosource.geojson = new_geojson.to_json()
    p_polymap.tools[-1].tooltips = [('Place name', '@{{{}}}'.format(verdeling[1])),
                                   ('Average sale price', '@sale_price')]


dropdown_polymap.on_change('value', callback_kaart)

# Specify layout
p_polymap.add_layout(color_bar, 'below')

#%% 
# KAART MET STIPPEN BOKEH per jaar per klasse ------------------------------------------------------------------------------------------------------------
# Databewerkingen
# Maak geodataframe
df_geo_stippen = woningen[['latitude', 'longitude', 'gross_square_feet', 'building_class_at_time_of_num', 'year_sold']].copy()

# Locaties omzetten naar iets wat bokeh begrijpt
df_geo_stippen['geometry'] = df_geo_stippen.apply(lambda x: str((x.latitude, x.longitude)), axis = 1)
def merc(Coords):
    Coordinates = literal_eval(Coords)
    lat = Coordinates[0]
    lon = Coordinates[1]
    
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale
    return (x, y)
      
df_geo_stippen['coords_x'] = df_geo_stippen['geometry'].apply(lambda x: merc(x)[0])
df_geo_stippen['coords_y'] = df_geo_stippen['geometry'].apply(lambda x: merc(x)[1])

# Oppervlakte wordt de grootte van de stippies
# Inspecteren oppervlakte
fig, ax = plt.subplots(figsize=(4,8))
ax.boxplot(df_geo_stippen[df_geo_stippen['gross_square_feet']>0]['gross_square_feet']) 
plt.title("Boxplot surface area (square feet)") 
# Hoogste twee waarden weghalen
df_geo_stippen = df_geo_stippen[df_geo_stippen['gross_square_feet'] < df_geo_stippen['gross_square_feet'].nlargest(2).iloc[-1]]

fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = df_geo_stippen,
            x = "gross_square_feet",
            kde=True)
plt.title("Histogram surface area (square feet")
plt.xlabel("Surface area (square feet)") # Weirddddddd die piek aan het begin?

fig, ax = plt.subplots(figsize = (7,5))
sns.histplot(data = df_geo_stippen[df_geo_stippen['gross_square_feet']<30],
            x = "gross_square_feet")
plt.title("Histogram surface area where surface area < 30 square feet")
plt.xlabel("Surface area (square feet)") 
# Alles onder 30 verwijderen
df_geo_stippen = df_geo_stippen[df_geo_stippen['gross_square_feet']>30]

# Grenzen outliers
q1_opp = df_geo_stippen.gross_square_feet.quantile(q=0.25)
q3_opp = df_geo_stippen.gross_square_feet.quantile(q=0.75)
iqr_opp = q3_opp - q1_opp
upper_opp = q3_opp + 1.5*iqr_opp
lower_opp = q1_opp - 1.5*iqr_opp

fig, ax = plt.subplots(figsize=(4,8))
ax.boxplot(df_geo_stippen['gross_square_feet']) 
plt.title("Boxplot surface area (square feet)") 
# Alle outliers verwijderen
f_1 = df_geo_stippen['gross_square_feet'] > lower_opp
f_2 = df_geo_stippen['gross_square_feet'] < upper_opp
df_geo_stippen = df_geo_stippen[f_1 & f_2].copy()

#%%
# Kaart maken
# Maak nieuwe var voor grootte van de stippies
df_geo_stippen['circle_sizes'] = df_geo_stippen['gross_square_feet']/300
# Maak nieuwe var voor kleuren cd stippies
#df_geo_stippen['circle_colors'] = np.where(df_geo_stippen['building_class_at_time_of_num']== "A", "#2AC1C5", "#C52A5B")

# Color mapper/legenda
LABELS_kaart_stippen = df_geo_stippen.building_class_at_time_of_num.unique().tolist()
color_mapper_kaart_stippen = CategoricalColorMapper(factors = LABELS_kaart_stippen, palette=[palette1[2], palette1[6]])

# Kaart maken
tile_provider = get_provider(CARTODBPOSITRON)
p_stipmap = figure(x_range=(-8254500, -8216000), # Geeft de min en max van long aan
                         x_axis_type="mercator", 
                         y_axis_type="mercator",
                         title = "Map of sold houses per building class",
                         tools = "pan,wheel_zoom,box_zoom",)
# range bounds supplied in web mercator coordinates
p_stipmap.add_tile(tile_provider)
source_stippen = ColumnDataSource(df_geo_stippen[df_geo_stippen['year_sold'] == '2019']) #Eerste keer plotten is voor 2019
p_stipmap.circle(source = source_stippen,
                       x = "coords_x",
                       y = "coords_y",
                       size = "circle_sizes",
                       fill_alpha = 0.05,
                       color={'field':"building_class_at_time_of_num", 'transform': color_mapper_kaart_stippen},
                       legend_field='building_class_at_time_of_num'
                       )
#show(p_stipmap)

# Slider voor maand - Callback functie staat bij polykaart
yearslider_kaart = Slider(title='Year', value=2018,
                    start=2016, end=2019, step=1)
yearslider_kaart.on_change('value', callback_kaart)


kaart_stippen_col = column(yearslider_kaart, p_stipmap)



#%%
# LAYOUT SAMENVOEGEN EN APP RUNNEN
from bokeh.models.layouts import Panel, Tabs
from bokeh.models import Div

# Plot/widget namen:
# 1D:
# p_hist_prijs
# p_boxplot

# p_pie (hoeveelheid huizen per borough)
# checkbox_button_group          (samen in p_pie_col)

# p_pie_class (hoeveelheid huizen per klasse)
# checkbox_pie_class       (samen column_pie_class)

# ----------

# 2D:
# p_line_prijs (prijs per klasse/borough)
# dropdown_line_prijs       (samen in p_line_prijs_col)

# p_line_prijs_bor (prijs per borough pm)
# date_range_slider       (samen in p_line_prijs_bor_col)

# p_scatter_prijs_sqf
# checkboxes (boroughs)
# monthslider (samen in p_scatter_prijs_sqf_row)

# ----------

# Regressie:
# p_scatter_regr

# p_hist_res
# p_scatter_res_fitted
# dropdown_regressie

# -----------

# Kaart:
# p_polymap
# dropdown_polymap

# p_stipmap
# yearslider_kaart

# Teksten
lay1_txt0 = Div(css_classes=["eigentekst", "intro"], margin=[80, 0, 50, 0], text=""" 
<h2>One dimensional visualizations</h2>
<p>Welcome to the New York City Property Sales dashboard. This dashboard contains information on sold buildings in New York 
from 2016 through 2019. Before visualization, the following steps have been taken to prepare and clean the data: </p>
<ul>
    <li>Houses with a price below $10,000 dolars have been removed.</li>
    <li>Duplicates have been removed.</li>
    <li>Houses without a latitude/longitude have been removed.</li> 
</ul>
<p>To see our extensive data preparation and cleaning process, please refer to the source code. <br />
<i>Data source: <a href="https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r">NYC Open Data</a>.</i></p>

<h3>Boxplots and histograms</h3>
<p>Unrealistic values (sale price of $0) have been removed. Since the distribution of the price is obviously skewed, we've
chosen to apply a log transformation. The result is plotted in the one dimensional visualizations below.
</p>
""")

lay1_txt1 = Div(css_classes=["eigentekst"], margin=[0, 0, 50, 0], text="""
<h2>Two dimensional visualizations</h2>
<h3>Average price vs. surface area</h3>
<p>In the following scatterplot we see the relationship between the sale price and surface area of buildings. 
In all boroughs, we see that a bigger surface area means a higher price. In Manhattan the price climbs the fastest, 
in Staten Island the least. Choose a borough with the dropdown menu, and a month with the slider.</p>
""")

lay1_txt2 = Div(css_classes=["eigentekst"], margin=[0, 0, 50, 0], text="""
<h3>Per borough: Average prices and amount of houses sold</h3>
<p>In the following lineplot, we see how the average price rises and falls through time, per borough. You can choose a time 
interval with the slider. Amongst other things, we see that the average price in Bronx, Staten Island and Queens 
has increased the most from 2016 through 2019.</p>
""")

lay1_txt3 = Div(css_classes=["eigentekst"], margin=[0, 0, 50, 0], text="""
<h3>Per building class: Average prices and amount of houses sold</h3>
<p>In the following lineplot we see the average price per building class over time. Choose a borough with the 
dropdown menu. The plot shows that for both classes, the average sale price is lowest in the Bronx. The pie chart 
shows a distribution of the amount of houses sold.</p>
""")

lay2_txt0 = Div(css_classes=["eigentekst"], margin=[80, 0, 50, 0], text="""
<h2>Regression model</h2>
<h3>Correlation between council district and sale price</h3>
<p>In this scatterplot, we see how the sale price of houses develops over time for the Bronx borough, divided into
its council districts. There is definitely a difference to be seen between the sale prices of the different council districts. 
The datapoints of district 18 are significantly lower in the plot. We'll analyze this with a regression model, and show you 
the results down below.</p>
""")

lay2_txt1 = Div(css_classes=["eigentekst"], margin=[0, 0, 50, 0], text="""
<h3>Regression model results</h3>
<p>In order to build a model to predict a house's sale price, we've removed outliers regarding the log-transformed sale 
price. The independent variables for our model are time, council district and building class. <br />

In the histogram, we see how the residuals of the regression model are distributed. This should look like a normal distribution 
or bell curve. In the fitted-vs-residual plot, we see the fitted values vs. the residuals. In a perfect scenario, there 
would be no patterns visible in this plot and points would be distributed randomly.</p>
""")

lay3_txt0 = Div(css_classes=["eigentekst"], margin=[80, 0, 50, 0], text="""
<h2>Geospatial visualisations</h2>
<h3>Sale price in different types of districts</h3>
<p>The following map shows us the average sale price of houses in different types of districts in New York City. Besides its 
well known boroughs, New York can also be classified several types of districts. As we saw in the regression model results,
the district in which a house lies strongly affects its sale price. Use the dropdown menu to select the type of districts
you want to see. Use the slider above to select the year.</p> <br />
<i>Data source geojson: <a href="https://github.com/nycehs/NYC_geography">NYC Environmental Health Services Github</a>.</i></p>
""")

lay3_txt1 = Div(css_classes=["eigentekst"], margin=[0, 0, 50, 0], text="""
<h3>Houses and their surface areas, per building class</h3>
<p>The following map shows all the houses sold in the categories A and R, the sizes of the circles indicating their 
surface areas. This map indicates that in 2016 and 2017, not many houses in the R-class were sold. Class A stands for 
detachted houses, while R means certain types of appartments. The map clearly shows that Manhattan doesn't have many 
detached houses.</p>
""")

# Rijen per layout
lay1_row0 = row(p_boxplot, p_hist_prijs)   # boxplot en histogram
lay1_row1 = row(p_scatter_prijs_sqf, column(monthslider, checkboxes))      # links: scatter | rechts: tekst (2d viz?), monthslider, checkboxes
lay1_row2 = row(column(date_range_slider, p_line_prijs_bor), column(checkbox_button_group, p_pie))
lay1_row3 = row(column(dropdown_line_prijs, p_line_prijs), column(checkbox_pie_class, p_pie_class))


lay2_row0 = row(p_scatter_regr) # + intro tekst regressie
lay2_row1 = row(column(dropdown_regressie, p_hist_res), p_scatter_res_fitted)


lay3_row0 = row(column(yearslider_kaart))   # + intro tekst kaarten?
lay3_row1 = row(dropdown_polymap, p_polymap)
lay3_row2 = row(p_stipmap)

# lay4_row0 = ()   # bronvermelding

# Classes eventueel extra aan rows toevoegen
# lay1_row0.css_classes = ["customrow", "intro"]

# Alle rows in layouts
layout1 = column(lay1_txt0,
                 lay1_row0,
                 lay1_txt1,
                 lay1_row1,
                 lay1_txt2,
                 lay1_row2,
                 lay1_txt3,
                 lay1_row3)

layout2 = column(lay2_txt0,
                 lay2_row0,
                 lay2_txt1,
                 lay2_row1)

layout3 = column(lay3_txt0,
                 lay3_row0,
                 lay3_row1,
                 lay3_txt1,
                 lay3_row2)

#layout4 = column(lay4_row0)

# Layouts in tabbladen
tab1 = Panel(child=layout1, title='1D and 2D visualizations')
tab2 = Panel(child=layout2, title='Regression model')
tab3 = Panel(child=layout3, title='Maps')
tabs = Tabs(tabs=[tab1, tab2, tab3], css_classes=['hoi'])


 #%%
# Theme
from bokeh.themes import Theme
from os.path import dirname, join

curdoc().theme = Theme(filename=join(dirname(__file__), "theme.yml"))

#%%
curdoc().add_root(tabs)

