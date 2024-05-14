# https://jingwen-z.github.io/how-to-draw-a-variety-of-maps-with-folium-in-python/
import folium
from matplotlib import pyplot as plt

# importing all necessary libraries
import folium
import geopandas as gpd
import pandas as pd	 
import pdb
import folium
import streamlit as st

from streamlit_folium import st_folium
import copy


vic_inc_map = folium.Map([-36.7569, 144.2786], zoom_start=12,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(vic_inc_map)



#Loading data dependencies using geopandas
vic_geo = gpd.read_file("suburb-10-vic.geojson")
vic_geo = vic_geo[['vic_loca_2','geometry']]
Bendigo = vic_geo[vic_geo["vic_loca_2"] == "BENDIGO"]
Strath = vic_geo[vic_geo["vic_loca_2"] == "STRATHFIELDSAYE"]
Axe_C = vic_geo[vic_geo["vic_loca_2"] == "AXE CREEK"]
Axe_D = vic_geo[vic_geo["vic_loca_2"] == "AXE DALE"]
Mandu = vic_geo[vic_geo["vic_loca_2"] == "MANDURANG"]
Sedgwick = vic_geo[vic_geo["vic_loca_2"] == "SEDGWICK"]

labels = ["BENDIGO","STRATHFIELDSAYE","AXE CREEK","AXE DALE","MANDURANG","SEDGWICK","MANDURANG SOUTH","EMU CREEK","EPPALOCK","SPRING GULLY"]
labels_enum = [i for (i,J) in enumerate(labels)]
#dfBendigo = copy.copy(Bendigo)
gdfBendigo = vic_geo[vic_geo["vic_loca_2"].isin(labels)]
#for i in vic_geo.iterrows(): 
#    print(i)
#pdb.set_trace()
#dfBendigo = vic_geo[i==labels.any() for i in vic_geo["vic_loca_2"]]
print(gdfBendigo)

def dontdo():
    suburbs = set(vic_geo["vic_loca_2"])
    sorted_suburbs = sorted(suburbs, reverse=False)
    print(sorted_suburbs)

    #dfBendigo.append(Strath)
    #dfBendigo.append(Axe_C)
    #dfBendigo.append(Axe_D)
    #dfBendigo.append(Mandu)
    #dfBendigo.append(Sedgwick)


#pdb.set_trace()
# Creating a map object
#melb_inc_map = folium.Map(location=[y_map, x_map], zoom_start=6,tiles=None)
gdfBendigo['new_col'] = range(1, len(gdfBendigo) + 1)
gdfBendigo
# Creating choropleth 
folium.Choropleth(geo_data=gdfBendigo,
             name='Choropleth',         
             data=gdfBendigo,
             columns=['vic_loca_2','geometry'], 
             fill_color='YlOrRd',
             fill_opacity=0.6, 
             line_opacity=0.8,
             smooth_factor=0,     
             highlight=True,
                 ).add_to(vic_inc_map) 


# Adding labels to map
style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
SuburbName = folium.features.GeoJson(
    gdfBendigo,
    style_function=style_function, 
    control=False,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['vic_loca_2'],
        aliases=['SuburbName'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
vic_inc_map.add_child(SuburbName)
vic_inc_map.keep_in_front(SuburbName)
folium.LayerControl().add_to(vic_inc_map)

folium.LayerControl().add_to(vic_inc_map)

#vic_inc_map.show_in_browser()
st.set_page_config(layout="wide")
st_data = st_folium(vic_inc_map, width=725)
"""
nbh_count_df = listing_df.groupby('neighbourhood')['id'].nunique().reset_index()
nbh_count_df.rename(columns={'id':'nb'}, inplace=True)
nbh_geo_count_df = pd.merge(nbh_geo_df, nbh_count_df, on='neighbourhood')
nbh_geo_count_df['QP'] = nbh_geo_count_df['nb'] / nbh_geo_count_df['nb'].sum()
nbh_geo_count_df['QP_str'] = nbh_geo_count_df['QP'].apply(lambda x : str(round(x*100, 1)) + '%')

from branca.colormap import linear
nbh_count_colormap = linear.YlGnBu_09.scale(min(nbh_count_df['nb']),
                                            max(nbh_count_df['nb']))

nbh_locs_map = folium.Map(location=[48.856614, 2.3522219],
                          zoom_start = 12, tiles='cartodbpositron')

style_function = lambda x: {
    'fillColor': nbh_count_colormap(x['properties']['nb']),
    'color': 'black',
    'weight': 1.5,
    'fillOpacity': 0.7
}

folium.GeoJson(
    nbh_geo_count_df,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['neighbourhood', 'nb', 'QP_str'],
        aliases=['Neighbourhood', 'Location amount', 'Quote-part'],
        localize=True
    )
).add_to(nbh_locs_map)

nbh_count_colormap.add_to(nbh_locs_map)
nbh_count_colormap.caption = 'Airbnb location amount'
nbh_count_colormap.add_to(nbh_locs_map)

"""
