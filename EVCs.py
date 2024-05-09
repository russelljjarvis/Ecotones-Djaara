import geopandas as gpd
import matplotlib.pyplot as plt
from geo_adjacency.adjacency import AdjacencyEngine
from mpl_toolkits.axes_grid1 import make_axes_locatable
evcs = gpd.read_file('dataframe.geojson', driver='GeoJSON')

vic_geo = gpd.read_file("suburb-10-vic.geojson")
source_geoms = gpd.read_file('gda2020_vicgrid/filegdb/whole_of_dataset/victoria/FLORAFAUNA1.gdb', layer=0)
engine = AdjacencyEngine(source_geoms.geometry)
adjacency_dict = engine.get_adjacency_dict()
import pdb
pdb.set_trace()
# defaultdict(<class 'list'>, {0: [7, 90, 98, 101], 1: [54, 59, 136, 221], 2: [10, 137, ... ]})
engine.plot_adjacency_dict()

#clip points using small polys
#points_subset = gpd.clip(points,vic_geo)

#filter polys based on remaining points


labels = ["BENDIGO","STRATHFIELDSAYE","AXE CREEK","AXE DALE","MANDURANG","SEDGWICK","MANDURANG SOUTH","EMU CREEK","EPPALOCK","SPRING GULLY"]
gdfBendigo = vic_geo[vic_geo["vic_loca_2"].isin(labels)]
polys_subset_evc_bendigo = evcs[evcs.geometry.apply(lambda x: gdfBendigo.geometry.within(x).any())]
#-36.7569, 144.2786
def plot_EVC(evcs):
    
    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    evcs.plot(ax=ax,legend=True,cax=cax)
    
    plt.show()
    #plt.ion()

plot_EVC(polys_subset_evc_bendigo)
#mallard_0.to_file('dataframe.geojson', driver='GeoJSON')
