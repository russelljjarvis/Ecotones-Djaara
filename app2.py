#import rasterio
#from matplotlib import pyplot
import dbfread
from dbfread import DBF
import pickle
data = [ record for record in DBF('BUSHBANK_DATABASE/BUSHBANK_DATABASE.tif.vat.dbf')]
#unique_EVCs
#data = [ record[""] for record in DBF('BUSHBANK_DATABASE/BUSHBANK_DATABASE.tif.vat.dbf')]
gold_fields = [d for d in data if d["BIOREGION"]=="Goldfields"]
EVCs = [ gold_fields[x]["EVCNAME"] for (x,i) in enumerate(gold_fields) ]
unique_EVCs = set(EVCs)

with open("EVCS_Goldfields","wb") as f:
    pickle.dump(unique_EVCs,f)

with open("EVCS_Goldfields","rb") as f:
    unique_EVCs = pickle.load(f)
print(unique_EVCs)




vic_geo = gpd.read_file("suburb-10-vic.geojson")
vic_inc_map = folium.Map([-36.7569, 144.2786], zoom_start=11,tiles=None)

    #print(record)
#import pdb
#pdb.set_trace()

#src = rasterio.open('BUSHBANK_DATABASE/BUSHBANK_DATABASE.tif')

#import rasterio


#src = rasterio.open("tests/data/RGB.byte.tif")

#pyplot.imshow(src.read(1), cmap='pink')
#<matplotlib.image.AxesImage object at 0x...>

#pyplot.show()




#print(dir(dataset))
#dataset.plot.show(dataset)#, with_bounds=True, contour=False, contour_label_kws=None, ax=None, title=None, transform=None, adjust=False, **kwargs)
