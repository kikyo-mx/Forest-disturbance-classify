from multiprocessing import Pool
from scipy import stats
import pandas as pd
import numpy as np
import joblib
import os

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from osgeo import ogr

feature_list = ['PT_NDMI_0','PT_NDMI_1','PT_NDMI_2','PT_NDMI_3','PT_NDMI_4','PT_NDMI_5','PT_NBR_0',                
           'PT_NBR_1','PT_NBR_2','PT_NBR_3','PT_NBR_4','PT_NBR_5','PT_NDVI_0','PT_NDVI_1',                
           'PT_NDVI_2','PT_NDVI_3','PT_NDVI_4','PT_NDVI_5', 'D/A', 'Shape Index', 'Perimeter'
            ,'area','LT_mag', 'NDVI_rvalue','NDMI_rvalue','NBR_rvalue','NDVI_slope','NDMI_slope','NBR_slope']
list_first = ['PT_NDMI_0','PT_NDMI_1','PT_NDMI_2','PT_NDMI_3','PT_NDMI_4','PT_NDMI_5','PT_NBR_0',                
           'PT_NBR_1','PT_NBR_2','PT_NBR_3','PT_NBR_4','PT_NBR_5','PT_NDVI_0','PT_NDVI_1',                
           'PT_NDVI_2','PT_NDVI_3','PT_NDVI_4','PT_NDVI_5', 'D/A', 'Shape Index', 'Perimeter'
            ,'area','LT_mag']
#               ,'NDVI_rvalue','NDMI_rvalue','NBR_rvalue','NDVI_slope','NDMI_slope','NBR_slope']

# canad_model = joblib.load("canada.model")
ne_model = joblib.load("ne_shpindex.model")
driver = ogr.GetDriverByName('ESRI Shapefile')
path = 'D:\\yyx\\GEE\\data\\shp_index\\'


def add_field_prediction(path):
    shp = driver.Open(path, 1)
    layer = shp.GetLayer()
#     field_canda = ogr.FieldDefn('type_cd', ogr.OFTReal)
    field_ne = ogr.FieldDefn('type_ne_sh', ogr.OFTReal)
#     layer.CreateField(field_canda)
    layer.CreateField(field_ne)
    for i in range(len(layer)):
        feature = layer.GetFeature(i)
        input_data = pd.DataFrame(columns=list_first)
        input_data = input_data.append(feature.items(), ignore_index=True)
        ts = pd.Series(input_data[list_first].values[0])
        # print(ts)
        if ts.isnull().any():
            feature.SetField("type_ne_sh", 4)
            continue
        ndvi = get_slope('NDVI', input_data)
        ndmi = get_slope('NDMI', input_data)
        nbr = get_slope('NBR', input_data)
        input_data = input_data.join(ndvi)
        input_data = input_data.join(ndmi)
        input_data = input_data.join(nbr)
        input_data = input_data[feature_list].values
#         prediction_canada = canad_model.predict(input_data)
        prediction_ne = ne_model.predict(input_data)
        feature.SetField("type_ne_sh", prediction_ne[0])
#         feature.SetField("type_cd", prediction_canada[0])
        layer.SetFeature(feature)
    shp.Destroy()
    print(path)


def add_index(name):
    index_list = []
    for i in range(1, 6):
        index_list.append('PT_' + name + '_' + str(i))
    return index_list


def get_slope(index, data):
    feature_list = [index + '_slope', index + '_intercept', index + '_rvalue', index + '_pvalue']
    dataframe = pd.DataFrame(columns=feature_list)
    index_list = add_index(index)
    x = [1, 2, 3, 4, 5]
    y_index = data[index_list].values
    for row in y_index:
        regress = stats.linregress(x, row)
        regress = [regress.slope, regress.intercept, regress.rvalue, regress.pvalue]
        data_regress = pd.Series(dict(zip(feature_list, regress)))
        dataframe = dataframe.append(data_regress, ignore_index=True)
    return dataframe


if __name__ == '__main__':
    file_list = []
    
    for year in range(1990, 2015):
        for i in range(115, 137):
            for j in range(54, 37, -1):
                input_path = path + str(year) + '//' + str(year) + '_' + str(i) + '_' + str(j) + '.shp'
                if os.path.exists(input_path) is True:
                    file_list.append(input_path)
                for k in range(0, 16):
                    input_path = path + str(year) + '//' + str(year) + '_' + str(i) + '_' + str(j) + '_' + str(k) + '.shp'
                    if os.path.exists(input_path) is True:
                        file_list.append(input_path)

#     file_list = r'D:\yyx\GEE\data\shp_index\1991//1991_117_44.shp'
    pool = Pool(processes=6)

    print('start')
#     add_field_prediction(file_list)
    pool.map(add_field_prediction, file_list)
    print('finish')
