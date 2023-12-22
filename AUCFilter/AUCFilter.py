import numpy as np
import os
from datetime import datetime
import simplejson as json
import matplotlib.pyplot as plt
# from typing import List, Dict
# from osgeo import gdal
from chensg import chen_sg_filter as cf
# import seaborn as sns
# from collections import defaultdict, Counter
from shapely import geometry
# from scipy.stats import norm
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
# from fastdtw import fastdtw
import geopandas as gpd
import pandas as pd
# import pandas as pd
# import pyproj.crs as crs
# import itertools
from scipy.stats import norm
# 设置字体
plt.rcParams['font.family'] = 'Microsoft YaHei'



#### 预定义函数

# %%

def si(f, g):
    """
    Compute the statistical index of coincidence (SIC) between two probability distributions.

    $$ SI(f, g)=1-\frac{\inf _{\substack{a[0,1]-\left[a, a^{\prime}\right] \\ \beta[0,1]-[b, b]}} \max _{t \in[0,1]}\|f(\alpha(t))-g(\beta(t))\|}{\sup _{\substack{\alpha[0,1]-\left[a, a^{\prime}\right] \\ \beta[0,1]-\left[b, b^{\prime}\right]}} \max _{t \in[0,1]}\|f(\alpha(t))-g(\beta(t))\|}
    $$

    Parameters
    ----------
    f : array-like
        The first probability distribution.
    g : array-like
        The second probability distribution.

    Returns
    -------
    float
        The SIC between f and g.
    """
    # Generate all possible alignments of f and g.
    a = 0
    a_prime = len(f)
    b = 0
    b_prime = len(g)
    alpha = np.arange(a, a_prime)
    beta = np.arange(b, b_prime)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta)
    alpha_beta = np.column_stack([alpha_grid.ravel(), beta_grid.ravel()])

    # Compute the distances between f and g for all possible alignments.
    distances = np.abs(f[alpha_beta[:, 0]] - g[alpha_beta[:, 1]])
    distances = distances.reshape(alpha_grid.shape)
    max_distances = np.max(distances, axis=1)
    min_distance = np.min(max_distances)

    # Compute the maximum distance for all possible alignments.
    alpha = np.arange(a, a_prime)
    beta = np.arange(b, b_prime)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta)
    alpha_beta = np.column_stack([alpha_grid.ravel(), beta_grid.ravel()])
    distances = np.abs(f[alpha_beta[:, 0]] - g[alpha_beta[:, 1]])
    distances = distances.reshape(alpha_grid.shape)
    max_distances = np.max(distances, axis=1)
    max_distance = np.max(max_distances)

    # Return the SIC.
    return 1 - min_distance / max_distance


# %%

def read_json(path, featureParam, DOYs, nodataValue=0, type=None):
    """
    Reads a JSON file from the given path, extracts the 'properties'
    field of each feature, truncates the keys to the first 8 characters
    and converts them to datetime objects, and returns a geopandas
    dataframe containing the updated properties.

    Args:
        path (str): The path to the JSON file.

    Returns:
        geopandas.GeoDataFrame: A geopandas dataframe containing the updated properties.
    """
    date_format = '%Y%m%d'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    df_data = {"id": [], "geometry": []}
    df_cropType = {"id": [], "geometry": [], "2017": [], "2018": [], "2019": []}
    for doy in DOYs:
    # for doy in range(120, 271, 10):
        df_data[str(doy)] = []

    for feature in data['features']:
        properties = feature['properties']
        # df_cropType['id'].append(feature["id"])
        # df_cropType['geometry'].append(geometry.Point(feature["geometry"]["coordinates"]))
        # df_cropType['2017'].append(properties["b1"])
        # df_cropType['2018'].append(properties["b1_1"])
        # df_cropType['2019'].append(properties["b1_2"])
        new_properties = {}
        if properties["year2019"] != type:
            continue
        for key, value in properties.items():
            if key.endswith(featureParam) and value != nodataValue and value < 1 and len(key) > 10:
                new_key = datetime.strptime(key[:8], date_format).timetuple().tm_yday
                if new_key in new_properties:
                    v1 = new_properties[new_key]
                    new_properties[new_key] = (value + v1) / 2
                else:
                    new_properties[new_key] = value

        doys, repi = list(new_properties.keys()), list(new_properties.values())

        if (len(doys) == 0):
            continue
        # delta_doy = [(doys[i+1] - doys[i]) for i in range(len(doys) - 1)]
        # if max(delta_doy) >= 30:
        #     continue

        reconstructData = {}
        for doy in DOYs:
        # for doy in range(120, 271, 10):
            _doy, idx = find_closest_value(doy, doys)  # _doy 表示观测序列中最接近目标doy的值，idx表示_doy在观测值序列中索引
            if abs(doy - _doy) > 5:
                reconstructData[doy] = np.interp(doy, doys, repi)
            else:
                if idx >= 2:
                    i = idx - 2
                    sub_ts = []
                    while i < idx + 3 and i < len(doys):
                        if abs(doy - doys[i]) < 5:
                            sub_ts.append(repi[i])
                        i += 1
                    if len(sub_ts) == 0:
                        reconstructData[doy] = np.interp(doy, doys, repi)
                    else:
                        reconstructData[doy] = np.median(np.array(sub_ts))
                else:
                    reconstructData[doy] = np.interp(doy, doys, repi)

        df_data['id'].append(feature["id"])
        df_data['geometry'].append(geometry.Point(feature["geometry"]["coordinates"]))

        ts = np.array(list(reconstructData.values()))
        ts_ft = cf(np.copy(ts), 40)
        # ts_ft = ts
        for i, doy in enumerate(DOYs):
            df_data[str(doy)].append(ts_ft[i])

    return gpd.GeoDataFrame(df_data, crs="EPSG:4326"), gpd.GeoDataFrame(df_cropType, crs="EPSG:4326")

def find_closest_value(target, values):
    # 找到最接近给定值的索引
    idx = np.abs(np.array(values) - target).argmin()
    # 返回最接近的值和索引
    return values[idx], idx

def guassParaComp(args, DOYs, VINum):
    """
    计算所有样本点REPI,RENDVI,LSWI,SWIR1时序曲线下面积的多元高斯分布

    参数：
    args：每个矩阵的每一行代表一个样本所对应的时间序列，每一列代表某一个时间下所有样本的值
    DOYs：时间序列的时间刻度

    返回：
    均值向量和协方差矩阵

    """
    # print("args'length:", len(args))

    EVI_background = 0
    area_array = np.array([])
    for i, data in enumerate(args):
        data = data.values[:, 2:]
        row_num = data.shape[0]
        print("曲线数量", row_num)
        for j in range(row_num):
            # 计算每一条曲线的曲线下面积
            row = data[j, :]
            row[row < EVI_background] = 0
            area = np.trapz(row, DOYs)
            area_array = np.append(area_array, area)
    area_array = area_array.reshape(VINum, -1)
    # area_array = area_array.reshape(3, -1)

    # 对数据进行转置
    transposed_data = np.transpose(area_array)

    # 计算均值向量
    mean_vector = np.mean(transposed_data, axis=0)

    # 计算协方差矩阵
    covariance_matrix = np.cov(transposed_data, rowvar=False)

    # 输出均值向量和协方差矩阵
    print("Mean Vector:")
    print(mean_vector)
    print("Covariance Matrix:")
    print(covariance_matrix)

    sum_array = np.array([])
    for i, data in enumerate(args):
        data['sum'] = data.iloc[:, 2:].apply(integrate_rendvi, axis=1)
        sum = data['sum'].values
        sum_array = np.append(sum_array, sum)
        geometryInfo = data.values[:, 0:2]

    sum_array = sum_array.reshape(VINum, -1)
    # sum_array = sum_array.reshape(3, -1)
    # 计算马氏距离
    mahalanobis_distance = [distance.mahalanobis(sum_array[:, i], mean_vector, np.linalg.inv(covariance_matrix)) for
                             i in range(sum_array.shape[1])]
    distance_array = np.array(mahalanobis_distance)
    distance_array = distance_array.reshape(-1, 1)
    percentile_soy = np.percentile(distance_array, 50)
    percentile_crop = np.percentile(distance_array, 95)
    print("percentile_crop:", percentile_soy)

    return mean_vector, covariance_matrix, percentile_soy, percentile_crop

def sampleFilter(args1, args2,mean_vector1, covariance_matrix1, mean_vector2, covariance_matrix2, percentile_soy1, percentile_soy2, percentile_crop1, percentile_crop2, VINum):
    """
    args:geodatafram，前两列分别为id和geometry，后面所有列为RENDVI值
    """
    sum_array1 = np.array([])
    for i, data in enumerate(args1):
        data['sum'] = data.iloc[:, 2:].apply(integrate_rendvi, axis=1)
        sum = data['sum'].values
        sum_array1 = np.append(sum_array1, sum)
        geometryInfo = data.values[:, 0:2]

    sum_array1 = sum_array1.reshape(VINum, -1)
    # sum_array = sum_array.reshape(3, -1)
    # 计算马氏距离
    mahalanobis_distance1 = [distance.mahalanobis(sum_array1[:, i], mean_vector1, np.linalg.inv(covariance_matrix1)) for i in range(sum_array1.shape[1])]
    distance_array1 = np.array(mahalanobis_distance1)
    distance_array1 = distance_array1.reshape(-1, 1)

    sum_array2 = np.array([])
    for i, data in enumerate(args2):
        data['sum'] = data.iloc[:, 2:].apply(integrate_rendvi, axis=1)
        sum = data['sum'].values
        sum_array2 = np.append(sum_array2, sum)
        geometryInfo = data.values[:, 0:2]

    sum_array2 = sum_array2.reshape(VINum, -1)
    # sum_array = sum_array.reshape(3, -1)
    # 计算马氏距离
    mahalanobis_distance2 = [distance.mahalanobis(sum_array2[:, i], mean_vector2, np.linalg.inv(covariance_matrix2)) for
                             i in range(sum_array2.shape[1])]
    distance_array2 = np.array(mahalanobis_distance2)
    distance_array2 = distance_array2.reshape(-1, 1)

    array = np.concatenate((geometryInfo, distance_array1, distance_array2), axis=1)
    df = gpd.GeoDataFrame(array)
    df.columns = ['id', 'geometry', 'distance1', 'distance2']

    # print("percentile_soy1_30:", np.percentile(distance_array1, 30))
    # print("percentile_soy1_30:", np.percentile(distance_array2, 30))
    # percentile_crop1 = np.percentile(distance_array1, 90)
    # percentile_crop2 = np.percentile(distance_array2, 90)

    print("percentile_crop:", np.percentile(distance_array1, 90))
    dfSoy = df[(df['distance1'] <= percentile_soy1)]
    dfSoy = dfSoy[(dfSoy['distance2'] <= percentile_soy2)]
    dfCrop = df[(df['distance1'] >= percentile_crop1)]
    dfCrop = dfCrop[(dfCrop['distance2'] >= percentile_crop2)]
    # dfCrop = df[(df['distance1'] >= percentile_crop1) & (df['distance2'] >= percentile_crop2)]
    # dfCrop = dfCrop[(dfCrop['distance2'] >= percentile_crop2)]

    # # 绘制直方图
    # n, bins, patches = plt.hist(mahalanobis_distance1, bins=50, alpha=0.5, density=True)
    # plt.show()
    # n, bins, patches = plt.hist(mahalanobis_distance2, bins=50, alpha=0.5, density=True)
    # plt.show()
    # dist_pd = pd.DataFrame(mahalanobis_distance)
    # dist_pd.to_csv(r'D:\科研\大豆产品\中部地区\河南\mahalanobisSample\2020\mahaDist.csv')
    print("大豆样点数为:", len(dfSoy))
    print("非大豆样点数为:", len(dfCrop))
    dfSoy.to_file(r'.\HLJSoy_IM19.geojson', driver='GeoJSON')
    dfCrop.to_file(r'.\HLJCrop_IM19.geojson', driver='GeoJSON')


    filteredCropList = []
    for i, data in enumerate(args1):
        data['distance1'] = distance_array1
        data['distance2'] = distance_array2
        data = data[(data['distance1'] <= percentile_soy1) & (data['distance2'] <= percentile_soy2)]
        filteredCropList.append(data)

    return filteredCropList

# 定义积分函数
def integrate_rendvi(rendvi_values):
    DOYs = np.arange(180, 270, 10)
    # 计算积分值
    integral = np.trapz(rendvi_values, DOYs)
    return integral

# 大豆VIGeojson文件路径
VIList1 = ["EVI", "RE2", "SWIR"]
VIList2 = ["LSWI", "RENDVI", "REPI"]
VINum = len(VIList1)
DOYs=np.arange(180, 270, 10)
soybean_VIGeojson_filFolder2 = r".\data\Inner Monglia\VIFolder2"
soybean_VIGeojson_fileName2 = os.listdir(soybean_VIGeojson_filFolder2)
soybean_VIGeojson_filPath2 = [os.path.join(soybean_VIGeojson_filFolder2, i) for i in soybean_VIGeojson_fileName2]

soybean_VIGeojson_filFolder1 = r".\data\Inner Monglia\VIFolder1"
soybean_VIGeojson_fileName1 = os.listdir(soybean_VIGeojson_filFolder1)
soybean_VIGeojson_filPath1 = [os.path.join(soybean_VIGeojson_filFolder1, i) for i in soybean_VIGeojson_fileName1]

# 读取VIGeojson，分作物种类
soybean_VIGeojson_list1 = []
for i in range(VINum):
    VIGeojson, _ = read_json(soybean_VIGeojson_filPath1[i], VIList1[i], DOYs=DOYs)
    # VIGeojson, _ = read_json(Anhui_soybean_VIGeojson_filPath1[i], VIList1[i], DOYs=DOYs, type=0)
    # VIGeojson, _ = read_json2(Anhui_soybean_VIGeojson_filPath1[i], VIList1[i], DOYs=DOYs, type=2)
    soybean_VIGeojson_list1.append(VIGeojson)

soybean_VIGeojson_list2 = []
for i in range(VINum):
    VIGeojson, _ = read_json(soybean_VIGeojson_filPath2[i], VIList2[i], DOYs=DOYs)
    # VIGeojson, _ = read_json(Anhui_soybean_VIGeojson_filPath2[i], VIList2[i], DOYs=DOYs, type=0)
    # VIGeojson, _ = read_json2(Anhui_soybean_VIGeojson_filPath2[i], VIList2[i], DOYs=DOYs, type=2)
    soybean_VIGeojson_list2.append(VIGeojson)


# 计算均值矩阵和方差
mu1, std1, percentile_soy1, percentile_crop1 = guassParaComp(soybean_VIGeojson_list1, DOYs=DOYs, VINum=VINum)
mu2, std2, percentile_soy2, percentile_crop2 = guassParaComp(soybean_VIGeojson_list2, DOYs=DOYs, VINum=VINum)
# filteredCropList = sampleFilter(Anhui_soybean_VIGeojson_list1, Anhui_soybean_VIGeojson_list2, mu1, std1, mu2, std2, VINum)

# 作物VIGeojson文件路径
crop_VIGeojson_filFolder1 = r".\data\Heilongjiang\VIFolder1"
crop_VIGeojson_fileName1 = os.listdir(crop_VIGeojson_filFolder1)
crop_VIGeojson_filPath1 = [os.path.join(crop_VIGeojson_filFolder1, i) for i in crop_VIGeojson_fileName1]
# 作物VIGeojson文件路径
crop_VIGeojson_filFolder2 = r".\data\Heilongjiang\VIFolder2"
crop_VIGeojson_fileName2 = os.listdir(crop_VIGeojson_filFolder2)
crop_VIGeojson_filPath2 = [os.path.join(crop_VIGeojson_filFolder2, i) for i in crop_VIGeojson_fileName2]
# 读取VIGeojson
crop_VIGeojson_list1 = []
for i in range(VINum):
    VIGeojson, _ = read_json(crop_VIGeojson_filPath1[i], VIList1[i], DOYs=DOYs)
    print("作物{}样点数为:".format(VIList1[i]), len(VIGeojson))
    crop_VIGeojson_list1.append(VIGeojson)

crop_VIGeojson_list2 = []
for i in range(VINum):
    VIGeojson, _ = read_json(crop_VIGeojson_filPath2[i], VIList2[i], DOYs=DOYs)
    print("作物{}样点数为:".format(VIList2[i]), len(VIGeojson))
    crop_VIGeojson_list2.append(VIGeojson)

# 筛选
filteredCropList = sampleFilter(crop_VIGeojson_list1, crop_VIGeojson_list2, mu1, std1, mu2, std2, percentile_soy1, percentile_soy2, percentile_crop1, percentile_crop2, VINum)
