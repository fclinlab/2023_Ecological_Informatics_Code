from osgeo import gdal
import numpy as np

#import cv2 as cv

def get_dataset_band(bandfile):
    input_dataset = gdal.Open(bandfile)
    input_band = input_dataset.GetRasterBand(1) #取得波段資訊, 並非取內容資料
    
    return [input_dataset, input_band]

def multi_band(): #待修正
    dataset = gdal.Open('./test_graph/test1.tif', gdal.GA_ReadOnly)

    tmp_img = dataset.ReadAsArray() #獲取資訊

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    outbandsize = dataset.RasterCount #波段數
    #band = dataset.GetRasterBand(1)
    inputdataset_1, inputband_1 = get_dataset_band('./test_graph/test1.tif')
    print(type(inputdataset_1))
    print(type(inputband_1))
    count = inputdataset_1.RasterCount
    print(count)
    file_driver = gdal.GetDriverByName('Gtiff')
    output_dataset = file_driver.Create(
        'natural_color.tif', inputband_1.XSize, inputband_1.YSize, 9, inputband_1.DataType
    )
    output_dataset.SetProjection(inputdataset_1.GetProjection())
    output_dataset.SetGeoTransform(inputdataset_1.GetGeoTransform())
    output_dataset.BuildOverviews('average', [2, 4, 8, 16, 32])  # 建立影像金字塔
    #print("outbandsize is %d"%(outbandsize))
    #im_geotrans = dataset.GetGeoTransform() #仿射矩陣資訊
    #im_proj = dataset.GetProjection() #投影訊息
    #datasettype = dataset.GetRasterBand(1).datasetType


    #cv.imshow('img', img_rgb)
    #cv.waitKey()
    #cv.destroyAllWindows()
    #print(type(dataset))
    #in_band1 = dataset.GetRasterBand(1)
    #print(type(in_band1))
    #max_inband7 = in_band1.GetMaximum()
    #print(max_inband7)
    #in_band2 = dataset.GetRasterBand(2).ReadAsArray()
    #in_band3 = dataset.GetRasterBand(8).ReadAsArray()
    #print(np.max(in_band3))
    

def merge_test(path): #正常運作
    #print('in merge_test!!')
    #print(type(path))
    dataset = gdal.Open(path)
   
    #num=ds.RasterCount
    #im_width = ds.RasterXSize
    #im_height = ds.RasterYSize
    #im_geotrans = ds.GetGeoTransform()
    #im_proj = ds.GetProjection()
    num = dataset.RasterCount  
    i = 1
    band_list = []
    while(i <= num):
        band_list.append(dataset.GetRasterBand(i).ReadAsArray())
        i += 1

    #merge_arr = np.concatenate((band, band2, band3), axis=0) # 結合三個波段(error)
    merge_arr = np.stack(band_list, axis=2) # 結合三個波段 (大概, 待修正)
    #print(merge_arr.shape)
    return merge_arr
    ############################################################################################################ 
    ##以上為輸出numpy部分 以下為輸出tif檢驗部分
    ############################################################################################################
    '''
    #print(merge_arr.shape)
    tif_driver = gdal.GetDriverByName('GTiff')    
    out_ds = tif_driver.Create('natural_color4.tif', im_width,im_height, 3, band1.DataType)    
    out_ds.SetProjection(im_proj)    
    out_ds.SetGeoTransform(im_geotrans)
    
    out_band = out_ds.GetRasterBand(1)    
    out_band.WriteArray(band)
    
    out_band = out_ds.GetRasterBand(2)    
    out_band.WriteArray(band2)
    
    out_band = out_ds.GetRasterBand(3)    
    out_band.WriteArray(band3)
    
    out_ds.FlushCache()
    
    for i in range(1,4):
        out_ds.GetRasterBand(i).ComputeStatistics(False)  
    del out_ds
    '''
'''
def ran_merge_test():
    band = np.random.randint(low=0, high=256, size=(256, 256))
    band2 = np.random.randint(low=0, high=256, size=(256, 256))
    band3 = np.random.randint(low=0, high=256, size=(256, 256))
    #print(band3.ndim)
    #print(band3.shape)
    merge_arr = np.stack([band, band2, band3], axis=2)
    print(merge_arr)
    #print(merge_arr.ndim)
    #print(merge_arr.shape)
    return merge_arr
'''    
    

def main():
    merge_test(r'../../test_images/000000098.tif')
    #multi_band()
    #a = ran_merge_test()
    #print(a)

if __name__ == "__main__":
    main()
    



#for i in in_band1:
    #print(i)
#print(in_band1)
#print(in_band2)
#print(in_band3)


'''
in_band1 = dataset.GetRasterBand(1)
in_band2 = dataset.GetRasterBand(2)
in_band3 = dataset.GetRasterBand(3)

#定義切圖初始點座標
offset_x = 0
offset_y = 0

block_xsize = 100
block_ysize = 100

k=0
for i in range(width//block_xsize):
    for j in range(height//block_ysize):
        out_band1 = in_band1.ReadAsArray(i*block_xsize, j*block_xsize, block_xsize, block_ysize)
        out_band2 = in_band2.ReadAsArray(i*block_xsize, j*block_xsize, block_xsize, block_ysize)
        out_band3 = in_band3.ReadAsArray(i*block_xsize, j*block_xsize, block_xsize, block_ysize)
        print(out_band3)
        k+=1
        
        gtif_driver = gdal.GetDriverByName("GTiff")
        
        out_ds = gtif_driver.Create('./test_graph/test1/' + str(k) + 'clip.tif', block_xsize, block_ysize, outbandsize, datasettype)
        
        ori_transform = dataset.GetGeoTransform()
        if ori_transform:
            print(ori_transform)
            print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
            print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        
        top_left_x = ori_transform[0]
        w_e_pixel_resolution = ori_transform[1]
        top_left_y = ori_transform[3]
        n_s_pixel_resolution = ori_transform[5]
        
        top_left_x = top_left_x + i*block_xsize * w_e_pixel_resolution
        top_left_y = top_left_y + j*block_ysize * n_s_pixel_resolution
        
        dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        
        out_ds.SetGeoTransform(dst_transform)
        
        out_ds.SetProjection(dataset.GetProjection())
        
        out_ds.GetRasterBand(1).WriteArray(out_band1)
        out_ds.GetRasterBand(2).WriteArray(out_band2)
        out_ds.GetRasterBand(3).WriteArray(out_band3)
        
        out_ds.FlushCache()
        print("FlushCache succeed")
        
        out_ds = None
        del out_ds
        
        print("End!")


'''