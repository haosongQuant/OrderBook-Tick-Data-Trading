#coding=utf-8
__author__ = 'haosong'
import numpy as np

def rise_ask(Ask1,timestamp_time_second,before_time):
    Ask1[Ask1 == 0] = np.mean(Ask1)
    rise_ratio = []
    index = np.where(timestamp_time_second >= before_time)[0][0]
    #open first before_time mins
    for i in np.arange(0,index):
        rise_ratio_ = round((Ask1[i] - Ask1[0])*(1.0)/Ask1[0]*100,5)
        rise_ratio.append(rise_ratio_)
    for i in np.arange(index,len(Ask1)):
        index_start_Array = np.where(timestamp_time_second[:i] >= timestamp_time_second[i] - before_time)
        if len(index_start_Array[0]) == 0:
            rise_ratio.append(0)
        else:
            index_start = index_start_Array[0][0]
            rise_ratio_ = round((Ask1[i] - Ask1[index_start])*(1.0)/Ask1[index_start]*100,5)
            rise_ratio.append(rise_ratio_)
    return np.array(rise_ratio)