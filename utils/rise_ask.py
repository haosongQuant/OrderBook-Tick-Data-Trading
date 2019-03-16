#coding=utf-8
__author__ = 'haosong'
import numpy as np

def rise_ask(ask_price, timestamp_time_second, before_time):
    ask_price[ask_price == 0] = np.mean(ask_price)
    rise_ratio = []
    index = np.where(timestamp_time_second >= before_time)[0][0]
    #open first before_time mins
    for i in np.arange(0,index):
        rise_ratio_ = round((ask_price[i] - ask_price[0])*(1.0)/ask_price[0]*100,5)
        rise_ratio.append(rise_ratio_)
    for i in np.arange(index,len(ask_price)):
        index_start_Array = np.where(timestamp_time_second[:i] >= timestamp_time_second[i] - before_time)
        if len(index_start_Array[0]) == 0:
            rise_ratio.append(0)
        else:
            index_start = index_start_Array[0][0]
            rise_ratio_ = round((ask_price[i] - ask_price[index_start])*(1.0)/ask_price[index_start]*100,5)
            rise_ratio.append(rise_ratio_)
    return np.array(rise_ratio)