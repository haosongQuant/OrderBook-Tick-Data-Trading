#coding=utf-8
__author__ = '郝松'

import numpy as np
from time_sec_def import *
from order_book import order_book
from time_transform import time_transform

def extract_by_sec_basic(sec_itvl_list, time_second_basic, arrayList):

    resultList = []
    for dataArray in arrayList:
        resultArray = np.array([])
        for sec_itvl in sec_itvl_list:
            time_strt = sec_itvl[0]
            time_end = sec_itvl[1]
            itvl_array = dataArray[(np.where((time_strt <= time_second_basic) & \
                                   (time_second_basic < time_end))[0])] #09:00 之后的数据
            resultArray = np.concatenate((resultArray, itvl_array), axis = 0)
        resultList.append(resultArray)

    time_trim = 0
    pre_time_end = 0
    time_second_array = np.array([])
    for sec_itvl in sec_itvl_list:
        time_strt = sec_itvl[0]
        time_end = sec_itvl[1]
        time_trim += time_strt - pre_time_end
        itvl_array = time_second_basic[np.where((time_strt <= time_second_basic) & \
                                                (time_second_basic < time_end))[0]] #09:00 之后的数据
        itvl_array -= time_trim
        pre_time_end = time_end
        time_second_array = np.concatenate((time_second_array, itvl_array), axis = 0)

    return time_second_array, resultList

if __name__ == '__main__':

    # sec_itvl_list = [[0.0, 3.0], [4.5, 5.5], [8, 10]]
    # time_second_basic = np.array([0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])
    # seriesList = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    #               np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'])]
    #
    # time_second_array, resultList = extract_by_sec_basic(sec_itvl_list, time_second_basic, seriesList)
    # print(time_second_array)
    # print(resultList)

    time_sec, time_millis, bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,\
            bid_quantity_2,bid_quantity_3,ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,\
            ask_quantity_2,ask_quantity_3 = order_book('E:\\高频五档行情\\dce\\20170103\\m1705_20170103.csv')

    time_second,time_second_basic = time_transform(time_sec, time_millis)
    print(time_second_basic)
    print(time_second)
    print(bid_price_1)
    print(bid_quantity_1)
    print(ask_price_1)
    print(ask_quantity_1)
    time_second_basic, [time_second, \
    bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,bid_quantity_2,bid_quantity_3, \
    ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,ask_quantity_2,ask_quantity_3] \
        = extract_by_sec_basic([[0, TIME_INTV_1_15], [TIME_INTV_1_30, TIME_INTV_2_30], [TIME_INTV_4_30, TIME_INTV_6_00]],
        time_second_basic,[time_second, \
         bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,bid_quantity_2,bid_quantity_3, \
         ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,ask_quantity_2,ask_quantity_3])
    print(time_second_basic)
    print(time_second)
    print(bid_price_1)
    print(bid_quantity_1)
    print(ask_price_1)
    print(ask_quantity_1)