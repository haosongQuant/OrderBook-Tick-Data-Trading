#coding=utf-8
__author__ = 'haosong'

import os, sys
import pandas as pd
import numpy as np
sys.path.append('..//utils')
from time_sec_def import *
from order_book import order_book
from time_transform import time_transform
from rise_ask import rise_ask
from weight_pecentage import weight_pecentage
from traded_label import traded_label_one_second
from extract_by_sec_basic import extract_by_sec_basic
from parsePara import parsePara

def feature_DataFrame(traded_time,time_second_basic,bid_price_1,ask_price_1,rise_ratio_ask_1,\
                         rise_ratio_ask_2,rise_ratio_ask_3,rise_ratio_ask_4,rise_ratio_ask_5,\
                         rise_ratio_ask_6,rise_ratio_ask_7,rise_ratio_ask_8,rise_ratio_ask_9,\
                         rise_ratio_ask_10,rise_ratio_ask_11,rise_ratio_ask_12,rise_ratio_ask_13,\
                         rise_ratio_ask_14,rise_ratio_ask_15,rise_ratio_ask_16,rise_ratio_ask_17,\
                         rise_ratio_ask_18,rise_ratio_ask_19,rise_ratio_ask_20,rise_ratio_ask_21,\
                         rise_ratio_ask_22,rise_ratio_ask_23,rise_ratio_ask_24,rise_ratio_ask_25,\
                         rise_ratio_ask_26,rise_ratio_ask_27,rise_ratio_ask_28,rise_ratio_ask_29,\
                         rise_ratio_ask_30,W_AB_100, W_A_B_100, W_AB_010, W_A_B_010, W_AB_001,\
                         W_A_B_001, W_AB_910, W_A_B_910, W_AB_820, W_A_B_820, W_AB_730 , W_A_B_730,\
                         W_AB_640, W_A_B_640, W_AB_550, W_A_B_550,W_AB_721, W_A_B_721, W_AB_532,\
                         W_A_B_532, W_AB_111, W_A_B_111, W_AB_190, W_A_B_190, W_AB_280 , W_A_B_280,\
                         W_AB_370, W_A_B_370, W_AB_460, W_A_B_460, W_AB_127, W_A_B_127, W_AB_235, W_A_B_235):
    # 09:00 ~ 15:00
    time1 = 0
    time2 = TIME_INTV_3_45
    traded,rise_ratio_second_1,rise_ratio_second_2,rise_ratio_second_3,\
    rise_ratio_second_4,rise_ratio_second_5,rise_ratio_second_6,rise_ratio_second_7,\
    rise_ratio_second_8,rise_ratio_second_9,rise_ratio_second_10,rise_ratio_second_11,\
    rise_ratio_second_12,rise_ratio_second_13,rise_ratio_second_14,rise_ratio_second_15,\
    rise_ratio_second_16,rise_ratio_second_17,rise_ratio_second_18,rise_ratio_second_19,\
    rise_ratio_second_20,rise_ratio_second_21,rise_ratio_second_22,rise_ratio_second_23,\
    rise_ratio_second_24,rise_ratio_second_25,rise_ratio_second_26,rise_ratio_second_27,\
    rise_ratio_second_28,rise_ratio_second_29,rise_ratio_second_30,w_divid_100,w_diff_100,\
    w_divid_010,w_diff_010,w_divid_001,w_diff_001,w_divid_910,w_diff_910,w_divid_820,w_diff_820,\
    w_divid_730,w_diff_730,w_divid_640,w_diff_640,w_divid_550,w_diff_550,w_divid_721,w_diff_721,\
    w_divid_532,w_diff_532,w_divid_111,w_diff_111,w_divid_190,w_diff_190,w_divid_280,w_diff_280,\
    w_divid_370,w_diff_370,w_divid_460,w_diff_460,w_divid_127,w_diff_127,w_divid_235,w_diff_235,\
    spread, Best_Ask, Best_Bid = \
        traded_label_one_second(time1,time2,time_second_basic,bid_price_1,ask_price_1,traded_time,\
                                rise_ratio_ask_1,rise_ratio_ask_2,rise_ratio_ask_3,rise_ratio_ask_4,\
                                rise_ratio_ask_5,rise_ratio_ask_6,rise_ratio_ask_7,rise_ratio_ask_8,\
                                rise_ratio_ask_9,rise_ratio_ask_10,rise_ratio_ask_11,rise_ratio_ask_12,\
                                rise_ratio_ask_13,rise_ratio_ask_14,rise_ratio_ask_15,rise_ratio_ask_16,\
                                rise_ratio_ask_17,rise_ratio_ask_18,rise_ratio_ask_19,rise_ratio_ask_20,\
                                rise_ratio_ask_21,rise_ratio_ask_22,rise_ratio_ask_23,rise_ratio_ask_24,\
                                rise_ratio_ask_25,rise_ratio_ask_26,rise_ratio_ask_27,rise_ratio_ask_28,\
                                rise_ratio_ask_29,rise_ratio_ask_30,W_AB_100, W_A_B_100, W_AB_010, W_A_B_010,\
                                W_AB_001,W_A_B_001, W_AB_910, W_A_B_910, W_AB_820, W_A_B_820, W_AB_730,\
                                W_A_B_730,W_AB_640, W_A_B_640, W_AB_550, W_A_B_550,W_AB_721, W_A_B_721,\
                                W_AB_532,W_A_B_532, W_AB_111, W_A_B_111, W_AB_190, W_A_B_190, W_AB_280,\
                                W_A_B_280,W_AB_370, W_A_B_370, W_AB_460, W_A_B_460, W_AB_127, W_A_B_127,\
                                W_AB_235, W_A_B_235)

    data = np.array([traded,rise_ratio_second_1,rise_ratio_second_2,rise_ratio_second_3,\
                    rise_ratio_second_4,rise_ratio_second_5,rise_ratio_second_6,rise_ratio_second_7,\
                    rise_ratio_second_8,rise_ratio_second_9,rise_ratio_second_10,rise_ratio_second_11,\
                    rise_ratio_second_12,rise_ratio_second_13,rise_ratio_second_14,rise_ratio_second_15,\
                    rise_ratio_second_16,rise_ratio_second_17,rise_ratio_second_18,rise_ratio_second_19,\
                    rise_ratio_second_20,rise_ratio_second_21,rise_ratio_second_22,rise_ratio_second_23,\
                    rise_ratio_second_24,rise_ratio_second_25,rise_ratio_second_26,rise_ratio_second_27,\
                    rise_ratio_second_28,rise_ratio_second_29,rise_ratio_second_30,w_divid_100,w_diff_100,\
                    w_divid_010,w_diff_010,w_divid_001,w_diff_001,w_divid_910,w_diff_910,w_divid_820,w_diff_820,\
                    w_divid_730,w_diff_730,w_divid_640,w_diff_640,w_divid_550,w_diff_550,w_divid_721,w_diff_721,\
                    w_divid_532,w_diff_532,w_divid_111,w_diff_111,w_divid_190,w_diff_190,w_divid_280,w_diff_280,\
                    w_divid_370,w_diff_370,w_divid_460,w_diff_460,w_divid_127,w_diff_127,w_divid_235,w_diff_235,\
                    spread, Best_Ask, Best_Bid]).T

    return pd.DataFrame(data)#,traded_1 #, columns = ['label', 'rise', 'depth_divid', 'depth_diff'])

def genData(quotefile, traded_time):

    time_sec, time_millis, bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,\
            bid_quantity_2,bid_quantity_3,ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,\
            ask_quantity_2,ask_quantity_3 = order_book(quotefile)

    time_second,time_second_basic = time_transform(time_sec, time_millis)

    time_second_basic, [time_second, \
    bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,bid_quantity_2,bid_quantity_3, \
    ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,ask_quantity_2,ask_quantity_3] \
        = extract_by_sec_basic([[0, TIME_INTV_1_15], [TIME_INTV_1_30, TIME_INTV_2_30], \
        [TIME_INTV_4_30, TIME_INTV_6_00]], time_second_basic,[time_second, \
         bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,bid_quantity_2,bid_quantity_3, \
         ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,ask_quantity_2,ask_quantity_3])

    before_time = 60.0 * 6
    rise_ratio_ask_1 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 6 + 30
    rise_ratio_ask_2 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 7
    rise_ratio_ask_3 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 7 + 30
    rise_ratio_ask_4 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 8
    rise_ratio_ask_5 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 8 + 30
    rise_ratio_ask_6 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 9
    rise_ratio_ask_7 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 9 + 30
    rise_ratio_ask_8 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 10
    rise_ratio_ask_9 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 10 + 30
    rise_ratio_ask_10 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 11
    rise_ratio_ask_11 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 11 + 30
    rise_ratio_ask_12 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 12
    rise_ratio_ask_13 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 12 + 30
    rise_ratio_ask_14 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 13
    rise_ratio_ask_15 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 13 + 30
    rise_ratio_ask_16 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 14
    rise_ratio_ask_17 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 14 + 30
    rise_ratio_ask_18 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 15
    rise_ratio_ask_19 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 15 + 30
    rise_ratio_ask_20 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 16
    rise_ratio_ask_21 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 16 + 30
    rise_ratio_ask_22 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 17
    rise_ratio_ask_23 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 17 + 30
    rise_ratio_ask_24 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 18
    rise_ratio_ask_25 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 18 + 30
    rise_ratio_ask_26 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 19
    rise_ratio_ask_27 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 19 + 30
    rise_ratio_ask_28 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 20
    rise_ratio_ask_29 = rise_ask(ask_price_1, time_second_basic, before_time)
    before_time = 60.0 * 20 + 30
    rise_ratio_ask_30 = rise_ask(ask_price_1, time_second_basic, before_time)

    #Weight Depth
    w1,w2,w3 = [100.0, 0.0, 0.0]
    W_AB_100 , W_A_B_100 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [0.0, 100.0, 0.0]
    W_AB_010 , W_A_B_010 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [0.0, 0.0, 100.0]
    W_AB_001 , W_A_B_001 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [90.0, 10.0, 0.0]
    W_AB_910 , W_A_B_910 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [80.0, 20.0, 0.0]
    W_AB_820 , W_A_B_820 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [70.0, 30.0, 0.0]
    W_AB_730 , W_A_B_730 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [60.0, 40.0, 0.0]
    W_AB_640 , W_A_B_640 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [50.0, 50.0, 0.0]
    W_AB_550 , W_A_B_550 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [70.0, 20.0, 10.0]
    W_AB_721 , W_A_B_721 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [50.0, 30.0, 20.0]
    W_AB_532 , W_A_B_532 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [1.0, 1.0, 1.0]
    W_AB_111 , W_A_B_111 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [10.0, 90.0, 1.0]
    W_AB_190 , W_A_B_190 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [20.0, 80.0, 0.0]
    W_AB_280 , W_A_B_280 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [30.0, 70.0, 0.0]
    W_AB_370 , W_A_B_370 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [40.0, 60.0, 0.0]
    W_AB_460 , W_A_B_460 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [10.0, 20.0, 70.0]
    W_AB_127 , W_A_B_127 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)
    w1,w2,w3 = [20.0, 30.0, 50.0]
    W_AB_235 , W_A_B_235 = weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3)

    data_DF =\
    feature_DataFrame(traded_time,time_second_basic,bid_price_1,ask_price_1,rise_ratio_ask_1,\
                         rise_ratio_ask_2,rise_ratio_ask_3,rise_ratio_ask_4,rise_ratio_ask_5,\
                         rise_ratio_ask_6,rise_ratio_ask_7,rise_ratio_ask_8,rise_ratio_ask_9,\
                         rise_ratio_ask_10,rise_ratio_ask_11,rise_ratio_ask_12,rise_ratio_ask_13,\
                         rise_ratio_ask_14,rise_ratio_ask_15,rise_ratio_ask_16,rise_ratio_ask_17,\
                         rise_ratio_ask_18,rise_ratio_ask_19,rise_ratio_ask_20,rise_ratio_ask_21,\
                         rise_ratio_ask_22,rise_ratio_ask_23,rise_ratio_ask_24,rise_ratio_ask_25,\
                         rise_ratio_ask_26,rise_ratio_ask_27,rise_ratio_ask_28,rise_ratio_ask_29,\
                         rise_ratio_ask_30,W_AB_100, W_A_B_100, W_AB_010, W_A_B_010, W_AB_001,\
                         W_A_B_001, W_AB_910, W_A_B_910, W_AB_820, W_A_B_820, W_AB_730 , W_A_B_730,\
                         W_AB_640, W_A_B_640, W_AB_550, W_A_B_550,W_AB_721, W_A_B_721, W_AB_532,\
                         W_A_B_532, W_AB_111, W_A_B_111, W_AB_190, W_A_B_190, W_AB_280 , W_A_B_280,\
                         W_AB_370, W_A_B_370, W_AB_460, W_A_B_460, W_AB_127, W_A_B_127, W_AB_235, W_A_B_235)

    return data_DF,len(W_AB_111)

def train_test_to_csv(quotefilepath, quotefilename, traded_time):

    quotefile = os.path.join(quotefilepath, quotefilename+'.csv')
    data_feature_label,len_ = genData(quotefile, traded_time)
    save_path = './' + quotefilename +'_label_feature.csv'
    data_feature_label.to_csv(save_path, index = False)
    return

if __name__ == '__main__':

    parms = parsePara()
    contract = parms['contract']
    datelist = parms['datelist']
    # quotefilepath = 'D:\\高频资料\\20170103'

    datelist = datelist.split(',')
    for date in datelist:
        print('----------------- ', contract, ' ---- ', date, ' -----------------')
        quotefilepath = 'E:\\高频五档行情\\dce\\' + date
        quotefilename = contract + '_' + date
        traded_time = 600 #持仓最长10分钟

        train_test_to_csv(quotefilepath, quotefilename, traded_time)
