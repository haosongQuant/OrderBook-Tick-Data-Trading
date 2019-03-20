#coding=utf-8
__author__ = 'haosong'
import numpy as np
from time_sec_def import *

def append_trade_label(tradedlist, tradeDir, openprice, counterprice, tradeCost):
    if tradeDir == 'short':
        if openprice > counterprice+tradeCost:
            tradedlist.append(1)
        else:
            tradedlist.append(0)
    elif tradeDir == 'long':
        if openprice < counterprice-tradeCost:
            tradedlist.append(1)
        else:
            tradedlist.append(0)
    else:
        print('error: trade direction is neither short nor long, tradeDir=', tradeDir)
        exit()

    return

def traded_label_one_second(time1,time2,time_second_basic,bid_price_1,ask_price_1,traded_time,\
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
                            W_AB_235, W_A_B_235, tradeDir = 'short', tradeCost = 0.0):
    global index

    traded = []

    rise_ratio_second_1 = []
    rise_ratio_second_2 = []
    rise_ratio_second_3 = []
    rise_ratio_second_4 = []
    rise_ratio_second_5 = []
    rise_ratio_second_6 = []
    rise_ratio_second_7 = []
    rise_ratio_second_8 = []
    rise_ratio_second_9 = []
    rise_ratio_second_10 = []
    rise_ratio_second_11 = []
    rise_ratio_second_12 = []
    rise_ratio_second_13 = []
    rise_ratio_second_14 = []
    rise_ratio_second_15 = []
    rise_ratio_second_16 = []
    rise_ratio_second_17 = []
    rise_ratio_second_18 = []
    rise_ratio_second_19 = []
    rise_ratio_second_20 = []
    rise_ratio_second_21 = []
    rise_ratio_second_22 = []
    rise_ratio_second_23 = []
    rise_ratio_second_24 = []
    rise_ratio_second_25 = []
    rise_ratio_second_26 = []
    rise_ratio_second_27 = []
    rise_ratio_second_28 = []
    rise_ratio_second_29 = []
    rise_ratio_second_30 = []

    w_divid_100 = []
    w_diff_100 = []
    w_divid_010 = []
    w_diff_010 = []
    w_divid_001 = []
    w_diff_001 = []
    w_divid_910 = []
    w_diff_910 = []
    w_divid_820 = []
    w_diff_820 = []
    w_divid_730 = []
    w_diff_730 = []
    w_divid_640 = []
    w_diff_640 = []
    w_divid_550 = []
    w_diff_550 = []
    w_divid_721 = []
    w_diff_721 = []
    w_divid_532 = []
    w_diff_532 = []
    w_divid_111 = []
    w_diff_111 = []
    w_divid_190 = []
    w_diff_190 = []
    w_divid_280 = []
    w_diff_280 = []
    w_divid_370 = []
    w_diff_370 = []
    w_divid_460 = []
    w_diff_460 = []
    w_divid_127 = []
    w_diff_127 = []
    w_divid_235 = []
    w_diff_235 = []

    Best_Ask = []
    Best_Bid = []
    spread = []

    # index_one = np.where(time_second_basic <= 0)[0][-1]

    close_time_basic = time_second_basic[-1]
    for i in np.arange(time1, time2):

        index_array = np.where((time_second_basic < i+1) & (time_second_basic >= i))[-1]

        if len(index_array) > 0:
            index = index_array[-1]
            openprice = bid_price_1[index] if tradeDir == 'short' else ask_price_1[index]
            if i < close_time_basic - traded_time:
                trade_time_end = np.where(time_second_basic <= i + traded_time)[0][-1]
                traded_time_counter_prices = ask_price_1[index:trade_time_end] if tradeDir == 'short' \
                                        else bid_price_1[index:trade_time_end]
                if len(traded_time_counter_prices) == 0:
                    traded.append(0)
                else:
                    counterprice = min(traded_time_counter_prices) if tradeDir == 'short' \
                              else max(traded_time_counter_prices)
                    append_trade_label(traded, tradeDir, openprice, counterprice, tradeCost)
            elif i >= close_time_basic - traded_time:
                counterprice = ask_price_1[-1] if tradeDir == 'short' else bid_price_1[-1]
                append_trade_label(traded, tradeDir, openprice, counterprice, tradeCost)

            rise_ratio_second_1.append(rise_ratio_ask_1[ index ])
            rise_ratio_second_2.append(rise_ratio_ask_2[ index ])
            rise_ratio_second_3.append(rise_ratio_ask_3[ index ])
            rise_ratio_second_4.append(rise_ratio_ask_4[ index ])
            rise_ratio_second_5.append(rise_ratio_ask_5[ index ])
            rise_ratio_second_6.append(rise_ratio_ask_6[ index ])
            rise_ratio_second_7.append(rise_ratio_ask_7[ index ])
            rise_ratio_second_8.append(rise_ratio_ask_8[ index ])
            rise_ratio_second_9.append(rise_ratio_ask_9[ index ])
            rise_ratio_second_10.append(rise_ratio_ask_10[ index ])
            rise_ratio_second_11.append(rise_ratio_ask_11[ index ])
            rise_ratio_second_12.append(rise_ratio_ask_12[ index ])
            rise_ratio_second_13.append(rise_ratio_ask_13[ index ])
            rise_ratio_second_14.append(rise_ratio_ask_14[ index ])
            rise_ratio_second_15.append(rise_ratio_ask_15[ index ])
            rise_ratio_second_16.append(rise_ratio_ask_16[ index ])
            rise_ratio_second_17.append(rise_ratio_ask_17[ index ])
            rise_ratio_second_18.append(rise_ratio_ask_18[ index ])
            rise_ratio_second_19.append(rise_ratio_ask_19[ index ])
            rise_ratio_second_20.append(rise_ratio_ask_20[ index ])
            rise_ratio_second_21.append(rise_ratio_ask_21[ index ])
            rise_ratio_second_22.append(rise_ratio_ask_22[ index ])
            rise_ratio_second_23.append(rise_ratio_ask_23[ index ])
            rise_ratio_second_24.append(rise_ratio_ask_24[ index ])
            rise_ratio_second_25.append(rise_ratio_ask_25[ index ])
            rise_ratio_second_26.append(rise_ratio_ask_26[ index ])
            rise_ratio_second_27.append(rise_ratio_ask_27[ index ])
            rise_ratio_second_28.append(rise_ratio_ask_28[ index ])
            rise_ratio_second_29.append(rise_ratio_ask_29[ index ])
            rise_ratio_second_30.append(rise_ratio_ask_30[ index ])

            w_divid_100.append(W_AB_100[ index ])
            w_diff_100.append(W_A_B_100[ index ])
            w_divid_010.append(W_AB_010[ index ])
            w_diff_010.append(W_A_B_010[ index ])
            w_divid_001.append(W_AB_001[ index ])
            w_diff_001.append(W_A_B_001[ index ])
            w_divid_910.append(W_AB_910[ index ])
            w_diff_910.append(W_A_B_910[ index ])
            w_divid_820.append(W_AB_820[ index ])
            w_diff_820.append(W_A_B_820[ index ])
            w_divid_730.append(W_AB_730[ index ])
            w_diff_730.append(W_A_B_730[ index ])
            w_divid_640.append(W_AB_640[ index ])
            w_diff_640.append(W_A_B_640[ index ])
            w_divid_550.append(W_AB_550[ index ])
            w_diff_550.append(W_A_B_550[ index ])
            w_divid_721.append(W_AB_721[ index ])
            w_diff_721.append(W_A_B_721[ index ])
            w_divid_532.append(W_AB_532[ index ])
            w_diff_532.append(W_A_B_532[ index ])
            w_divid_111.append(W_AB_111[ index ])
            w_diff_111.append(W_A_B_111[ index ])
            w_divid_190.append(W_AB_190[ index ])
            w_diff_190.append(W_A_B_190[ index ])
            w_divid_280.append(W_AB_280[ index ])
            w_diff_280.append(W_A_B_280[ index ])
            w_divid_370.append(W_AB_370[ index ])
            w_diff_370.append(W_A_B_370[ index ])
            w_divid_460.append(W_AB_460[ index ])
            w_diff_460.append(W_A_B_460[ index ])
            w_divid_127.append(W_AB_127[ index ])
            w_diff_127.append(W_A_B_127[ index ])
            w_divid_235.append(W_AB_235[ index ])
            w_diff_235.append(W_A_B_235[ index ])

            Best_Ask.append(ask_price_1[index])
            Best_Bid.append(bid_price_1[index])
            spread.append(ask_price_1[index] - bid_price_1[index])

        elif len(index_array) == 0:
            print('warning: index_array is empty!')

            traded.append(traded[-1])
            rise_ratio_second_1.append(rise_ratio_second_1[-1])
            rise_ratio_second_2.append(rise_ratio_second_2[-1])
            rise_ratio_second_3.append(rise_ratio_second_3[-1])
            rise_ratio_second_4.append(rise_ratio_second_4[-1])
            rise_ratio_second_5.append(rise_ratio_second_5[-1])
            rise_ratio_second_6.append(rise_ratio_second_6[-1])
            rise_ratio_second_7.append(rise_ratio_second_7[-1])
            rise_ratio_second_8.append(rise_ratio_second_8[-1])
            rise_ratio_second_9.append(rise_ratio_second_9[-1])
            rise_ratio_second_10.append(rise_ratio_second_10[-1])
            rise_ratio_second_11.append(rise_ratio_second_11[-1])
            rise_ratio_second_12.append(rise_ratio_second_12[-1])
            rise_ratio_second_13.append(rise_ratio_second_13[-1])
            rise_ratio_second_14.append(rise_ratio_second_14[-1])
            rise_ratio_second_15.append(rise_ratio_second_15[-1])
            rise_ratio_second_16.append(rise_ratio_second_16[-1])
            rise_ratio_second_17.append(rise_ratio_second_17[-1])
            rise_ratio_second_18.append(rise_ratio_second_18[-1])
            rise_ratio_second_19.append(rise_ratio_second_19[-1])
            rise_ratio_second_20.append(rise_ratio_second_20[-1])
            rise_ratio_second_21.append(rise_ratio_second_21[-1])
            rise_ratio_second_22.append(rise_ratio_second_22[-1])
            rise_ratio_second_23.append(rise_ratio_second_23[-1])
            rise_ratio_second_24.append(rise_ratio_second_24[-1])
            rise_ratio_second_25.append(rise_ratio_second_25[-1])
            rise_ratio_second_26.append(rise_ratio_second_26[-1])
            rise_ratio_second_27.append(rise_ratio_second_27[-1])
            rise_ratio_second_28.append(rise_ratio_second_28[-1])
            rise_ratio_second_29.append(rise_ratio_second_29[-1])
            rise_ratio_second_30.append(rise_ratio_second_30[-1])

            w_divid_100.append(w_divid_100[-1])
            w_diff_100.append(w_diff_100[-1])
            w_divid_010.append(w_divid_010[-1])
            w_diff_010.append(w_diff_010[-1])
            w_divid_001.append(w_divid_001[-1])
            w_diff_001.append(w_diff_001[-1])
            w_divid_910.append(w_divid_910[-1])
            w_diff_910.append(w_diff_910[-1])
            w_divid_820.append(w_divid_820[-1])
            w_diff_820.append(w_diff_820[-1])
            w_divid_730.append(w_divid_730[-1])
            w_diff_730.append(w_diff_730[-1])
            w_divid_640.append(w_divid_640[-1])
            w_diff_640.append(w_diff_640[-1])
            w_divid_550.append(w_divid_550[-1])
            w_diff_550.append(w_diff_550[-1])
            w_divid_721.append(w_divid_721[-1])
            w_diff_721.append(w_diff_721[-1])
            w_divid_532.append(w_divid_532[-1])
            w_diff_532.append(w_diff_532[-1])
            w_divid_111.append(w_divid_111[-1])
            w_diff_111.append(w_diff_111[-1])
            w_divid_190.append(w_divid_190[-1])
            w_diff_190.append(w_diff_190[-1])
            w_divid_280.append(w_divid_280[-1])
            w_diff_280.append(w_diff_280[-1])
            w_divid_370.append(w_divid_370[-1])
            w_diff_370.append(w_diff_370[-1])
            w_divid_460.append(w_divid_460[-1])
            w_diff_460.append(w_diff_460[-1])
            w_divid_127.append(w_divid_127[-1])
            w_diff_127.append(w_diff_127[-1])
            w_divid_235.append(w_divid_235[-1])
            w_diff_235.append(w_diff_235[-1])

            Best_Ask.append(ask_price_1[index])
            Best_Bid.append(bid_price_1[index])
            spread.append(ask_price_1[index] - bid_price_1[index])

    return traded,rise_ratio_second_1,rise_ratio_second_2,rise_ratio_second_3,\
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
            w_divid_370,w_diff_370,w_divid_460,w_diff_460,w_divid_127,w_diff_127,w_divid_235,w_diff_235, \
            spread, Best_Ask, Best_Bid