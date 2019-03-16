#coding=utf-8
__author__ = 'haosong'
import pandas as pd
import numpy as np

def order_book(quotefile):

    order_book = pd.read_csv(quotefile)

    time_sec = order_book['time'].values
    time_millis = order_book['millis'].values
    bid_price_1 = order_book['bid1'].astype('float').values
    bid_price_2 = order_book['bid2'].astype('float').values
    bid_price_3 = order_book['bid3'].astype('float').values
    bid_quantity_1 = order_book['b1size'].astype('float').values
    bid_quantity_2 = order_book['b2size'].astype('float').values
    bid_quantity_3 = order_book['b3size'].astype('float').values
    ask_price_1 = order_book['ask1'].astype('float').values
    ask_price_2 = order_book['ask2'].astype('float').values
    ask_price_3 = order_book['ask3'].astype('float').values
    ask_quantity_1 = order_book['a1size'].astype('float').values
    ask_quantity_2 = order_book['a2size'].astype('float').values
    ask_quantity_3 = order_book['a3size'].astype('float').values

    bid_quantity_1[np.isnan(bid_quantity_1)] = 0
    bid_quantity_2[np.isnan(bid_quantity_2)] = 0
    bid_quantity_3[np.isnan(bid_quantity_3)] = 0
    ask_quantity_1[np.isnan(ask_quantity_1)] = 0
    ask_quantity_2[np.isnan(ask_quantity_2)] = 0
    ask_quantity_3[np.isnan(ask_quantity_3)] = 0

    return time_sec, time_millis, bid_price_1,bid_price_2,bid_price_3,bid_quantity_1,\
            bid_quantity_2,bid_quantity_3,ask_price_1,ask_price_2,ask_price_3,ask_quantity_1,\
            ask_quantity_2,ask_quantity_3