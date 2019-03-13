#coding=utf-8
__author__ = 'haosong'
import numpy as np

def weight_pecentage(w1,w2,w3,ask_quantity_1,ask_quantity_2,ask_quantity_3,\
                     bid_quantity_1,bid_quantity_2,bid_quantity_3):

    Weight_Ask = (w1 * ask_quantity_1 + w2 * ask_quantity_2 + w3 * ask_quantity_3)
    Weight_Bid = (w1 * bid_quantity_1 + w2 * bid_quantity_2 + w3 * bid_quantity_3)
    W_AB = Weight_Ask/Weight_Bid
    W_A_B = (Weight_Ask - Weight_Bid)/(Weight_Ask + Weight_Bid)

    W_AB[np.isnan(W_AB)] = 1
    W_AB[np.isinf(W_AB)] = 1
    W_A_B[np.isnan(W_A_B)] = 0
    W_A_B[np.isinf(W_A_B)] = 0

    return W_AB, W_A_B