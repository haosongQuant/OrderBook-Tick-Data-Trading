#coding=utf-8
__author__ = '郝松'
import getopt
import sys

def parsePara():
    opts, args = getopt.getopt(sys.argv[1:], "", ["contract=", "datelist=", "tradeDir="])
    parms = {}
    for opt, val in opts:
        if opt == '--contract':
            parms['contract'] = val
        elif opt == '--datelist':
            parms['datelist'] = val
        elif opt == '--tradeDir':
            parms['tradeDir'] = val
    if 'contract' not in parms.keys():
        print('please input contract code!')
        exit()
    if 'datelist' not in parms.keys():
        print('please input datelist!')
        exit()
    if 'tradeDir' not in parms.keys():
        parms['tradeDir'] = 'short'
    return parms

