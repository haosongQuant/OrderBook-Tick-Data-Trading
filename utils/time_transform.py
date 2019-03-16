#coding=utf-8
__author__ = 'haosong'
import numpy as np
from time_sec_def import *

def time_transform(time_sec, time_millis):
    time_second_basic = []
    time_second = []
    for i in np.arange(0,len(time_sec)):
        sec = time_sec[i]
        sec = sec.split(':')
        second = float(sec[0])*3600 + float(sec[1])*60+float(sec[2])
        second += float(time_millis[i]) / 1000
        time_second_basic.append(second - TIME_SEC_9_00 * 1.0)  #相对上午9:00的秒数
        time_second.append(second)
    return np.array(time_second),np.array(time_second_basic)

if __name__ == '__main__':
    time_sec = np.array(['8:59:00', '9:00:00', '9:00:00', '9:00:00', '9:00:01', '9:00:01', '9:00:01'])
    time_millis = np.array([399, 440, 691, 941, 185, 441, 691])

    time_second, time_second_basic = time_transform(time_sec, time_millis)
    print(time_second, time_second_basic)