#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:13:53 2019
Statistic lib contains functions to undertake statistic analysis
@author: chunan
"""
import numpy
import pylab

# define useful constants
pi = numpy.pi

# calculation Standard Deviation (SD) ------------------------------------------
def standard_deviation(sig=0, i_test=0):
  if i_test == 1:
    sig = test_data(dim=1, size=50)
  
  SD = 0
  s_mean = numpy.mean(sig)
  N = len(sig)
  for i in range(0, N):
    SD = SD + (sig[i] - s_mean)**2
  SD = numpy.sqrt(SD/N)
  
  if i_test == 1:
    x = numpy.arange(0, N)
    pylab.figure()
    pylab.plot(x, sig,'o', color='blue', markersize=5)
    pylab.errorbar(0, s_mean, yerr=SD, fmt='s', color='green', markersize=5, capsize=3)
    pylab.legend(['sig', 'SD'], loc='upper right')
    pylab.xlabel('number')
    pylab.ylabel('value')
    pylab.title('standard deviation')
    print('mean', 'SD')
    print(s_mean, SD)
  return SD



# calculation Standard Error (SE) ----------------------------------------------
def standard_error(sig=0, i_test=0):
  '''
  standard error calculate just how certain the mean value we got shall be trusted!
  Reference:
  [1] http://berkeleysciencereview.com/errorbars-anyway/
  '''
  if i_test == 1:
    sig = test_data(dim=1, size=80)
  

  SD = standard_deviation(sig=sig)
  SE= SD/numpy.sqrt(len(sig))
  if i_test == 1:
    s_mean = numpy.mean(sig)
    x = numpy.arange(0, len(sig))
    pylab.figure()
    pylab.plot(x, sig,'o', color='blue', markersize=5)
    pylab.errorbar(0, s_mean, yerr=SD, fmt='s', color='green', markersize=5, capsize=3)
    pylab.errorbar(numpy.int(len(sig)/2), s_mean, yerr=SE, fmt='s', color='red', markersize=5, capsize=3)
    pylab.legend(['sig', 'SD', 'SE'], loc='upper right')
    pylab.xlabel('number')
    pylab.ylabel('value')
    pylab.title('standard error')
    print('mean', 'SD', 'SE')
    print(s_mean, SD, SE)
  
  return SE

  
  
  
def test_data(dim=1, size=30):
  if dim == 1:
    s = 5 + 0.1*numpy.random.randn(size)
    
  return s




i_test = 1
if i_test == 1:
  pylab.close('all')
  #standard_deviation(i_test=1)
  standard_error(i_test=1)  