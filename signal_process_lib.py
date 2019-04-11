#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:59:49 2019
Mode analysis lib for MHD instabilities.
@author: chunan@mail.ustc.edu.cn
"""
import numpy
import pylab

pi = numpy.pi

def SVD_analysis(sig_matrix=0, i_test=0, i_plot=1, i_check=[0,1,2,3,4,5]):
  '''
  use SVD to get the toroidal/poloidal mode number of MHD instabilities
  i_check is the order of singular values to be plot and check
  '''
  if i_test == 1:
    sig_matrix, t = test_data(dim=2, f_MHD=[300,800, 500], n=[3,0,2], Amp=[10,5,3], N=35)
    # print(sig_matrix.shape)
    
  N = sig_matrix.shape[1]  
  # conduct SVD to signal matrix
  u,s,vh = numpy.linalg.svd(sig_matrix)
  Fs = numpy.round(1/(t[1]-t[0]))
  
  if i_plot == 1:
    index = numpy.arange(1,len(s)+1)
    pylab.figure()
    ax0 = pylab.subplot(1,1,1)
    pylab.plot(index, s, 'o', color='red')
    pylab.xlabel('order (n)')
    pylab.ylabel('Amp (A.U)')
    pylab.title('singular values')
    pylab.grid('on')
    pylab.minorticks_on()
    single_axis_paras(ax0, fontsize=13)
    
    for i in range(0, len(i_check)):
      # get the time eigenvector and calculate Fourier frequency spectrum
      u_i = u[:, i_check[i]]
      A = numpy.fft.fft(u_i)
      NFFT = numpy.int(len(A)/2)
      f = numpy.linspace(0, Fs/2, NFFT)
      A_real = numpy.abs(A[0:NFFT])
      imax = numpy.argmax(A_real)
      
      # find the time singular value harmonics
      S_i = numpy.zeros([u.shape[1],vh.shape[0]])
      S_i[i_check[i],i_check[i]] = s[i_check[i]]
      # reconstuct singular value related signal matrix harmonic  
      M_i = numpy.dot(numpy.dot(u, S_i),vh)
      i_display = numpy.int(1/f[imax]*5/(1/Fs))
      t_display = t[0:i_display]
      M_display = M_i[0:i_display, :]
      index = numpy.arange(1, M_i.shape[1]+1)
      
      # get the space eigenvector
      theta = numpy.linspace(0,2*pi*(N-1)/N, N)
      v_i = vh.T[:, i_check[i]]
      
      theta_inter = numpy.linspace(0, 2*pi*(N*5-1)/(N*5), 50)
      v_inter = numpy.interp(theta_inter, theta, v_i)
      # fix the link problem between last and first points
      theta_inter[-1] = 0
      v_inter[-1] = v_inter[0]
      
      theta_basis = numpy.linspace(0, 2*pi, 50)
      v_basis = numpy.ones(theta_basis.shape)

      
      pylab.figure(figsize=(8,6))
      ax_i1 = pylab.subplot(2,2,1)
      pylab.plot(f/1000, A_real)
      pylab.plot(f[imax]/1000, A_real[imax],'o',color='red')

      pylab.xlim([0, Fs/2/1000])
      pylab.xlabel('$f$ (kHz)')
      pylab.ylabel('$Amp$ (A.U)')
      pylab.title('singular order = '+str(i_check[i]+1))  
      single_axis_paras(ax_i1,gridon=True,fontsize=13)
      
      
      ax_i2 = pylab.subplot(2,2,3)
      # print(i_display)
      # print(t_display.shape, index.shape, M_display.shape)
      pylab.contourf(t_display, index, M_display.T, 50, cmap='seismic')
      pylab.xlabel('$t$ (s)')
      pylab.ylabel('mode structure contour')
      
      single_axis_paras(ax_i2,gridon=False,fontsize=13)      
     
      ax_i3 = pylab.subplot(2,2,2,polar=True)
      pylab.polar(theta, v_i+1, '*', color='magenta')
      pylab.polar(theta_inter, v_inter+1, '-', color='blue')
      pylab.polar(theta_basis, v_basis, '--', color='black')
      pylab.legend(['space eigenvector', 'interped curve', 'basis value = 1'], bbox_to_anchor=(0.83, -0.08))
      
      # set the positions of the subplots            
      ax_i1.set_position([0.1, 0.55, 0.3, 0.35])
      ax_i2.set_position([0.1, 0.1, 0.3, 0.35])
      ax_i3.set_position([0.45, 0.25, 0.5, 0.6])
      
  return u,s,vh




def test_data(dim=1, f_MHD = [0.1*1000, 0.2*1000], shift=0, n=[0, 3], Amp=[10,5], N=15):
  ''' generate test signal '''
  pylab.close('all')
  print('test data is used: dim = ' + str(dim))
  f_MHD = numpy.array(f_MHD)
  n = numpy.array(n)
  Fs = 2*1000    
  dt = 1.0/Fs
  t = numpy.arange(1, 1.5, dt)
  if dim == 1:
    # generate 1 dimensional test signal
    # calculate sampling frequency
    f1 = 100
    f2 = 250
    f3 = 500        
    y1 = numpy.sin(2*pi*f1*t)
    y2 = numpy.sin(2*pi*f2*t)
    y3 = numpy.sin(2*pi*f3*t)
    + 0.01*numpy.random.normal(t.shape)
    sig = y1 + y2 + y3
  elif dim == 2:
    # number of imaginary coils
    # N = 15
    sig = numpy.zeros([len(t), N])
    for i in range(0, N):
      if len(f_MHD) == 1:
        sig[:,i] = numpy.cos(2*pi*f_MHD + i/N*n*2*pi)
      elif len(f_MHD) > 1:
        for j in range(0, len(f_MHD)):
          sig[:,i] = sig[:,i] + Amp[j]*numpy.cos(2*pi*f_MHD[j]*t + i/N*n[j]*2*pi)
          #print(f_MHD[j])
    sig = sig + 0.05*numpy.random.randn(sig.shape[0],sig.shape[1])
  return sig, t




def single_axis_paras(ax, fontsize=15, numpoints=1, gridon =False, tick_pos='all') :
    """parameters for a paticular axis in plot"""
    pylab.rcParams['legend.numpoints'] = numpoints
    pylab.rcParams.update({'font.size': fontsize})
    pylab.rc('font', size=fontsize)        # controls default text sizes
    pylab.rc('axes', titlesize=fontsize)   # fontsize of the axes title
    pylab.rc('axes', labelsize=fontsize)   # fontsize of the x and y labels
    pylab.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    pylab.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
    pylab.rc('legend', fontsize=fontsize)  # legend fontsize
    pylab.rc('figure', titlesize=fontsize) # fontsize of the figure title    
    
    if gridon == True:
        ax.grid('on',linestyle='--')
    ax.minorticks_on()

    ax.tick_params(which = 'major', labelsize = fontsize, width = 1.5, \
                   length = 8, direction='in')#, top='on', right='on')
    ax.tick_params(which = 'minor', width = 1, length = 4, direction='in')#, \
                   #bottom='on', top='on', left='on', right='on')
    if tick_pos == 'left':
      ax.tick_params(which='both', left='on', right='off', top='on', bottom='on')
    elif tick_pos == 'right':
      ax.tick_params(which='both', left='off', right='on', top='on', bottom='on')
    elif tick_pos == 'top':
      ax.tick_params(which='both', left='on', right='on', top='on', bottom='off')
    elif tick_pos == 'bottom':
      ax.tick_params(which='both', left='on', right='on', top='off', botton='on')
    elif tick_pos == 'all':
      ax.tick_params(which='both', bottom='on', top='on', left='on', right='on')



i_test = 1
if i_test == 1:
  u,s,vh = SVD_analysis(i_test=1)
  pylab.show()
  