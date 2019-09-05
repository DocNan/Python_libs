#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:59:49 2019
Mode analysis lib for MHD instabilities.
@author: chunan@mail.ustc.edu.cn
"""
import numpy
import pylab
from scipy import signal

pi = numpy.pi


def bicoherence(sig=0, t=0, N_slice=10, i_test=0, i_plot=0):
  '''
  Auto-bicoherence with fix to previous mistakes by splitting orignal signal into
  N slices. The the normalization shall be normal. 
  Reference: [Satoshi Hagihira et al, Anesth Analg 2001;93:966–70]
  Coding by Nan CHU on 2019.02.18
  '''
  if i_test == 1:
    i_plot = 1
    print('test data is used.')
    dt = 0.0003
    t = numpy.arange(1, 1.5, dt)
    f1 = 100
    f2 = 300
    f3 = f1 + f2
    f4 = 1200
    phi_1 = pi/12
    phi_2 = pi/3
    phi_3 = phi_1 + phi_2        
    phi_4 = 0
    # f3 is coupling of f1 and f3, while f4 is independ signal frequency
    # random noise is also added to the signal
    sig = numpy.cos(2*pi*f1*t + phi_1) + numpy.cos(2*pi*f2*t + phi_2)\
          + numpy.cos(2*pi*f3*t + phi_3) + 1*numpy.cos(2*pi*f4*t + phi_4)\
          + 0.1*numpy.random.randn(len(t))   
      
  # remove the mean value
  sig = sig - sig.mean()
  # split orignal signal to N slices
  dt = (t[2] - t[0])/2
  # get signal sampling frequency
  Fs = 1/dt
  i_window = numpy.arange(0, t.shape[0], numpy.int(t.shape[0]/N_slice))
  nfft = numpy.int(numpy.round(t.shape[0]/(2*N_slice))*2)
  # generate window function to do smooth slice edge
  sig_matrix = window_points(i_window, sig, nfft)
  win = signal.hamming(nfft)
  # convert the window to fit the shape of signal matrix
  # adjust shape of win function
  win.shape = [win.shape[0], 1]
  win_matrix = win*numpy.ones([1, sig_matrix.shape[1]])  
  # apply the window smooth to signal matrix before do fft
  sig_matrix = numpy.multiply(sig_matrix, win_matrix)
  # do fft for the windowed signal
  sig_matrix_fft = numpy.fft.fft(sig_matrix, axis=0)/nfft
  sig_matrix_fft = numpy.fft.fftshift(sig_matrix_fft)
  # remove the head and tail slices to avoid error by slice
  sig_fft_refine = sig_matrix_fft[:, 1:N_slice-1]
  # slice number without head and tail 
  N_refine = N_slice - 2
  # positive half of fft frequency
  f = numpy.linspace(-Fs/2.0, Fs/2.0, nfft)
  # create empty bispectrum and bicoherence matrix
  bi_spectrum = numpy.zeros([sig_fft_refine.shape[0], sig_fft_refine.shape[0]])
  BSP = numpy.zeros([sig_fft_refine.shape[0], sig_fft_refine.shape[0]])
  bi_coherence_s2 = numpy.zeros([sig_fft_refine.shape[0], sig_fft_refine.shape[0]])
  bico_fifj_s2 = numpy.zeros([sig_fft_refine.shape[0], sig_fft_refine.shape[0]])
  bico_fi_plus_fj_s2 = numpy.zeros([sig_fft_refine.shape[0], sig_fft_refine.shape[0]])
  
  for n in range(0, N_refine):
    for i in range(0, nfft):
      for j in range(0, nfft):
        f_ij_plus = f[i] + f[j]
        if numpy.abs(f_ij_plus) <= Fs/2.0:
          # confine the bispectrum within the Nquist frequency limit
          i_plus = find_time_points(f, f_ij_plus)
          i_plus = i_plus[0]
          # calculate bi-spectrum strength
          bi_spectrum[i, j] = bi_spectrum[i, j] \
            + (sig_fft_refine[i, n]*sig_fft_refine[j, n])*numpy.conj(sig_fft_refine[i_plus, n])
          bico_fifj_s2[i, j] = bico_fifj_s2[i, j] \
            + (numpy.abs(sig_fft_refine[i, n]*sig_fft_refine[j, n]))**2
          bico_fi_plus_fj_s2[i, j] = bico_fi_plus_fj_s2[i, j] \
          + (numpy.abs(sig_fft_refine[i_plus, n]))**2
        else:
          bi_spectrum[i, j] = 0
          bico_fifj_s2[i, j] = 1000
          bico_fi_plus_fj_s2[i, j]=1000
    
    if n == 0:
      # get the bispectrum from the first time slice
      BSP = bi_spectrum
  
  bi_coherence_s2 = (numpy.abs(bi_spectrum))**2/(bico_fifj_s2*bico_fi_plus_fj_s2)
  I = numpy.ones([len(f), len(f)])
  I_up = numpy.triu(I, k = 0)
  I_down = numpy.rot90(I_up.T)
  bico_meaningful = bi_coherence_s2*I_up*I_down
  
  if i_plot == 1:
    # plot signal bispectrum and bicoherence^2 for checking
    pylab.figure()
    pylab.contourf(f/1000, f/1000, numpy.abs(bi_spectrum), 50, cmap=pylab.cm.Spectral_r)
    pylab.colorbar()
    pylab.xlabel('f(kHz)')
    pylab.ylabel('f(kHz)')
    pylab.title('bispectrum')
    
    pylab.figure()
    pylab.contourf(f/1000, f/1000, bi_coherence_s2, 50, cmap=pylab.cm.Spectral_r)
    pylab.colorbar()
    pylab.xlabel('f(kHz)')
    pylab.ylabel('f(kHz)')
    pylab.title(r'$b^2(f_1, f_2)$ full region')    

    pylab.figure()
    pylab.contourf(f/1000, f/1000, bico_meaningful, 50, cmap=pylab.cm.Spectral_r)
    pylab.colorbar()
    pylab.xlabel('f(kHz)')
    pylab.ylabel('f(kHz)')
    pylab.title(r'$b^2(f_1, f_2)$ meaningful region')   
    pylab.xlim([0, Fs/2/1000])
    pylab.ylim([-Fs/4/1000, Fs/4/1000])
    
    f_1D, sig_fft_1D = fft_1D_2sides(sig = sig, Fs = Fs)
    pylab.figure()
    pylab.plot(f_1D/1000, numpy.abs(sig_fft_1D), linestyle='-', marker='o')
    pylab.xlabel('f(kHz)')
    pylab.ylabel('Amp(A.U)')
    pylab.title('Spectrum')
    pylab.xlim([0, Fs/2.0/1000])
    
    if i_test == 1:
      pylab.figure()
      pylab.contourf(f/1000, f/1000, numpy.abs(BSP), 50, cmap=pylab.cm.Spectral_r)
      pylab.colorbar()
      pylab.xlabel('f(kHz)')
      pylab.ylabel('f(kHz)')
      pylab.title('bispectrum from one slice')
  
  return bi_coherence_s2, bico_meaningful




def SVD_analysis(sig_matrix=0, t=0, i_test=0, i_plot=1, i_check=[0,1,2,3,4,5]):
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


      
      
# ------------------------------------------------------------------------------
# define function to take data in a small time window
# inputs: i_window should be an ndarray with 2D shape
def window_points(i_window=0, sig=0, nfft=1024, i_test=0):
    if i_test == 1:
        sig, t = test_data(dim=1)
        i_window = numpy.round([numpy.round(len(sig)/3.0), numpy.round(len(sig)/3.0*2)])
        print(i_window)    
        print(sig.shape)
        print(t.shape)
        nfft = 512
    
    # nfft should be a even number
    i_window=i_window.astype(int)
    nfft = numpy.int(nfft)
    half_win=numpy.int(nfft/2) # transform from round to int
    print('nfft, len(i_window):', nfft, i_window.shape[0])
    sig_window_matrix=numpy.zeros([nfft,i_window.shape[0]])
    for i in range(0,i_window.shape[0]):
        if i_window[i]>=half_win and i_window[i]<=sig.shape[0]-half_win :
            sig_window_matrix[:,i]=sig[i_window[i]-half_win:i_window[i]+half_win]
        elif i_window[i]<half_win and i_window[i]>=0:
            sig_window_matrix[:,i]=sig[0:nfft]
        elif i_window[i]>sig.shape[0]-half_win and i_window[i]<sig.shape[0]:
            sig_window_matrix[:,i]=sig[sig.shape[0]-nfft:sig.shape[0]]
        else:
            print('index out of sig range')
    return sig_window_matrix




def fft_1D_2sides(sig = 0, i_test = 0, Fs = 0, i_plot = 0):
    '''
    This function will record the fft value with range [-Fs/2, Fs/2]
    Where the normal fft has f range [0, Fs/2] || [-Fs/2, 0]
    chunan@ipp.ac.cn 2018.09.19
    '''
    if i_test == 1:
        # prepare default inputs
        dt = 0.001
        t = numpy.arange(1, 1.501, dt)
        sig = numpy.cos(2*pi*10*t)
        Fs = 1.0/dt
        i_plot = 1
        
    sig_fft = numpy.fft.fft(sig)
    f_len = len(sig_fft)
    sig_fft_2sides = numpy.concatenate([sig_fft[numpy.int(f_len/2) + 1:f_len], \
                     sig_fft[0:numpy.int(f_len/2) + 1]])
    # equalized the fft amplitude
    sig_fft_2sides = sig_fft_2sides/len(sig)
    f = numpy.linspace(-Fs/2.0, Fs/2.0, len(sig))
    if i_plot == 1:
        pylab.figure()
        pylab.plot(f/1000, numpy.abs(sig_fft_2sides),'-*')
        pylab.xlabel('f(kHz)')
    return f, sig_fft_2sides




def find_time_points(t=0, t_want=0, i_plot=0, i_test=0, i_warn=1, method='even'):
    """
    Find time index for an array of time points on time array t
    """
    if i_test == 1:
      print('ueven test data used')
      ind = numpy.linspace(-3, 2.2, 343)
      t = 10**ind
      t_want = numpy.array([9, 0.1, 0.05, 1.732, 12])
      i_plot = 1
    elif i_test == 2:
      t = numpy.linspace(0, 5, 50000)
      t_want = numpy.array([1.111, 3.222])
      i_plot = 1
      
    # convert a pure number t_want to a list with length and attribute
    if not(type(t_want) == list):
        t_want = [t_want]
        
        
    if numpy.min(t_want) < t[0] or numpy.max(t_want) > t[-1]:
        print('t[0]=',t[0])
        print('t[len(t)-1]',t[len(t)-1])
        raise Exception('Error: t_want goes out the range of t')
    # sum up the difference of time difference to judge whether it is even.
    dt_sum = numpy.sum(numpy.diff(numpy.diff(t))) 
    if dt_sum < 10**-10 or method == 'even':
        dt = (t[3] - t[0])/3.0
        i_want = numpy.round((t_want - t[0])/dt)
        # convert i_want to python style index that start from 0
        i_want = i_want - 1
    elif dt_sum > 10**-10 or method == 'uneven':
        if i_warn == 1 and method == 'even':
          print('Sum of ddt: ', dt_sum)
          print('Time array is not even, slow loop method used!')
        # i_want = numpy.ones(len(t_want))*-1
        i_want = numpy.zeros(len(t_want))
        for i in range(0, len(t_want)) :
            for j in range(0, len(t)) :
                if t_want[i] >= t[j] and t_want[i] < t[j+1] :
                    i_want[i] = j
    # convert index i_want to integers
    i_want = numpy.int_(i_want)
    if i_plot == 1:
        print('t_want: ', t_want)
        print('i_want: ', i_want)
        pylab.figure()
        x=numpy.arange(0, t.shape[0], 1)
        pylab.plot(x, t, '-o', color = 'blue', markersize=3)
        pylab.hold('on')
        pylab.plot(x[i_want], t[i_want], 'o', color = 'red')
        pylab.xlabel('index')
        pylab.ylabel('time (s)')
        pylab.grid('on')
        pylab.minorticks_on()
        pylab.tick_params(which = 'major', labelsize = 10, width = 2, 
                      length = 10, color = 'black')
        pylab.tick_params(which = 'minor', width = 1, length = 5)
    return i_want


  
  
i_test = 1
if i_test == 1:
  # u,s,vh = SVD_analysis(i_test=1)
  bicoherence(i_test=1)
  pylab.show()
  
