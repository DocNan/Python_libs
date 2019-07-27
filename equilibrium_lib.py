#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:48:03 2019
module to manipulate the equilibrium file and flux coordinates
@author: nan
"""
import numpy
import pylab
import os
import sys
import re
from scipy import integrate
from scipy import interpolate
pi = numpy.pi

#from . import math_lib




# ------------------------------------------------------------------------------
def read_eqdsk(gfile=0, i_plot=0, i_test=0):
    '''
    gdata = read_eqdsk(gfile=0, i_plot=0, i_test=0)
    Read eqdsk format equilibrium gfile with python. eqdsk gfile is the standard
    equilibrium data output of equilibrium reconstruction code like EFIT, CHEASE, etc
    In this module, gfile will be read as a dictionary, each variable is save as
    an element. Details of each element refer to the introduction  in namelist.
    An element is called via gdata['element'], eg q = gdata['q'].
    chunan@mail.ustc.edu.cn 2019.07.25, Hefei, China.
    '''
    if i_test == 1:
        gfile = './data/circle15.eqdsk'
    fid = open(gfile, 'r')
    g_string = fid.read()

    # read content of gfile
    gdata = {}
    # split gstring
    g_split = re.split(string=g_string, pattern='\n')

    # read title
    # stitle = g_string[1-1: 60]
    # gdata['title'] = stitle[1-1:48-1]
    gdata['title'] = g_split[0]

    # read dimension information
    # dims = numpy.array(re.findall('\d+', stitle[49-1:60]), dtype='int')
    stitle = g_split[0]
    print(stitle)
    dims = numpy.array(re.findall('\d+', stitle[len(stitle)-12:len(stitle)]), dtype='int')
    imfit = dims[0]
    nr = dims[1]
    nz = dims[2]

    # read dR, dZ, R0, Rl, Z0
    # scientific pattern from site: https://stackoverflow.com/questions/41668588/regex-to-match-scientific-notation
    pattern_sci = '([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+))'
    dum = numpy.array(re.findall(pattern=pattern_sci, string=g_split[1]), dtype='float')
    gdata['dR'] = dum[0]
    gdata['dZ'] = dum[1]
    gdata['R0'] = dum[2]
    gdata['Rl'] = dum[3]
    gdata['Z0'] = dum[4]

    # read R axsi, Z axis, PSI axis, PSI edge, B0
    dum = numpy.array(re.findall(pattern=pattern_sci, string=g_split[2]), dtype='float')
    gdata['Raxis'] = dum[0]
    gdata['Zaxis'] = dum[1]
    gdata['psiaxis'] = dum[2]
    gdata['psiedge'] = dum[3]
    gdata['B0'] = dum[4]

    # read Ip, PSI axis, zero, R axis, zero
    dum = numpy.array(re.findall(pattern=pattern_sci, string=g_split[3]), dtype='float')
    gdata['Ip'] = dum[0]

    # read Z axis, zero, PSI edge, zero, zero
    dum = numpy.array(re.findall(pattern=pattern_sci, string=g_split[4]), dtype='float')

    # read F array: nr*1 array
    index = 4 + 1
    delta_index = numpy.int(numpy.ceil(nr/5))
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
    gdata['F'] = dum

    # read pressure P: nr*1 array
    index = index + delta_index
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
    gdata['P'] = dum

    # read FFprime: nr*1 array
    index = index + delta_index
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
    gdata['FFprime'] = dum

    # read Pprime: nr*1 array
    index = index + delta_index
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
    gdata['Pprime'] = dum

    # read PSI(R, Z): nr*nr matrix
    index = index + delta_index
    delta_index_2 = numpy.int(numpy.ceil(nr*nz/5))
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index_2])), dtype='float')
    gdata['PSI'] = numpy.reshape(dum, (nr, nz), order='F')

    # read q: nr*1 array
    index = index + delta_index_2
    dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
    gdata['q'] = dum

    # read boundary parameters
    index = index + delta_index
    dum = numpy.array(re.findall(pattern='\d+', string=str(g_split[index])), dtype='int')
    nbbound = dum[0]
    nblim = dum[1]

    # read LCS
    if nbbound > 0:
        index = index + 1
        delta_index = numpy.int(numpy.ceil(nbbound*2/5))
        dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
        # dum = dum[0:nbbound*2]
        dum = numpy.reshape(dum, (2, nbbound), order='F')
        gdata['Rlcs'] = dum[0,:] 
        gdata['Zlcs'] = dum[1,:] 

    # read Limitor
    if nblim > 0:
        index = index + delta_index
        delta_index = numpy.int(numpy.ceil(nblim*2/5))
        dum = numpy.array(re.findall(pattern=pattern_sci, string=str(g_split[index:index+delta_index])), dtype='float')
        dum = numpy.reshape(dum, (2, nblim), order='F')
        gdata['Rlim'] = dum[0,:] 
        gdata['Zlim'] = dum[1,:]

    # generate R,Z grid
        gdata['rg'] = numpy.linspace(gdata['Rl'], gdata['Rl']+gdata['dR'], nr)
        gdata['zg'] = numpy.linspace(gdata['Z0']-gdata['dZ']/2, gdata['Z0']+gdata['dZ']/2, nz)
        gdata['gpsi'] = numpy.linspace(0, 1, nr)

    if i_plot == 1:
        pylab.figure(figsize=(10,7))
        pylab.subplot(1,2,1)
        pylab.contourf(gdata['rg'], gdata['zg'], gdata['PSI'].T, 100, cmap='jet')
        pylab.colorbar()
        pylab.plot(gdata['Raxis'], gdata['Zaxis'], 'o', color='red', label=r'$B_0$ axis')
        if nbbound > 0:
            pylab.plot(gdata['Rlcs'], gdata['Zlcs'], '--', color='magenta', label='LCFS')
        if nblim > 0:
            pylab.plot(gdata['Rlim'], gdata['Zlim'], '-', color='black', label='Limitor')
            pylab.axis('equal')
            pylab.xlabel('R (m)')
            pylab.ylabel('Z (m)')
            pylab.title(gdata['title'])
        pylab.legend()
        # pylab.legend(['PSI P', 'LCFS', 'Limitor'])
    
        pylab.subplot(2,2,2)
        pylab.plot(gdata['gpsi'], gdata['q'], '-o', label='q')
        pylab.plot(gdata['gpsi'], gdata['F'], '-o', label='F')
        pylab.plot(gdata['gpsi'], gdata['FFprime'], '-o', label='FFprime')  
        pylab.legend()

        pylab.subplot(2,2,4)
        pylab.plot(gdata['gpsi'], gdata['P'], '-o', label='P')
        pylab.plot(gdata['gpsi'], gdata['Pprime'], '-o', label='Pprime')
        pylab.xlabel(r'$\bar{\Psi}_p$')
        pylab.legend()
        pylab.tight_layout()
        #pylab.legend(['q', 'F', 'P', 'FFprime', 'Pprime'])
  
    fid.close()
    # prepare gdata parameter introduction
    gdata['namelist'] = [
            '[dR, dZ]: plasma width and length in (R, Z) directions',
            '[R0, Z0]: plasma geometric center (R, Z) positions',
            '[Rl]: plasma left edge board position',
            '[Raxis, Zaxis]: magnetic axis (R, Z) positions',
            '[psiaxis, psiedge]: poloidal magnetic flux at axis and edge of plamsa',
            '[B0]: toroidal magnetic field at R0',
            '[Ip]: toroidal plasma current',
            '[F]: F=R*B is poloidal current, which is a flux function of normalized poloical flux (F=F(\psi_p))',
            '[P]: pressure, which is also a function of normal psi_p',
            '[FFprime]: FFprime=Fprime*d(Fprime)/dpsi_p',
            '[Pprime]: Pprime=dP/dpsi_p',
            '[PSI]: poloidal magnetic flux in the (R,Z) grid',
            '[q]: safety factor, which is a function of normal psi_p',
            '[Rlcs, Zlcs]: (R,Z) position of the last closed flux surface',
            '[Rlim, Zlim]: (R,Z) position of the limitor wall',
            '[rg, zg]: (R,Z) grids array for PSI',
            '[gpsi]: normal psi_p',
            '[namelist]: namelist for all output variables']
    return gdata




# ------------------------------------------------------------------------------
# write eqdsk format file based on gdata read with python code
def write_eqdsk(newfile=0, gdata=0, i_test=0, i_plot=0):
    '''
    write_eqdsk(newfile=0, gdata=0, i_test=0, i_plot=0)
    write eqdsk format tokamak equilibrium file based on the gdata dictionary 
    data file information.
    chunan@mail.ustc.edu.cn
    '''
    if i_test == 1: 
        newfile = './data/gfiletest.output'
        gdata = read_eqdsk(i_test=1, i_plot=0)
        gdata['title'] = '  TEST    00/00/0000           '

    fid = open(newfile, 'w')
    # write 1st line: title, imfit, nr, nz
    fid.writelines(gdata['title'] + ' 3 ' + str(len(gdata['rg'])) + ' ' + str(len(gdata['zg'])) + '\n')

    # write 2nd line: dr, dZ, R0, Rl, Z0
    fid.writelines('{:16.9E}'.format(gdata['dR']) + '{:16.9E}'.format(gdata['dZ']) \
                   + '{:16.9E}'.format(gdata['R0']) + '{:16.9E}'.format(gdata['Rl']) \
                   + '{:16.9E}'.format(gdata['Z0']) + '\n')

    # write third line: Raxis, Zaxis, psiaxis, psiedge, B0
    fid.writelines('{:16.9E}'.format(gdata['Raxis']) + '{:16.9E}'.format(gdata['Zaxis']) \
                   + '{:16.9E}'.format(gdata['psiaxis']) + '{:16.9E}'.format(gdata['psiedge']) \
                   + '{:16.9E}'.format(gdata['B0']) + '\n')

    # write 4th line: Ip, PSI axis, zero, R axis, zero
    fid.writelines('{:16.9E}'.format(gdata['Ip']) + '{:16.9E}'.format(gdata['psiaxis']) \
                   + '{:16.9E}'.format(0) + '{:16.9E}'.format(gdata['Raxis']) \
                   + '{:16.9E}'.format(0) + '\n') 

    # write 5th line: Z axis, zero, PSI edge, zero, zero
    fid.writelines('{:16.9E}'.format(gdata['Zaxis']) + '{:16.9E}'.format(0) \
                   + '{:16.9E}'.format(gdata['psiedge']) + '{:16.9E}'.format(0) \
                   + '{:16.9E}'.format(0) + '\n')

    # write array: F
    ngrid = len(gdata['q'])
    string = []
    for i in range(0, ngrid):
        string.append('{:16.9E}'.format(gdata['F'][i]))
        if ((numpy.mod(i+1, 5) == 0) and (i != 0)) or (i == ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write array: P
    string = []
    for i in range(0, ngrid):
        string.append('{:16.9E}'.format(gdata['P'][i]))
        if (numpy.mod(i+1, 5) == 0 and i!= 0) or (i == ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write array: FFprime
    string = []
    for i in range(0, ngrid):
        string.append('{:16.9E}'.format(gdata['FFprime'][i]))
        if (numpy.mod(i+1, 5) == 0 and i != 0) or (i == ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write array: Pprime
    string = []
    for i in range(0, ngrid):
        string.append('{:16.9E}'.format(gdata['Pprime'][i]))
        if (numpy.mod(i+1, 5) == 0 and i != 0) or (i == ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write matrix: PSI
    string = []
    PSI = gdata['PSI']
    PSI_1D = numpy.reshape(PSI, (ngrid*ngrid), order='F')
    for i in range(0, ngrid*ngrid):
        string.append('{:16.9E}'.format(PSI_1D[i]))
        if (numpy.mod(i+1, 5) == 0 and (i != 0)) or (i == ngrid*ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write array: q
    string = []
    for i in range(0, ngrid):
        string.append('{:16.9E}'.format(gdata['q'][i]))
        if (numpy.mod(i+1, 5) == 0 and i != 0) or (i == ngrid-1):
            string.append('\n')
    fid.writelines(string)

    # write numbers: nlcs, nlim
    nlcs = len(gdata['Rlcs'])
    nlim = len(gdata['Rlim'])
    fid.writelines('  ' + str(nlcs) + '   ' + str(nlim) + '\n')

    # write array: Rlcs, Zlcs
    RZ_array = numpy.zeros((2,nlcs))
    RZ_array[0,:] = gdata['Rlcs']
    RZ_array[1,:] = gdata['Zlcs']
    RZ_1D = numpy.reshape(RZ_array, (2*nlcs), order='F')
    string = []
    for i in range(0, 2*nlcs):
        string.append('{:16.9E}'.format(RZ_1D[i]))
        if (numpy.mod(i+1, 5) == 0 and i != 0) or (i == 2*nlcs-1):
            string.append('\n')
    fid.writelines(string)

    # write array: Rlim, Zlim
    RZ_array = numpy.zeros((2,nlim))
    RZ_array[0,:] = gdata['Rlim']
    RZ_array[1,:] = gdata['Zlim']
    RZ_1D = numpy.reshape(RZ_array, (2*nlim), order='F')
    string = []
    for i in range(0, 2*nlim):
        string.append('{:16.9E}'.format(RZ_1D[i]))
        if (numpy.mod(i+1, 5) == 0 and i != 0) or (i == 2*nlim-1):
            string.append('\n')
    fid.writelines(string)
    
    fid.close()

    if i_plot == 1:
        read_eqdsk(gfile=newfile, i_plot=1)
    return




pylab.close('all')
read_eqdsk(i_plot=1, i_test=1)
write_eqdsk(i_test=1, i_plot=1)






