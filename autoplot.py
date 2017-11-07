import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import yt
import pyart
import os

from yt.visualization.api import Streamlines
from yt.visualization.volume_rendering.api import Scene, VolumeSource, LineSource, PointSource
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

import readfunctions as rf
import lmasources as ls
from coordinateSystems import GeographicSystem, MapProjection

geo   = GeographicSystem()

for filename in os.listdir('/localdata/wind_analysis/'):
    if filename[-8:] == '_var.out':
        print (filename)

        dataf       = rf.outfile('/localdata/wind_analysis/'+filename)
        r,u,v,w     = dataf.read_data() # These are in z,y,x
        lon,lat,alt = dataf.latlon(geo)

        rx,ry,rz = dataf.data_grid()
        rx       = rx.reshape(np.shape(w))
        ry       = ry.reshape(np.shape(w))
        rz       = rz.reshape(np.shape(w))

        lma_directory = '/home/vanna.chmielewski/analyzedlightning/notgz/'
        lma_file = '{0}LYLOUT_{1}0.exported.dat'.format(lma_directory,dataf.time[:3])
        lmad = ls.lmafile(lma_file).readfile()

        if int(dataf.time[3]) > 7:
            sfi = datetime.datetime(2012,5,29,int(dataf.time[:2]),int(dataf.time[2:])) + datetime.timedelta(minutes=10)
            lma_file2 = '{0}LYLOUT_{1}0.exported.dat'.format(lma_directory,sfi.strftime('%H%M')[:3])
            lmad2 = ls.lmafile(lma_file2).readfile()
            lmad = pd.concat((lmad,lmad2))

        tstart = int(dataf.time[:2])*3600 + int(dataf.time[2:])*60
        tend   = tstart + 3*60
        mxchi  = 1.0
        mnnum  = 7.0

        sources = lmad[(lmad['time'] >= tstart)  & (lmad['time'] < tend) & 
                       (lmad['chi']  <= mxchi)   & (lmad['num'] >= mnnum)]
        psource = sources[sources['charge'] ==  3]
        nsource = sources[sources['charge'] == -3]

        exs,eys,ezs = geo.toECEF(np.array(sources['lon']),
                                 np.array(sources['lat']),
                                 np.array(sources['alt']))
        projl = MapProjection(projection='laea', lat_0=dataf.lat, lon_0=dataf.lon, alt_0=dataf.alt)
        nxs,nys,nzs = projl.fromECEF(exs,eys,ezs)

        binz = np.arange(dataf.iz,dataf.nz*dataf.dz+dataf.iz+dataf.dz,dataf.dz)-dataf.dz/2.
        biny = np.arange(dataf.iy,dataf.ny*dataf.dy+dataf.iy+dataf.dy,dataf.dy)-dataf.dy/2.
        binx = np.arange(dataf.ix,dataf.nx*dataf.dx+dataf.ix+dataf.dx,dataf.dx)-dataf.dx/2.

        negs = sources['charge'] == -3
        poss = sources['charge'] ==  3
        unas = sources['charge'] ==  0
        abins = (binx,biny,binz)

        all_counts = np.histogramdd(np.array([nxs,nys,nzs]).T,bins=abins)[0]
        p_counts   = np.histogramdd(np.array([nxs[poss],nys[poss],nzs[poss]]).T,bins=abins)[0]
        n_counts   = np.histogramdd(np.array([nxs[negs],nys[negs],nzs[negs]]).T,bins=abins)[0]

        #########################################
        plt.figure(figsize=(10,7))
        level = 9.2

        dist = np.abs(nzs-level*1e3)
        alpha = np.exp(-dist/400.)
        rgrid = int((level-0.2)*1e3/dataf.dy)
        # lgrid = np.argmin(np.abs(gnys[:,0,0] - mfrom0))

        plt.contourf(rx[rgrid,:,:],ry[rgrid,:,:],np.sum(w,axis=0),40,cmap=pyart.graph.cm.Carbone11,alpha=0.5,antialiased=True)
        plt.colorbar()
        # plt.contour(rx[rgrid,:,:],ry[rgrid,:,:],DLA[7,rgrid,:,:],colors='k',linewidths=0.2)
        plt.contour(rx[rgrid,:,:],ry[rgrid,:,:],np.sum(p_counts,axis=2).T,30,colors='k',linewidths=0.2)

        # step=3
        # plt.quiver(rx[rgrid,::step,::step],ry[rgrid,::step,::step],
        #             u[rgrid,::step,::step], v[rgrid,::step,::step],
        #            scale=1500,width=0.002,pivot='middle',
        #           )

        # Note the flashgrid is 0.2 km off
        # plt.contour(gnxs[:,:,rgrid//2],gnys[:,:,rgrid//2],np.mean(flashgrid['flash_footprint'][8,:,:,rgrid//2-1:rgrid//2+2],axis=2).T,
        #             levels=np.arange(5,1500,50),colors='purple',linewidths=0.5)
        # plt.contour(gnxs[lgrid,:,:],gnzs[lgrid,:,:],flashgrid['flash_footprint'][8,:,lgrid,:],colors='purple',linewidths=0.2)

        # c = np.asarray([(0,0,1,a) for a in alpha])[negs]
        # plt.scatter(nxs[negs],nys[negs],color=c,edgecolors=c,s=5)
        # c = np.asarray([(1,0,0,a) for a in alpha])[poss]
        # plt.scatter(nxs[poss],nys[poss],color=c,edgecolors=c,s=3)
        # c = np.asarray([(0.2,0.8,0,a) for a in alpha])[unas]
        # plt.scatter(nxs[unas],nys[unas],color=c,edgecolors=c,s=1)

        if dataf.time == '2357':
            j,i = np.unravel_index(np.argmax(np.sum(w[:,:60],axis=0)), np.sum(w,axis=0).shape)
        else:
            j,i = np.unravel_index(np.argmax(np.sum(w[:],axis=0)), np.sum(w,axis=0).shape)
        plt.scatter(i*500+4000,j*500-2000)

        # plt.xlim(50*dataf.dx,170*dataf.dx)
        # plt.ylim(0,40000)
        plt.tight_layout()
        plt.savefig('/localdata/wcheck_{0}.png'.format(dataf.time))
        plt.close()