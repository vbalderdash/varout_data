import numpy as np
import pandas as pd

from coordinateSystems import GeographicSystem, MapProjection

class outfile:
    def __init__(self,filename):
        self.file = filename
        self.time = filename[-12:-8]
        self.header = self.read_header()
        self.nx = int(self.header.iloc[0][0])
        self.ny = int(self.header.iloc[0][1])
        self.nz = int(self.header.iloc[0][2])
        self.lon = -self.header.iloc[1][0]
        self.lat,self.alt = self.header.iloc[1][1:]
        self.dx,self.dy,self.dz    = self.header.iloc[2]*1000
        self.ix,self.iy,self.iz    = self.header.iloc[3]*1000
    
    def read_header(self):
        return pd.read_csv(self.file, delim_whitespace=True,header=None,nrows =4)
    
    def read_data(self):
        if self.nx*self.ny*self.nz%9>0:
            rowc = self.nx*self.ny*self.nz//9 + 1
        if self.nx*self.ny*self.nz%9==0:
            rowc = self.nx*self.ny*self.nz//9
        
        refl = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=5,       nrows=rowc)
        u    = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=rowc+6,  nrows=rowc)
        v    = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=rowc*2+7,nrows=rowc)
        w    = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=rowc*3+8,nrows=rowc)

        refl = np.array(refl).flatten()[:-1].reshape(self.nz,self.ny,self.nx)
        u    = np.array(u).flatten()[:-1].reshape(self.nz,self.ny,self.nx)
        v    = np.array(v).flatten()[:-1].reshape(self.nz,self.ny,self.nx)
        w    = np.array(w).flatten()[:-1].reshape(self.nz,self.ny,self.nx)
        return refl,u,v,w
    
    def data_grid(self):
        initial_points = np.array(np.meshgrid(np.arange(self.iy,self.ny*self.dy+self.iy,self.dy),
                                              np.arange(self.iz,self.nz*self.dz+self.iz,self.dz),
                                              np.arange(self.ix,self.nx*self.dx+self.ix,self.dx)))

        y,z,x    = initial_points.reshape((3,np.size(initial_points)//3))
        return x,y,z
    
    def latlon(self,geo):
        x,y,z = self.data_grid()
        projl = MapProjection(projection='laea', lat_0=self.lat, lon_0=self.lon, alt_0=self.alt)
        points2  = np.array(projl.toECEF(x,y,z)).T
        xp,yp,zp = points2.T
        lonp,latp,altp = geo.fromECEF(xp,yp,zp)
        lonp = lonp.reshape(self.nz,self.ny,self.nx)
        latp = latp.reshape(self.nz,self.ny,self.nx)
        altp = altp.reshape(self.nz,self.ny,self.nx)
        return lonp,latp,altp
