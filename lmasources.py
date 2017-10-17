import numpy as np
import pandas as pd

from coordinateSystems import GeographicSystem, MapProjection

class lmafile:
    def __init__(self,filename):
        self.file = filename
        # self.

    def datastart(self):
        with open(self.file) as f:         
            for line_no, line in enumerate(f):
                if line.rstrip() == "*** data ***":
                    break
        f.close()
        return line_no

    def readfile(self):
        lmad = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=self.datastart()+1)
        columns = ['time','lat','lon','alt','chi','num','p','charge','mask']
        lmad.columns = columns
        return lmad