import pandas as pd
import gzip

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


    def datastart_gz(self):
        with gzip.open(fname,'rt') as f:
            for line_no, line in enumerate(f):
                    if line.rstrip() == "*** data ***":
                        break
            f.close()
            return line_no

    def readfile(self):
        if self.file[-4:] == '.dat':
            lmad = pd.read_csv(self.file,delim_whitespace=True,header=None,skiprows=self.datastart()+1)
        if self.file[-7:] == '.dat.gz':
            lmad = pd.read_csv(self.file,compression='gzip',delim_whitespace=True,
                                header=None,skiprows=self.datastart_gz()+1)
        columns = ['time','lat','lon','alt','chi','num','p','charge','mask']
        lmad.columns = columns
        return lmad