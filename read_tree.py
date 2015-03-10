#! /usr/bin/env python
""" read data from tree """

import root_numpy

from root_numpy import tree2rec, root2rec, root2array
from root_numpy.testdata import get_filepath

if __name__ == '__main__' : 

    """testing functions"""

    #easy
    filename = get_filepath('test.root')
    print filename
    arr = root2array(filename, 'tree' )
    print type(arr)
    rec = root2rec(filename, 'tree' ) 
    print type(rec)

    #K*mumu
    kstmmfile = "../data/kstmumu_sim_sel.root"
    arr = root2array(kstmmfile,'DecayTree')
    
    kstmmfile = "../data/kstmumu_data.root"
    arr = root2array(kstmmfile,'DecayTree')

