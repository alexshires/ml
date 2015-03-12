#! /usr/bin/env python
""" read data from tree """

import root_numpy

from root_numpy import tree2rec, root2rec, root2array

from matplotlib import rc, rc_file
rc_file('/home/alexshires/.config/matplotlib/matplotlibrc')
rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#specfiy branches


bvars = ["B_P", "B_PT", "B_IPCHI2_OWNPV", "B_DIRA_OWNPV",
         "B_IP_OWNPV", "B_FDCHI2_OWNPV", "B_ENDVERTEX_CHI2",
         "B_ISOLATION_BDT_Hard", "B_ISOLATION_BDT2_Hard",
         "B_ISOLATION_BDT3_Hard", "B_ISOLATION_BDT_Soft",
         "B_ISOLATION_BDT2_Soft", "B_ISOLATION_BDT3_Soft"
        ]
psivars = ["Psi_P", "Psi_PT", "Psi_IPCHI2_OWNPV", "Psi_DIRA_OWNPV"
           , "Psi_IP_OWNPV", "Psi_FDCHI2_OWNPV", "Psi_ENDVERTEX_CHI2"
          ]
kstvars = ["Kstar_P", "Kstar_PT", "Kstar_IPCHI2_OWNPV", "Kstar_DIRA_OWNPV"
           , "Kstar_IP_OWNPV", "Kstar_FDCHI2_OWNPV", "Kstar_ENDVERTEX_CHI2"
          ]

mupvars = ["muplus_P", "muplus_PT", "muplus_IPCHI2_OWNPV", "muplus_IP_OWNPV"
           , "muplus_PIDmu", "muplus_PIDK", "muplus_PIDp"
           , "muplus_ProbNNmu", "muplus_ProbNNk", "muplus_ProbNNp"
           , "muplus_ProbNNpi", "muplus_ProbNNghost"
           , "muplus_TRACK_CHI2NDOF", "muplus_TRACK_GhostProb"
          ]

mumvars = [var.replace("muplus", "muminus") for var in mupvars]
kaonvars = [var.replace("muplus", "Kplus") for var in mupvars]
pionvars = [var.replace("muplus", "piminus") for var in mupvars]


sigvars = ["B_M", "ctl", "ctk", "qsq"]



mvavars = bvars + psivars + kstvars + mupvars + mumvars + kaonvars + pionvars
totvars = mvavars + sigvars

import pandas 
from pandas import DataFrame

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

def get_hist(data, nbins=40, norm=False):
    yvals, binedges = np.histogram(data, bins=nbins)
    xvals = (binedges[1:]+binedges[:-1])/2
    xerrs = (binedges[1]-binedges[0])/2
    yerrs = np.sqrt(yvals)
    if norm:
        nevents = float(sum(yvals))
        binwidth = (binedges[1]-binedges[0])
        yvals = yvals/nevents/binwidth
        yerrs = yerrs/nevents/binwidth
    return (xvals, xerrs, yvals, yerrs)


def makeplots(dataset, plotname, dataset2=DataFrame(), norm=False):
    xvalues2 = None
    with PdfPages(plotname+".pdf") as pdf:
        for col in dataset.columns:
            print "plotting", col
            #convert data to points
            arrs1 = get_hist(dataset[col], 40, norm)
            xvalues, xerrors, yvalues, yerrors = arrs1
            if not dataset2.empty:
                arrs2 = get_hist(dataset2[col], 40, norm)
                xvalues2, xerrors2, yvalues2, yerrors2 = arrs2
            if "DIRA" in col:
                xvalues = np.arccos(xvalues)
                if not dataset2.empty:
                    xvalues2 = np.arccos(xvalues2)
            #axes labels
            maxval = np.amax(yvalues)
            #print maxval
            #print yvalues
            #print xvalues
            plt.errorbar(xvalues, yvalues, xerr=xerrors,
                         yerr=yerrors,
                         fmt=".", color='black', label='data')
            if not dataset2.empty:
                plt.errorbar(xvalues2, yvalues2,
                             xerr=xerrors2, yerr=yerrors2,
                             fmt=".", color='blue', label='sim')
            plt.ylabel('events', ha='right', y=1)
            label = col.replace("_", ": ")
            plt.xlabel(label, ha='right', x=1)
            plt.ylim([0, 1.1*maxval])
            plt.minorticks_on()
            plt.draw()
            ax = plt.gca()
            plt.legend()
            ##sorts out offset
            offset = ax.get_xaxis().get_offset_text()
            #print offset, offset.get_text()
            if offset.get_text():
                plt.xlabel("%s (%s)" % (label, offset.get_text()),
                           ha='right', x=1)
            offset.set_visible(False)
            pdf.savefig()
            plt.clf()






if __name__ == '__main__':
    #K*mumu
    #test converting to hdf5i --rootpy
    print "reading data"
    kstmmsimfile = "/home/alexshires/data/BsKstmm/Sim/kstmumu_sim_sel.root"
    #kstmmsimfile2 = "/home/alexshires/data/BsKstmm/Sim/kstmumu_sim_sel.hdf5"
    #root2hdf5(kstmmsimfile, kstmmsimfile2)
    kstmmdatafile = "/home/alexshires/data/BsKstmm/Data/"\
            +"kstmumu_data_kstmumu.root"
    dataarr = pandas.DataFrame(root2array(kstmmdatafile, "DecayTree",
                                          branches=totvars))
    simarr = pandas.DataFrame(root2array(kstmmsimfile, "DecayTree",
                                         branches=totvars))

    print "Data"
    print dataarr["B_M"]
    print simarr["B_M"]

    #makeplots(simarr, "dataplots", bkgarr, True)


    base_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
    base_ada = AdaBoostClassifier(base_estimator=base_tree,
                                  n_estimators=400,
                                  learning_rate=0.2)

    print "bakcground array" 
    bkgarr = dataarr.query("B_M>5450 & qsq<20").drop(sigvars, axis=1)
    sigarr = simarr.query("qsq<20").drop(sigvars, axis=1)

    bkgtrain = bkgarr[:5000]
    sigtrain = sigarr[:5000]
    sigtest = sigarr[5000:10000]
    bkgtest = bkgarr[5000:10000]


    #ifrom sklearn.datasets import make_gaussian_quantiles
    #make_gaussian_quantiles(nsamples=100,
    #                        n_features=2,
    #                        n_classes=3,

    print "creating mask"
    os = np.ones(len(bkgtrain))
    zs = np.zeros(len(sigtrain))
    print "adding samples together"
    tot = pandas.concat([sigtrain, bkgtrain])
    mask = np.append(os, zs)
    print "training"
    base_ada.fit(tot, mask)


    sigoutput = base_ada.decision_function(X=sigtest)
    bkgoutput = base_ada.decision_function(X=bkgtest)
    with PdfPages("bdtplots.pdf") as pdf:
        xs, xe, ys, ye = get_hist(bkgoutput)
        plt.errorbar(xs, ys, xerr=xe, yerr=ye,
                     color='red', fmt='.',
                     label='bkg')
        xs, xe, ys, ye = get_hist(sigoutput)
        plt.errorbar(xs, ys, xerr=xe, yerr=ye,
                     color='blue', fmt='.',
                     label='sig')
        plt.legend()
        plt.xlabel("BDT decision")
        pdf.savefig()



    #print "config MVAs"
    #mvadict = configMVAs(mvavars, sigvars, 3, 400, 5)
    #process tree

    #mvadict["adabdt"].fit(bkgarr, simarr)



