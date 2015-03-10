#! /usr/bin/env python

from sklearn import datasets
print datasets , type(datasets)

iris = datasets.load_iris()
digits = datasets.load_digits()
#print iris
print digits.target

#plot them
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

images_and_labels = zip(digits.images, digits.target)

with PdfPages("images.pdf") as pdf :
    for (image, label) in images_and_labels[:4]:
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("training %g"% label)
        pdf.savefig()
        plt.close()
         
nsamples = len(digits.images)
print digits.images[0]
data = digits.images.reshape((nsamples,-1))
print data[0]


#decision tree

