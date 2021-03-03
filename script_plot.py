from plot import PlotPosterior
import matplotlib.pyplot as plt
import numpy as np
from example import GenerateData,Reta
import sys

if (len(sys.argv) != 2):
  print("Please provide 1 args: File")
  sys.exit(1)
file = sys.argv[1];
MAP = PlotPosterior(file,2,color="gray")
print(MAP)

Nqoi = 10
Nrep = 20
OutputM = np.zeros((Nrep, Nqoi));
for i in range( 0, Nrep ):
    OutputM[i] = Reta(MAP,Nqoi)

meanQOI = np.mean(OutputM, axis=0)
stdQOI = np.std(OutputM, axis=0)
plt.plot(np.linspace(0,1,Nqoi),meanQOI, color='black')
plt.fill_between(np.linspace(0,1,Nqoi), meanQOI-stdQOI, meanQOI+stdQOI, fc='gray')
plt.scatter(np.linspace(0,1,Nqoi),GenerateData(), marker='o',color = 'red', alpha = 1.0)
plt.show()
