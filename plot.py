import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def PlotPosterior(file, Npar, color, plot = True): 
  # Read file
  input = np.loadtxt(file, dtype='f', delimiter=' ')
  # Store in a matrix
  Par_ant = np.array(input[:,0])
  for i in range(1, Npar):    
    MatrixPar = np.column_stack((Par_ant,np.array(input[:,i])))
    Par_ant = MatrixPar
  # Plot
  sns.set()
  sns.set_style('white')
  MAP = np.zeros(MatrixPar.shape[1])  
  fig, ax = plt.subplots(1,MatrixPar.shape[1])
  for i in range(0, MatrixPar.shape[1]): 
    
    value = sns.distplot(MatrixPar[:,i],color=color,ax=ax[i]).get_lines()[0].get_data()
    MAP[i] = value[0][np.argmax(value[1])]
    ax[i].set_title("MAP = %2.4f" % (MAP[i]), fontsize=18)
    ax[i].set_xlabel("Parameter %d" % (i+1),fontsize=18)
    ax[i].set_ylabel('Density',fontsize=18)
  if (plot): plt.show()
  else: plt.close()
  return MAP
  
