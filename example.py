import numpy as np
import pickle #library to store the model trained
from ABC_parallel import ABC_SMC,ABC_MCMC

# Run mpiexec -n 5 py example.py

# MPI initializing
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def GenerateData(): # 2.0 x + 0.5, 0 < x < 1, 100 points
  x = np.linspace(0,1,10)
  Par = np.array([2.0,0.5])
  return Par[0]*x + Par[1]

def Reta(Par,Nqoi): # ax + b + noise, 0 < x < 1, 100 points
  x = np.linspace(0,1,Nqoi)
  return Par[0]*x + Par[1] + np.random.normal( 0.0,0.1) # Noise from normal dist.

if __name__ == '__main__':
  # Observational data
  Data = GenerateData()
  # Boundary of parameter space (prior: uniform distribution)
  UpperLimit = np.array([2.1,0.51])
  LowLimit = np.array([1.9,0.49])
  # Tolerance vector
  epsilon = np.array([1.0,0.4,0.1])

  #ABC_MCMC(Reta,Data,LowLimit,UpperLimit,Nrep=20,tol=epsilon[-1],NumAccept=5000)
  ABC_SMC(Reta,Data,LowLimit,UpperLimit,Nrep=20,tol=epsilon,NumAccept=1000)
