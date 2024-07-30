"""
Aim : To simulate distribution of mean first passage times in 3D Heterogeneous media with pores
Source : https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.240601
Author : Shree Ganesha Sharma.M.S
Date : 25-July-2023
"""
#necessary libraries
import time
import math
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.random.seed(1956)

#Brownian motion parameters
#Dimensions - 2, 3 or higher
dim = 3
#Time step
dt = 0.0001
#Number of steps
nums = 20000
#Number of particles
nump = 1000000
#Radius of the inner ball, reflecting boundary with pores
R1 = 0.5
#Radius of the outer ball, completely reflecting boundary
R2 = 1
#Diffusivity of inner medium
D1 = 1
#Diffusivity of outer medium
D2 = 1

#Radius of the target at the center
R = 0.3

#Hitting times if the target is hit
hittingTime = np.zeros(nump)

#Parameters of the normal distribution
mean = np.zeros(dim)
covar = np.identity(dim)

#Trajectory of a particle
X = np.zeros([dim, nums])

#nump particles starting at init
Init3D = np.tile(np.array([[.4], [0], [0]]), nump)

#Number of pores
N = 1000

#Max radius of pores at 500 pores = 0.0447
#Radius of pores
r = 0.0141

def pores():
  #A list of centers of pores
  Centers = []
  #First pore, selected randomly
  center = np.random.randn(dim)
  #Project the center of the pore onto the surface of the inner ball
  center = R1 * center / np.linalg.norm(center)
  Centers.append(center)

  #Add N-1 more pores uniformly randomly on the surface of the inner ball
  while len(Centers) < N:
    center = np.random.randn(dim)
    center = R1 * center / np.linalg.norm(center)
    #Ensure that there is no overlap between pores
    if np.all(np.linalg.norm(center - Centers, axis=1) > 2*r) == True:
      Centers.append(center)

  #Return the list of pore centers
  return Centers
     
#Trajectories
start_time = time.time()
for i in range(nump):
  #Initialize - The particle starts at a random position outside the target
  X[:, 0] = Init3D[:, i]

  #Randomize pore position for each trajectory
  poreCenters = pores()

  #Find particle trajectories
  for j in range(1, nums):
    #Random increment in the position
    dRand = np.random.multivariate_normal(mean, covar)
    #Increment if particle is in medium1
    dX1 = np.sqrt(2*D1*dt)*dRand
    #Increment if particle is in medium2
    dX2 = np.sqrt(2*D2*dt)*dRand

    #Particle hits the target
    if R < np.linalg.norm(X[:, j - 1]) < R1 and np.linalg.norm(X[:, j - 1] + dX1) < R:
      X[:, j] = X[:, j - 1] + dX1
      #For plotting
      hittingTime[i] = j
      """TODO - How to remove the particle once it hits the target?"""
      X[:, j+1:] = -1
      #print(j, "Hit", X[:, j], np.linalg.norm(X[:, j]), np.linalg.norm(X[:, j] - X[:, j-1]))
      break

    #Particle is traveling in medium1 outside the target
    elif R < np.linalg.norm(X[:, j - 1]) < R1 and R < np.linalg.norm(X[:, j - 1] + dX1) < R1:
      X[:, j] = X[:, j - 1] + dX1

    #Particle is traveling in medium2, without hitting the reflecting boundary
    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and R1 < np.linalg.norm(X[:, j - 1] + dX2) < R2:
      X[:, j] = X[:, j - 1] + dX2

    #Particle is reflected by the outer ball when its next step crosses the bound R2 - Reflect in D2
    #https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and np.linalg.norm(X[:, j - 1] + dX2) > R2:
      #Find the point of reflection
      lamb = ( -np.dot(X[:, j - 1], dX2) + np.sqrt(np.dot(X[:, j - 1], dX2)**2 - (np.linalg.norm(dX2)**2)*(np.linalg.norm(X[:, j - 1])**2 - R2**2)) )/(np.linalg.norm(dX2)**2)
      r0 = X[:, j - 1] + lamb*dX2
      #Image in D2
      X[:, j] = X[:, j - 1] + dX2 - 2*(1-lamb)*np.dot(dX2, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)

    elif R < np.linalg.norm(X[:, j - 1]) < R1 and np.linalg.norm(X[:, j - 1] + dX1) > R1:
      #Find the point of reflection/diffusion
      lamb = ( -np.dot(X[:, j - 1], dX1) + np.sqrt(np.dot(X[:, j - 1], dX1)**2 - (np.linalg.norm(dX1)**2)*(np.linalg.norm(X[:, j - 1])**2 - R1**2)) )/(np.linalg.norm(dX1)**2)
      r0 = X[:, j - 1] + lamb*dX1
      if any(np.linalg.norm(r0 - poreCenters, axis=1) < r):
        #If particle's projection on surface of the inner ball is near any pore, diffuse to D2.
        X[:, j] = X[:, j - 1] + dX1
      else:
        #Particle is reflected by inner ball when it's near the media boundary and far from all the pores
        #Image in D1
        X[:, j] = X[:, j - 1] + dX1 - 2*(1-lamb)*np.dot(dX1, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)
        #print(j, "Reflection in D1", X[:, j], np.linalg.norm(X[:, j]), np.linalg.norm(X[:, j] - X[:, j-1]))

    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and np.linalg.norm(X[:, j - 1] + dX2) < R1:
      #Find the point of reflection/diffusion
      lamb = ( -np.dot(X[:, j - 1], dX2) - np.sqrt(np.dot(X[:, j - 1], dX2)**2 - (np.linalg.norm(dX2)**2)*(np.linalg.norm(X[:, j - 1])**2 - R1**2)) )/(np.linalg.norm(dX2)**2)
      r0 = X[:, j - 1] + lamb*dX2
      #If particle projected on surface of ball 1 is near a pore, diffuse to D1. Else reflect it
      if any(np.linalg.norm(r0 - poreCenters, axis=1) < r):
        #Diffuse to D1 via a pore
        X[:, j] = X[:, j - 1] + dX2
      else:
        #Reflect - Image in D2
        X[:, j] = X[:, j - 1] + dX2 - 2*(1-lamb)*np.dot(dX2, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)
        #print(j, "Reflection in D2", X[:, j], np.linalg.norm(X[:, j]), np.linalg.norm(X[:, j] - X[:, j-1]))
  #print(i)

#Time taken to simulate motion of nump particles each taking nums steps
print("Ito nump = ", nump, "nums = ", nums, "dt = ", dt, "D1 = ", D1, " D2 = ", D2, "N = ", N, "r = ", r, "--- %s seconds ---" % (time.time() - start_time))

#MFPT
hit = hittingTime[hittingTime != 0]
print("MFPT(s) ", np.average(hit)*dt, "Particles hitting the target = ", len(hit))
