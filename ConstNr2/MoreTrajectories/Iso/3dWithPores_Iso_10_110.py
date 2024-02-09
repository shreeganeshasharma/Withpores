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
np.random.seed(1746)

#Brownian motion parameters
#Dimensions - 2, 3 or higher
dim = 3
#Time step
dt = 0.0001
#Number of steps
nums = 20000
#Number of particles
nump = 10000
#Radius of the inner ball, reflecting boundary with pores
R1 = 0.5
#Radius of the outer ball, completely reflecting boundary
R2 = 1
#Diffusivity of inner medium
D1 = 1
#Diffusivity of outer medium
D2 = 10

#Radius of the target at the center
R = 0.3
"""Get r0 value from paper"""
#Hitting times if the target is hit
hittingTime = np.zeros(nump)

#Parameters of the normal distribution
mean = np.zeros(dim)
covar = np.identity(dim)

#Trajectory of a particle
X = np.zeros([dim, nums])
#Trajectories of all particles
Y = np.zeros([dim, nums, nump])

#Random steps
Eps = np.random.multivariate_normal(mean, covar, (nump, nums))
dX1s = np.sqrt(2*D1*dt)*Eps
dX2s = np.sqrt(2*D2*dt)*Eps

#nump particles starting at init
Init3D = np.tile(np.array([[.4], [0], [0]]), nump)

"""Initialize X[:, 0] here
Initialize counts for t=0"""

#Number of pores
N = 20

#Can neglect the curvature in higher dimensions as long as R1 >> R  - > Arc length = 2r*arcsin(d/2r) ~ d
#Maximum number of pores possible at radius R - floor(4*pi*R1^2 / pi*R^2)
#Maximum diameter at N pores = floor(4*pi*R1^2/pi*R^2)
#Radius of pores
r = 0.1

#A list of centers of pores
poreCenters = []
#First pore, selected randomly
center = np.random.randn(dim)
#Project the center of the pore onto the surface of the inner ball
center = R1 * center / np.linalg.norm(center)
poreCenters.append(center)

#Add N-1 more pores uniformly randomly on the surface of the inner ball
while len(poreCenters) < N:
  center = np.random.randn(dim)
  center = R1 * center / np.linalg.norm(center)
  #Ensure that there is no overlap between pores
  if all(np.linalg.norm(center - poreCenters, axis=1) > 2*r) == True:
    poreCenters.append(center)
"""
Refer to BetterPorePassage if necessary - https://colab.research.google.com/drive/1IaztD_VBbBy5TW3F0XF4Thw1xvNx-mkf#scrollTo=pMhdohz7N7Rd
"""

start_time = time.time()
for i in range(nump):
  #Initialize - The particle starts at a random positio outside the target
  X[:, 0] = Init3D[:, i]

  #Find particle trajectories
  for j in range(1, nums):
    #Random increment in the position
    dRand = Eps[i, j, :]
    #Increment if particle is in medium1
    dX1 = dX1s[i, j, :]
    #Increment if particle is in medium2
    dX2 = dX2s[i, j, :]

    #Particle hits the target
    if R < np.linalg.norm(X[:, j - 1]) < R1 and np.linalg.norm(X[:, j - 1] + dX1) < R:
      X[:, j] = X[:, j - 1] + dX1
      X[:, j+1:] = -1
      hittingTime[i] = j
      break

    #Particle travels in medium1, without hitting the target
    elif R < np.linalg.norm(X[:, j - 1]) < R1 and R < np.linalg.norm(X[:, j - 1] + dX1) < R1:
      X[:, j] = X[:, j - 1] + dX1

    #Particle is traveling in medium2
    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and R1 < np.linalg.norm(X[:, j - 1] + dX2) < R2:
      X[:, j] = X[:, j - 1] + dX2

    #Particle is reflected by the outer ball when its next step crosses the bound R2
    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and np.linalg.norm(X[:, j - 1] + dX2) > R2:
      #Find the point of reflection
      lamb = ( -np.dot(X[:, j - 1], dX2) + np.sqrt(np.dot(X[:, j - 1], dX2)**2 - (np.linalg.norm(dX2)**2)*(np.linalg.norm(X[:, j - 1])**2 - R2**2)) )/(np.linalg.norm(dX2)**2)
      r0 = X[:, j - 1] + lamb*dX2
      #Image in D2
      X[:, j] = X[:, j - 1] + dX2 - 2*(1 - lamb)*np.dot(dX2, r0/np.linalg.norm(r0)) * r0/np.linalg.norm(r0)

    #Particle diffuses from D1 to D2 with a one sided reflection coefficient
    elif R < np.linalg.norm(X[:, j - 1]) < R1 and np.linalg.norm(X[:, j - 1] + dX1) > R1:
      #Find the point of diffusion/reflection
      lamb = ( -np.dot(X[:, j - 1], dX1) + np.sqrt(np.dot(X[:, j - 1], dX1)**2 - (np.linalg.norm(dX1)**2)*(np.linalg.norm(X[:, j - 1])**2 - R1**2)) )/(np.linalg.norm(dX1)**2)
      r0 = X[:, j - 1] + lamb*dX1
      if any(np.linalg.norm(r0 - poreCenters, axis=1) < r):
        #If particle's projection on surface of the inner ball is near any pore, diffuse to D2.
        if bernoulli.rvs(min(1, math.sqrt(D2/D1))):
          #Diffuse with time-splitting, if the random variable generated is 1
          #Residence times in each medium
          dt1 = (np.linalg.norm(lamb*dX1)/(np.linalg.norm(dRand)*np.sqrt(2*D1)))**2
          dt2 = ((1 - np.sqrt(dt1/dt))**2)*dt
          #Particle diffuses, increment by motion in D2
          X[:, j] = r0 + np.sqrt(2*D2*dt2)*dRand
        else:
          #Reflect, if the random variable generated is 0
          X[:, j] = X[:, j - 1] + dX1 - 2*(1-lamb)*np.dot(dX1, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)
      else:
          #Reflect particle is far from all pores
          X[:, j] = X[:, j - 1] + dX1 - 2*(1-lamb)*np.dot(dX1, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)

    #Particle diffuses from D2 to D1 with a one sided reflection coefficient
    elif R1 < np.linalg.norm(X[:, j - 1]) < R2 and np.linalg.norm(X[:, j - 1] + dX2) < R1:
      #Find the point of diffusion/reflection
      lamb = ( -np.dot(X[:, j - 1], dX2) - np.sqrt(np.dot(X[:, j - 1], dX2)**2 - (np.linalg.norm(dX2)**2)*(np.linalg.norm(X[:, j - 1])**2 - R1**2)) )/(np.linalg.norm(dX2)**2)
      r0 = X[:, j - 1] + lamb*dX2
      if any(np.linalg.norm(r0 - poreCenters, axis=1) < r):
        #If particle's projection on surface of the inner ball is near any pore, diffuse to D2.
        if bernoulli.rvs(min(1, math.sqrt(D1/D2))):
          #Diffuse with time-splitting, if the random variable generated is 1
          #Residence times in each medium
          dt2 = (np.linalg.norm(lamb*dX2)/(np.linalg.norm(dRand)*np.sqrt(2*D2)))**2
          dt1 = ((1 - np.sqrt(dt2/dt))**2)*dt
          #Particle diffuses, increment by motion in D1
          X[:, j] = r0 + np.sqrt(2*D1*dt1)*dRand
        else:
          #Reflect, if the random variable generated is 0
          X[:, j] = X[:, j - 1] + dX2 - 2*(1-lamb)*np.dot(dX2, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)
      else:
          #Reflect, if projection is far from all pores
          X[:, j] = X[:, j - 1] + dX2 - 2*(1-lamb)*np.dot(dX2, r0/np.linalg.norm(r0))*r0/np.linalg.norm(r0)

  #Save the particle's trajectory
  Y[:, :, i] = X
  #print(i)

#Time taken to simulate motion of nump particles each taking nums steps
print("Iso nump = ", nump, "nums = ", nums, "dt = ", dt, "D1 = ", D1, " D2 = ", D2, "N = ", N, "r = ", r, "--- %s seconds ---" % (time.time() - start_time))

#MFPT
hit = hittingTime[hittingTime != 0]
print("MFPT(s) ", np.average(hit)*dt, "Particles hitting the target = ", len(hit))


