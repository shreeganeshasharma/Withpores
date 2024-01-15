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
R = 0.4
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
Init3D = np.tile(np.array([[.75], [0], [0]]), nump)

"""Initialize X[:, 0] here
Initialize counts for t=0"""

#Number of pores
N = 10

#Max radius of pores at 500 pores = 0.0447
#Radius of pores
r = 0.07

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
  #Initialize - The particle starts at a random position outside the target
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
  #Save the particle's trajectory
  Y[:, :, i] = X
  #print(i)

#Time taken to simulate motion of nump particles each taking nums steps
print("Ito nump = ", nump, "nums = ", nums, "dt = ", dt, "D1 = ", D1, " D2 = ", D2, "N = ", N, "r = ", r, "--- %s seconds ---" % (time.time() - start_time))

#MFPT
hit = hittingTime[hittingTime != 0]
print("MFPT(s) ", np.average(hit)*dt, "Particles hitting the target = ", len(hit))

"""# Plots"""

#Histogram hitting times and plot MFPT
hit = hittingTime[hittingTime != 0]
#Should zero be removed for MFPT calculation?
MFPT = np.average(hit)
print("Mean First Passage Time = ", MFPT)
plt.plot(np.tile(MFPT, int(max(plt.hist(hit, 30)[0]))), range(int(max(plt.hist(hit, 30)[0]))), label="MFPT")
#plt.hist(hit, 30, label="Zero removed")
#plt.hist(hittingTime, 30, label="Zero inflated")
plt.ylabel("Frequency")
plt.xlabel("hittingTime, nums")
plt.title("Hitting times and MFPT")
plt.legend()
plt.savefig(f'Images/Hist_Ito1_{N}_{R}_{r}_{nump}_{nums}_{D1}_{D2}.png')
#plt.show()

#Probability of finding a particle in medium1 or medium2 or hiting the target
count1 = np.zeros(nums)
count2 = np.zeros(nums)
count3 = np.zeros(nums)
for t in range(nums):
  for p in range(nump):
    count3[t] += np.count_nonzero((0 < np.linalg.norm(Y[:, t, p])) & (np.linalg.norm(Y[:, t, p]) < R))
    count1[t] += np.count_nonzero((R < np.linalg.norm(Y[:, t, p])) & (np.linalg.norm(Y[:, t, p]) < R1))
    count2[t] += np.count_nonzero((R1 < np.linalg.norm(Y[:, t, p])) & (np.linalg.norm(Y[:, t, p]) < R2))
xticks = np.linspace(1/nums, 1, nums)
yticks = np.linspace(1/nums, 1, nums)
plt.figure(figsize = (3, 6))
plt.yscale("log")
plt.plot(count1/nump, yticks)
plt.plot(count2/nump, yticks)
plt.plot((count1 + count2)/nump, yticks)
plt.plot(xticks, np.tile(MFPT/nums, nums))
#plt.plot(count3/nump)
plt.legend(["In D1", "In D2", "In D1 or D2", "MFPT", "Hit target"])
plt.ylabel("Time, t")
plt.xlabel("Pr(t)")
plt.title("Likelihood in a region")

plt.savefig(f'Images/Prob_Ito1_{N}_{R}_{r}_{nump}_{nums}_{D1}_{D2}.png')
#plt.show()

#At dt = 0.001 and nums = 1000, nump = 5000,
#Half the number of particles are still in D1 at the end of simulation.
#Increase dt to 0.01
#Gaphs look similar

#Plot heatmap of particles over time and space

gridX = 100
gridT = nums
pos = np.round(np.linspace(0, 1, gridX + 1), 2)
tim = np.linspace(0, gridT, gridT+1)
PDF1D = np.zeros((gridT, gridX+1))

#Derive a heatmap of the particles
#Find the norm of particles at each time step
Z = np.linalg.norm(Y/R2, axis=0)

#Quantize it to 2 ( = log10(gridX)) decimal places as grix X is with such precision
Z = np.round(Z, 2)

for t in range(1, nums):
#position of all particles at time t
  for p in range(len(pos)):
    PDF1D[t][p] = np.count_nonzero(Z[t, :] == pos[p])

#Plot the heatmap of pdf
plt.figure(figsize = (8, 6))
plt.title("Joint PDF of particles, P(x, t)")
#PDF1D is a 2D matrix with (0,0)th element on top left. Fip vertically
plt.yscale('log')
plt.imshow(np.flipud(PDF1D/nump), cmap = 'hsv', extent = [0, 1, dt, dt*nums], aspect = "auto")
plt.xlabel("Distance, norm(x)")
plt.ylabel("Time, t")
plt.colorbar()
plt.savefig(f'Images/Heat_Ito1_{N}_{R}_{r}_{nump}_{nums}_{D1}_{D2}.png')
