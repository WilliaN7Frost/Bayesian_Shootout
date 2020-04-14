# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 02:43:08 2020

@author: wilia
"""
import numpy as np
import BayesianShootoutFuncs_2 as b
import emcee
import corner

buildingDims , bSeper , red_loc , blue_loc , red_hitbox , blue_hitbox = b.setup()

# shotWasHit collects boolean values on whether a certain shot was a hit or not
# shotsForCheckHit collects the shot coordinates used to check if a shot was a hit against Blue
# shotsForVisual collects shot coordinates to be plotted onto the map
# shotsSeenByDrone collects the shot coordinates used to run the MCMC
shotsAllowed = 40
shotsTaken = 0
shotWasHit = []
shotsForCheckHit = []
shotsForVisual = []
shotsSeenByDrones = []    
yloc = -50      # Used to plot shots that cross the enitre map

    # The allowed firing angle ranges for Red
theta_range = [  np.arctan(-red_loc[0]/(red_loc[1]-buildingDims[1])) , np.arctan((buildingDims[0]-red_loc[0])/(red_loc[1]-buildingDims[1]))  ]
phi_range = [  np.arctan(-red_loc[2]/(red_loc[1]-buildingDims[1])) , np.arctan((buildingDims[2]-red_loc[2])/(red_loc[1]-buildingDims[1]))  ]
print("theta_range = " +str(theta_range)+ " and phi_range = " +str(phi_range))

    # Loop to fire off specified number of random shots. Detect them also
for i in range(shotsAllowed):
    theta = np.random.uniform( theta_range[0] , theta_range[1] )
    phi = np.random.uniform( phi_range[0] , phi_range[1] )
    shotsForCheckHit.append( b.shotAtLocationY(theta , phi , red_loc , yloc=blue_loc[1]) )
    shotsSeenByDrones.append( b.shotAtLocationY(theta , phi , red_loc , yloc=buildingDims[1]) )
    shotsForVisual.append( b.shotAtLocationY(theta , phi , red_loc , yloc=yloc) )
        # checking to see if a shot was a hit on Blue
    if (b.wasShotAHit(blue_loc , shotsForCheckHit[-1] , blue_hitbox)):
        print("Shot hit at " + str(shotsForCheckHit[-1]))
        shotsTaken += 1
        shotWasHit.append(True)
    else:
        shotWasHit.append(False)



confidence_tolerance = 0.68
num_iter = 5000;  paramDims = 3;  nwalkers = 10

alpha_min = 0.;  alpha_max = buildingDims[0]
beta_min = 0;  beta_max = buildingDims[1] + bSeper
gamma_min = 0.;  gamma_max = buildingDims[2]

param_bounds = np.array(((alpha_min, alpha_max), (beta_min, beta_max), (gamma_min,gamma_max)))
bestShot = np.array( ( (alpha_max-alpha_min)/2 , (beta_max-beta_min)/2 , (gamma_max-gamma_min)/2 )  )


initial_pos = bestShot + 0.01 * np.random.randn(nwalkers, paramDims)
    # The X-Z data of our 2D detection plane to input into the MCMC
shots = np.transpose(shotsSeenByDrones)
shotsXZ = np.array([shots[0],shots[2]])

sampler = emcee.EnsembleSampler(nwalkers, paramDims, b.log_posterior, args=(shotsXZ[0], shotsXZ[1], param_bounds))
sampler.run_mcmc(initial_pos, num_iter, progress=True);

    # Adjusting the beta values to fit where they should really be on the map.
param_collection = sampler.get_chain(flat=True , discard=100)
for i in range(len(param_collection)):
    param_collection[i][1] += buildingDims[1]

    # the median values and errors to plot    
medians = np.quantile(np.transpose(param_collection) , 0.5 , axis=1) 
lowerQuantiles = medians - np.quantile(np.transpose(param_collection) , 0.5-(confidence_tolerance/2) , axis=1)
upperQuantiles = np.quantile(np.transpose(param_collection) , 0.5+(confidence_tolerance/2) , axis=1) - medians



b.plotShots2D( blue_loc , red_loc , buildingDims , bSeper , shotsForVisual , shotWasHit , 
               blue_hitbox , red_hitbox , ylevel=yloc , multipleMC_outs=np.array([[medians,lowerQuantiles,upperQuantiles]]) , 
               plotHitBoxes=True)

corner.corner(param_collection, labels=[r"$\alpha\,$[cm]", r"$\beta\,$[cm]" , r"$\gamma\,$[cm]"] ,
              quantiles = [0.16,0.5,0.84] , show_titles=True )
