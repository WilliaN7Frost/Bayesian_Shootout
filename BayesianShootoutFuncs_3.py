# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:50:56 2020

@author: wilia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import corner
import emcee



def setup( buildingDims=[] , bSeper=0. , blue_loc=[] , red_loc=[] , red_hitbox=[] , blue_hitbox=[] , doPrint=True):
    """
    buildingDims : 1D array
            The 3D dimensions [x,y,z] (or [alpha,beta,gamma] if you prefer) in cm of both buildings on the map
    bSeper : float
            Seperation between buildings in cm
    blueLoc , redLoc : 1D array
            Location in 3D space ([x,y,z]) of each person within their respective buildings. Should be within building limits.
    blue_hitbox , red_hitbox : 1D array
            3D volume of each person. Input with [x,y,z] lengths
    """
    if (len(buildingDims) == 0):
        buildingDims = [1500. , 500. , 1000.]
    if (len(blue_hitbox) == 0):
        blue_hitbox = [100. , 50. , 200.]
    if (len(red_hitbox) == 0):
        red_hitbox = [100. , 50. , 200.]
    
    if (len(red_loc)!=0):
        if ( red_loc[0] > buildingDims[0]-red_hitbox[0]/2 ): red_loc[0] = buildingDims[0]-red_hitbox[0]/2
        if ( red_loc[0] < red_hitbox[0]/2 ): red_loc[0] = red_hitbox[0]/2
        if ( red_loc[1] > buildingDims[1]-red_hitbox[1]/2 ): red_loc[1] = buildingDims[1]-red_hitbox[1]/2
        if ( red_loc[1] < red_hitbox[1]/2 ): red_loc[1] = red_hitbox[1]/2
        if ( red_loc[2] > buildingDims[2]-red_hitbox[2]/2 ): red_loc[2] = buildingDims[2]-red_hitbox[2]/2
        if ( red_loc[2] < red_hitbox[2]/2 ): red_loc[2] = red_hitbox[2]/2
    else:
        red_loc = np.zeros(3)
        for i in range(3):
            red_loc[i] = np.random.randint(low = red_hitbox[i] , high = buildingDims[i]-red_hitbox[i]+1)
    red_loc[1] += buildingDims[1] + bSeper
    
    if (len(blue_loc)!=0):
        if ( blue_loc[0] > buildingDims[0]-blue_hitbox[0]/2 ): blue_loc[0] = buildingDims[0]-blue_hitbox[0]/2
        if ( blue_loc[0] < blue_hitbox[0]/2 ): blue_loc[0] = blue_hitbox[0]/2
        if ( blue_loc[1] > buildingDims[1]-blue_hitbox[1]/2 ): blue_loc[1] = buildingDims[1]-blue_hitbox[1]/2
        if ( blue_loc[1] < blue_hitbox[1]/2 ): blue_loc[1] = blue_hitbox[1]/2
        if ( blue_loc[2] > buildingDims[2]-blue_hitbox[2]/2 ): blue_loc[2] = buildingDims[2]-blue_hitbox[2]/2
        if ( blue_loc[2] < blue_hitbox[2]/2 ): blue_loc[2] = blue_hitbox[2]/2
    else:
        blue_loc = np.zeros(3)
        for i in range(3):
            blue_loc[i] = np.random.randint(low = blue_hitbox[i] , high = buildingDims[i]-blue_hitbox[i]+1)
    
    if doPrint:
        print("Building dimensions are " + str(buildingDims) + " (cm)")
        print("Buildings Seperated by " + str(bSeper) + " cm")
        print("Blue Located at " + str(blue_loc) + " and hitbox within " 
              + str( [[blue_loc[0]-blue_hitbox[0]/2,blue_loc[0]+blue_hitbox[0]/2]
                      ,[blue_loc[1]-blue_hitbox[1]/2,blue_loc[1]+blue_hitbox[1]/2]
                       ,[blue_loc[2]-blue_hitbox[2]/2,blue_loc[2]+blue_hitbox[2]/2]] ))
        print("Red Located at " + str(red_loc) + " and hitbox within " 
          + str( [[red_loc[0]-red_hitbox[0]/2,red_loc[0]+red_hitbox[0]/2]
                  ,[red_loc[1]-red_hitbox[1]/2,red_loc[1]+red_hitbox[1]/2]
                   ,[red_loc[2]-red_hitbox[2]/2,red_loc[2]+red_hitbox[2]/2]] ))
    
    return buildingDims , bSeper , red_loc , blue_loc , red_hitbox , blue_hitbox



"""
This function returns a 3D coordinate corresponding to a destination that a given shot trajectory reaches. 
"""
def shotAtLocationY( theta , phi , shotOrigin , yloc=0 ):
    """
    theta , phi : float
            Angles. Theta lies in the X-Y plane while Phi lies in the Y-Z plane
    shotOrigin : 1D array
            3D coordinate from where the shot came from
    yloc : int
            The Y-location (or Beta value) we want our shot to reach
    """
    shotX = shotOrigin[0]+((shotOrigin[1]-yloc)*np.tan(theta))
    shotZ = shotOrigin[2]+((shotOrigin[1]-yloc)*np.tan(phi))
    return np.array([shotX,yloc,shotZ])



"""
Returns True/False whether a given shot destination was able to hit within the hitbox of a person's location.
If the shot coordinate does not fall within this hitbox, the trajectory of the shot is analyzed to determine
whether its path crossed the hitbox of interest
"""
def wasShotAHit(loc , shot , hitbox , shotFrom=[]):
    """
    loc , shot , hitbox : 1D array
            'loc' is the taret location, 'hitbox' is the dimensions of the target, 'shot' is the shot coordinate
    shotFrom : 1D array
            if supplied, allows the function to check whether the shot trajectory passed through the loc's hitbox 
    """
    x_hitbox = [ loc[0]-hitbox[0]/2 , loc[0]+hitbox[0]/2 ]
    y_hitbox = [ loc[1]-hitbox[1]/2 , loc[1]+hitbox[1]/2 ]
    z_hitbox = [ loc[2]-hitbox[2]/2 , loc[2]+hitbox[2]/2 ]
    
    if ( shot[0] >= x_hitbox[0] and shot[0] <= x_hitbox[1] 
           and shot[1] >= y_hitbox[0] and shot[1] <= y_hitbox[1]
               and shot[2] >= z_hitbox[0] and shot[2] <= z_hitbox[1] ):
        return True
    elif(len(shotFrom) != 0):
        theta = np.arctan( (shot[0]-shotFrom[0]) / (shot[1]-shotFrom[1]) )
        phi = np.arctan( (shot[2]-shotFrom[2]) / (shot[1]-shotFrom[1]) )
        projectedX = shot[0] + (np.array(y_hitbox)-shot[1]) * np.tan(theta)
        projectedZ = shot[2] + (np.array(y_hitbox)-shot[1]) * np.tan(phi)
        return ( ( projectedX[0] >= x_hitbox[0] and projectedX[0] <= x_hitbox[1]
                     and projectedZ[0] >= z_hitbox[0] and projectedZ[0] <= z_hitbox[1] )
                or
                 ( projectedX[1] >= x_hitbox[0] and projectedX[1] <= x_hitbox[1]
                     and projectedZ[1] >= z_hitbox[0] and projectedZ[1] <= z_hitbox[1] ) )
    else:
        return False






"""
The mother of all plot functions. Allows you to visualize the map, shots taken and results from MCMC
"""
def plotShots2D( blue_loc , red_loc , buildingDims , bSeper, shots , shotWasHit , blue_hitbox , 
                 red_hitbox , ylevel=0 , blueShot=[] , emceeOutput=[] , multipleMC_outs=[] , 
                 plotHitBoxes=True , plotDotsForManyMC=False , plotAllShots=True , plotPlayers=True ,
                 plotMultMCshots=False):
    """
    blue_loc , red_loc , buildingDims , bSeper , blue_hitbox , red_hitbox : 
            Refer to the setup() function for explanations on these parameters
            
    shots , shotWasHit : 2D and 1D arrays respectively
            'shots' holds in each element a 3D coordinate cooresponding to a shot destination. 'shotWasHit' should contain 
             an ordering of boolean values for which the nth element specifies if the nth shot was a hit against Blue or not.
   
    ylevel , blueShot : int and 1D array
            'ylevel' refers to the depth in y (beta) that we want our shots to be shown reaching on the map.
            If 'blueShot' is supplied, it corresponds to the 3D coordinate that Blue decided to aim at.
            
    emceeOutput , multipleMC_outs , plotDotsForManyMC , plotMultMCshots : 1D , 3D arrays and boolean values
    
            - If supplied, 'emceeOutput' refers to the inferred location determined by the MCMC. Plotted as scatter.
            - If supplied, 'multipleMC_outs' is a 3D array that stores in each element the median, 16th and 84th quantiles
              of a 3D coordinate. This array can be used to plot many inferred locations from our MCMC process.
            - Given 'multipleMC_outs', 'plotDotsForManyMC' is a boolean value controlling whether we plot simple dots and
              error bars to represent the 'multipleMC_outs' or we use rectangles to show the error range.
            - Given 'multipleMC_outs', 'plotMultMCshots' allows a very niche functionality to be acheived. It plots straight
              lines connecting the actual and inferred shooter locations.
    
    plotHitBoxes , plotPlayers , plotAllShots : boolean
            'plotHitBoxes' tells us to print players as rectangles or dots
            'plotPlayers'  tells us whether to plot players onto the map or not
            'plotAllShots' tells us to print the shots from the red shooter or not
    """
    
    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=30)     # fontsize of the axes title
    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    
    plt.figure(figsize=(15,7))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(shots))]
    if (bSeper==0): sepCoeff = 1.02;  # if bSeper=0, we still add an artificial seperation to better view the buildings
    else: sepCoeff = 1.
    
    # Plotting the top view of the map
    plt.subplot(1,2,1)
    plt.title("Top Drone View ("+r"$\theta\,$"+" axis)"); plt.xlabel(r"$\alpha\,$"+' axis (cm)'); plt.ylabel(r"$\beta\,$"+' axis (cm)')
    blueBuildingXY = plt.Rectangle((0,0) , buildingDims[0] , buildingDims[1] , fc='silver',ec="black",zorder=1.)
    redBuildingXY = plt.Rectangle((0,buildingDims[1]*sepCoeff+bSeper) , buildingDims[0] , buildingDims[1] , fc='silver',ec="black",zorder=1.)
    plt.gca().add_patch(blueBuildingXY)
    plt.gca().add_patch(redBuildingXY)
    plt.gca().set_facecolor('dimgray')
    
    if plotPlayers:
        if (plotHitBoxes):
            blueBoxXY = plt.Rectangle( (blue_loc[0]-blue_hitbox[0]/2 , blue_loc[1]-blue_hitbox[1]/2)
                                , blue_hitbox[0] , blue_hitbox[1] , fc='blue' , ec="black" , zorder=10.)
            redBoxXY = plt.Rectangle( (red_loc[0]-red_hitbox[0]/2 , red_loc[1]-red_hitbox[1]/2+buildingDims[1]*(sepCoeff-1))
                                , red_hitbox[0] , red_hitbox[1] , fc='red' , ec="black" , zorder=10.)
            plt.gca().add_patch(blueBoxXY)
            plt.gca().add_patch(redBoxXY)
        else:
            plt.scatter( blue_loc[0]  , blue_loc[1] , label='Blue' , color='blue' , zorder=10.)
            plt.scatter( red_loc[0] , red_loc[1]+buildingDims[1]*(sepCoeff-1) , label='Red' , color='red' , zorder=10.)
    if plotAllShots:
        for i, color in enumerate(colors):
            if (shotWasHit[i]):
                plt.plot( [red_loc[0],shots[i][0]] , [red_loc[1],ylevel] , color='black' , linewidth=3 , zorder=4.0)
            else:
                plt.plot( [red_loc[0],shots[i][0]] , [red_loc[1],ylevel] , color=color , zorder=2.0)
    
    if (len(blueShot) != 0):
        plt.plot( [blue_loc[0],blueShot[0]] ,  [blue_loc[1],blueShot[1]] , color='white' , linewidth=3 , zorder=6.0)
    if (len(emceeOutput) != 0):
        plt.scatter( emceeOutput[0]  , emceeOutput[1] , s=81 , label='mcmcOutput' , marker='o' , ec='black',linewidth=2 , color='Yellow' , zorder=15.)
    elif (len(multipleMC_outs) != 0):
        meds = np.transpose(multipleMC_outs[:,:1]) #  x,y,z vals
        errs = np.transpose(multipleMC_outs[:,1:3]) # x,y,z lower and upper errors
        if plotDotsForManyMC:
            plt.errorbar(meds[0][0] , meds[1][0]+buildingDims[1]*(sepCoeff-1) , xerr=errs[0] , yerr=errs[1] , fmt='o' , color='yellow' , markeredgecolor='black' , ecolor='red' , zorder=15.)
            if plotMultMCshots:
                for i in range(len(multipleMC_outs)):
                    plt.scatter( shots[i][0][0] , shots[i][0][1] , color='r' , s=100 , marker='o' , zorder=10.)
                    plt.plot( [ shots[i][0][0],shots[i][1][0] ] , [ shots[i][0][1],shots[i][1][1] ] , color='black' )
        else:
            patchCollectXY = make_error_boxes( meds[0][0] , meds[1][0]+buildingDims[1]*(sepCoeff-1) , errs[0] , errs[1] )
            plt.gca().add_collection(patchCollectXY)
            plt.errorbar(meds[0][0] , meds[1][0]+buildingDims[1]*(sepCoeff-1) , xerr=errs[0] , yerr=errs[1] , fmt='None' , ecolor='white' , zorder=15.)
            #plt.scatter(meds[0][0] , meds[1][0] , s=100 , color='y' , zorder=20.)
    plt.xlim( -0.1*buildingDims[0] , 1.1*buildingDims[0] )
    plt.ylim( -0.1*buildingDims[1] , (2.1*buildingDims[1]*sepCoeff)+bSeper )

    
    
    # Plotting the side view of the map
    plt.subplot(1,2,2)
    plt.title("Side Drone View ("+r"$\phi\,$"+" axis)"); plt.xlabel(r"$\beta\,$"+' axis (cm)'); plt.ylabel(r"$\gamma\,$"+' axis (cm)')
    blueBuildingYZ = plt.Rectangle((0,0) , buildingDims[1] , buildingDims[2] , fc='silver',ec="black",zorder=1.)
    redBuildingYZ = plt.Rectangle((buildingDims[1]*sepCoeff+bSeper,0) , buildingDims[1] , buildingDims[2] , fc='silver',ec="black",zorder=1.)
    plt.gca().add_patch(blueBuildingYZ)
    plt.gca().add_patch(redBuildingYZ)
    plt.gca().set_facecolor('dimgray')
    
    if plotPlayers:
        if (plotHitBoxes):
            blueBoxYZ = plt.Rectangle( (blue_loc[1]-blue_hitbox[1]/2,blue_loc[2]-blue_hitbox[2]/2)
                                , blue_hitbox[1] , blue_hitbox[2] , fc='blue',ec="black",zorder=10.)
            redBoxYZ = plt.Rectangle( (red_loc[1]-red_hitbox[1]/2+buildingDims[1]*(sepCoeff-1) , red_loc[2]-red_hitbox[2]/2)
                                , red_hitbox[1] , red_hitbox[2] , fc='red',ec="black",zorder=10.)
            plt.gca().add_patch(blueBoxYZ)
            plt.gca().add_patch(redBoxYZ)
        else:
            plt.scatter( blue_loc[1]  , blue_loc[2] , label='Blue' , color='blue' , zorder=10.)
            plt.scatter( red_loc[1]+buildingDims[1]*(sepCoeff-1) , red_loc[2] , label='Red' , color='red' , zorder=10.)
    if plotAllShots:   
        for i, color in enumerate(colors):
            if (shotWasHit[i]):
                plt.plot( [red_loc[1],ylevel],[red_loc[2],shots[i][2]] , color='black' , linewidth=3 , zorder=4.) 
            else:
                plt.plot( [red_loc[1],ylevel],[red_loc[2],shots[i][2]] , color=color , zorder=2.)
    
    if (len(blueShot)!=0):
        plt.plot( [blue_loc[1],blueShot[1]] ,  [blue_loc[2],blueShot[2]] , color='white' , linewidth=3 , zorder=6.0)
    if (len(emceeOutput) != 0):
        plt.scatter( emceeOutput[1]  , emceeOutput[2] , s=81 , label='mcmcOutput' , marker='o' , ec='black',linewidth=2 , color='Yellow' , zorder=15.)
    elif (len(multipleMC_outs) != 0):
        if plotDotsForManyMC:
           plt.errorbar(meds[1][0]+buildingDims[1]*(sepCoeff-1) , meds[2][0] , xerr=errs[1] , yerr=errs[2] , fmt='o' , color='yellow' , markeredgecolor='black' , ecolor='red' , zorder=15.)
           if plotMultMCshots:
                for i in range(len(multipleMC_outs)):
                    plt.scatter( shots[i][0][1] , shots[i][0][2] , color='r' , s=100 , marker='o' , zorder=10.)
                    plt.plot( [ shots[i][0][1],shots[i][1][1] ] , [ shots[i][0][2],shots[i][1][2] ] , color='black' )
        else:
            patchCollectYZ = make_error_boxes( meds[1][0]+buildingDims[1]*(sepCoeff-1) , meds[2][0] , errs[1] , errs[2] )
            plt.gca().add_collection(patchCollectYZ)
            plt.errorbar(meds[1][0]+buildingDims[1]*(sepCoeff-1) , meds[2][0] , xerr=errs[1] , yerr=errs[2] , fmt='None' , ecolor='white' , zorder=15.)
            #plt.scatter(meds[1][0] , meds[2][0] , s=10 , color='y' , zorder=20.)
    plt.xlim( -0.1*buildingDims[1] , (2.1*buildingDims[1]*sepCoeff)+bSeper )
    plt.ylim( -0.1*buildingDims[2] , 1.1*buildingDims[2] )
    
    plt.tight_layout()






"""
The following functions calculate logarithmic likelihoods, priors and posteriors to feed to the MCMC.
Version 2 of these functions can be omitted
"""
def log_likelihood(params, x, y):
    alpha, beta, gamma = params
    like = 0     
    for i in range(len(x)):
        like += np.log( beta**2 / ( (beta**2 + (x[i] - alpha)**2) * (beta**2 + (y[i] - gamma)**2) ) )
    #return np.log( like / np.pi**2)
    return like
def log_likelihood2(params, x, y , buildingDims):
    alpha, beta, gamma = params
    like = 0     
    for i in range(len(x)):
        like += np.log( beta**2 / ( ( np.arctan((buildingDims[0]-alpha)/beta)-np.arctan(-alpha/beta) ) * 
                          ( np.arctan((buildingDims[2]-gamma)/beta)-np.arctan(-alpha/gamma) ) * 
                          ( beta**2 + (x[i] - alpha)**2 ) * ( beta**2 + (y[i] - gamma)**2 )     )    )
    return like

def log_prior(params, param_bounds):
    alpha, beta, gamma = params
    alpha_min, alpha_max = param_bounds[0];  beta_min, beta_max = param_bounds[1];  gamma_min, gamma_max = param_bounds[2]
    alpha_cond = (alpha_min < alpha and alpha_max > alpha)
    beta_cond = (beta_min < beta and beta_max > beta)
    gamma_cond = (gamma_min < gamma and gamma_max > gamma)
    if alpha_cond and beta_cond and gamma_cond:
        return 0.0 # additive constant is not necessary 
    return -np.inf 

def log_posterior(params, x, y, param_bounds):
    lp = log_prior(params, param_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y)
def log_posterior2(params, x, y, param_bounds , knownP):
    lp = log_prior(params, param_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood2(params, x, y, knownP)




"""
Function used to start the encounter between the player and the antagonist
"""
def beginGame(blue_loc=[] , red_loc=[] , blue_hitbox=[] , red_hitbox = [] , buildingDims=[] , bSeper=0. , 
              hit_tolerance=0 , confidence_tolerance=0.68 , emceeAtEach=10 , start_MCing_at = 20 , shotsAllowed=80 , 
              log_post1=True , easyMode=False , hardMode=True , doPrint=True , doPlot=True , doCornerPlot=True):
    """
    blue_loc , red_loc , buildingDims , bSeper : 
            Refer to the setup() function for explanations on these parameters
    confidence_tolerance , emceeAtEach , start_MCing_at : 
            - 'confidence_tolerance' refers to the n% confidence region we want the algorithm to use to know when to stop.
              It will do so once the alpha and gamma values inferred have their n% confidence regions less than the opponent hit-box dimensions.
            - The algorithm performs successive MCMCs to better continually better constrain the inferred location.
              'emceeAtEach' tells the algorithm to perform a new MCMC after 'n' number of new shots detected.
            - 'start_MCing_at' tells the algorithm to start the first MCMC after 'n' number of shots detected
    easyMode , hardMode : boolean
            Decides if the allowed firing angles for the Red are spread out or constrained to hit Blue's building. Default is hardMode
    doPrint , doPlot , doCornerPlot : boolean
            'doPrint' allows certain additional info to be printed. 'doPlot' and 'doCornerPlot' decide if the game map and parameter corner
            maps from the MCMC are created.
    """
    
    buildingDims, bSeper, red_loc, blue_loc, red_hitbox, blue_hitbox = setup(blue_loc=blue_loc,red_loc=red_loc,buildingDims=buildingDims,bSeper=bSeper,blue_hitbox=blue_hitbox,red_hitbox=red_hitbox)
        
    # shotsSeenByDrone collects the shot coordinates used to run the MCMC
    # shotsForCheckHit collects the shot coordinates used to check if a shot was a hit against Blue
    # shotsForVisualize collects shot coordinates to be plotted onto the map
    # shotsReceived collects shots that were a hit on Blue
    # shotWasHit collects boolean values on whether a certain shot was a hit or not
    shotsSeenByDrone = [];  shotsForCheckHit = [];  shotsForVisualize = [];  shotsReceived = [];  shotWasHit = [];  shotsFiredByRed = 0
    alpha_min = 0.;  alpha_max = buildingDims[0]
    beta_min = 0;  beta_max = buildingDims[1] + bSeper
    gamma_min = 0.;  gamma_max = buildingDims[2]
    
    if (easyMode):
        theta_range = [-np.pi/2,np.pi/2]
        phi_range = [-np.pi/2,np.pi/2]
        print("Easy Mode. All firing angles all permitted")
    elif (hardMode):
        theta_range = [  np.arctan(-red_loc[0]/(red_loc[1]-buildingDims[1])) , 
                         np.arctan((buildingDims[0]-red_loc[0])/(red_loc[1]-buildingDims[1]))  ]
        phi_range = [  np.arctan(-red_loc[2]/(red_loc[1]-buildingDims[1])) , 
                       np.arctan((buildingDims[2]-red_loc[2])/(red_loc[1]-buildingDims[1]))  ]
        print("Hard mode. Firing angles will now be directed strictly towards Blue's building.")
    else:
        theta_range = [-np.pi/2,np.pi/2]
        phi_range = [  np.arctan(-red_loc[2]/(red_loc[1]-buildingDims[1])) , np.pi/2  ]
        print("Normal mode. Firing angles are now restricted so that they dont hit the ground before exiting their building of origin.")
    print("")
    
    # MCMC parameters
    num_iter = 5000;  paramDims = 3;  nwalkers = 10
    param_bounds = np.array(((alpha_min, alpha_max), (beta_min, beta_max), (gamma_min, gamma_max)))
    # The initial parameter guesses are directly at the center of the opponents building
    bestShot = np.array( ( (alpha_max-alpha_min)/2 , (beta_max-beta_min)/2 , (gamma_max-gamma_min)/2 )  )
    # Confidence regions used to determine when the MCMC analysis ends
    confidenceRegionX = alpha_max-alpha_min
    confidenceRegionY = beta_max-beta_min
    confidenceRegionZ = gamma_max-gamma_min
    numMCMCs = 0
    blueShotNotTaken = True
    blueDied = False
    
    # While the X and Z confidence regions are to high, continue receiving shots
    #while((confidenceRegionX > red_hitbox[0] or confidenceRegionZ > red_hitbox[2]) and shotsFiredByRed < shotsAllowed):
    while(shotsFiredByRed < shotsAllowed):
        
        theta = np.random.uniform( theta_range[0] , theta_range[1] )
        phi = np.random.uniform( phi_range[0] , phi_range[1] )
        shotsSeenByDrone.append( shotAtLocationY(theta , phi , red_loc , yloc=buildingDims[1]) ) # bSeper stuff here
        shotsForCheckHit.append( shotAtLocationY(theta , phi , red_loc , yloc=blue_loc[1]) )
        shotsForVisualize.append( shotAtLocationY(theta , phi , red_loc , yloc=0) )
        shotsFiredByRed += 1
        
        if (wasShotAHit(blue_loc , shotsForCheckHit[-1] , blue_hitbox)):
            if doPrint: print("Shot hit Blue at " + str(shotsForCheckHit[-1]))
            shotsReceived.append( shotsForCheckHit[-1] )
            shotWasHit.append(True)
        else:
            shotWasHit.append(False)
            
        if (hit_tolerance < len(shotsReceived)):
            print("Blue DIED. Red landed " + str(len(shotsReceived)) + " out of " + str(shotsFiredByRed) + " of his shots")
            #if doPlot: plotShots2D( blue_loc , red_loc , buildingDims , bSeper , shotsForVisualize , 
            #                        shotWasHit , blue_hitbox , red_hitbox , ylevel=0)
            blueDied = True
            break
        
        if (np.mod(shotsFiredByRed,emceeAtEach) == 0 and shotsFiredByRed >= start_MCing_at and blueShotNotTaken==True):
            
            # MCMC part of the code
            initial_pos = bestShot + 0.01 * np.random.randn(nwalkers, paramDims)
            shots = np.transpose(shotsSeenByDrone)
            shotsXZ = np.array([shots[0],shots[2]])
            # this if-else is used if we wish to experiment with other likelihood functions
            if (log_post1):
                sampler = emcee.EnsembleSampler(nwalkers, paramDims, log_posterior, args=(shotsXZ[0], shotsXZ[1], param_bounds))
            else:
                sampler = emcee.EnsembleSampler(nwalkers, paramDims, log_posterior2, args=(shotsXZ[0], shotsXZ[1], param_bounds, buildingDims))
            sampler.run_mcmc(initial_pos, num_iter, progress=True);
            
            param_collection = sampler.get_chain(flat=True , discard=100)
            medians = np.quantile(np.transpose(param_collection) , 0.5 , axis=1) 
            lowerQuantiles = np.quantile(np.transpose(param_collection) , 0.5-(confidence_tolerance/2) , axis=1) 
            upperQuantiles = np.quantile(np.transpose(param_collection) , 0.5+(confidence_tolerance/2) , axis=1)
            
            bestShot = medians
            confidenceRegionX = upperQuantiles[0]-lowerQuantiles[0]
            confidenceRegionY = upperQuantiles[1]-lowerQuantiles[1]
            confidenceRegionZ = upperQuantiles[2]-lowerQuantiles[2]
            numMCMCs += 1
            
            if doPrint:
                print("Best Shot is now in direction of " + str( (bestShot[0],bestShot[1]+buildingDims[1],bestShot[2]) ))
                print("Confidence interval of bestShot in x,y,z is " + str((confidenceRegionX,confidenceRegionY,confidenceRegionZ)) )
                print("")
            
            if ( confidenceRegionX <= red_hitbox[0] and confidenceRegionZ <= red_hitbox[2] and blueShotNotTaken==True):
                blueShotNotTaken = False
                # take best shot, hit the other (hopefully)
                # Since the MCMC searches for a coordinate within the opponents building, adjust the inferred coordinate
                # to accurately represent it on the map
                bestShot[1] = bestShot[1] + buildingDims[1]
                if (wasShotAHit(red_loc , bestShot , red_hitbox , shotFrom=blue_loc)):
                    print("SUCCESSFUL HIT. Bayesian Inference triumphs in battle! Blue can now go home to his family and celebrate")
                    break
                else:
                    print("Blue missed! Bayesian Inference is a lie! Screw you Bayes!")
                    print('')
                    
            elif ( shotsAllowed == shotsFiredByRed and blueShotNotTaken==True):
                blueShotNotTaken = False
                # take best shot even though your confidence is not optimal, hit the other (hopefully)
                bestShot[1] = bestShot[1] + buildingDims[1]
                if (wasShotAHit(red_loc , bestShot , red_hitbox , shotFrom=blue_loc)):
                    print("Blue got lucky and lived. With those confidence intervals, that was almost like shooting blind. But his desperate shot worked")
                    break
                else:
                    print("What a conundrum. Since Red ran out of ammo, Blue had to take the shot with the data he had, and missed."
                          +" After some discussion, it has been agreed that life and death shall now be decided by a game of rock-paper-scissors")
                    print("")
                    playRPS("Blue" , "Red")
                    break
    
    
    
    if doCornerPlot and shotsFiredByRed >= start_MCing_at:
        # adding the distance between buildings and the buildings y-length to the beta parameter list
        for i in range(len(param_collection)):
            param_collection[i][1] += buildingDims[1]#+bSeper/2
        corner.corner(param_collection, labels=[r"$\alpha\,$[cm]", r"$\beta\,$[cm]" , r"$\gamma\,$[cm]"] ,
              quantiles = [0.16,0.5,0.84] , show_titles=True )
    
    
    
    if shotsAllowed < start_MCing_at or ( blueDied and blueShotNotTaken) :
        bestShot[1] = bestShot[1] + buildingDims[1]
    
    # Using the angles of blue shot, get a coordinate point such as to plot blue shot on the map
    visualizeBestShot = shotAtLocationY( -np.arctan( (bestShot[0]-blue_loc[0])/(bestShot[1]-blue_loc[1]) ) ,
                               -np.arctan( (bestShot[2]-blue_loc[2])/(bestShot[1]-blue_loc[1]) ) , blue_loc , yloc=2.1*buildingDims[1]+bSeper)
    if doPlot:
        if (  blueShotNotTaken and shotsAllowed >= start_MCing_at  ):
            plotShots2D( blue_loc , red_loc , buildingDims , bSeper , shotsForVisualize , shotWasHit , 
                                  blue_hitbox , red_hitbox , ylevel=0 , emceeOutput=bestShot)
        else:
            plotShots2D( blue_loc , red_loc , buildingDims , bSeper , shotsForVisualize , shotWasHit , 
                                  blue_hitbox , red_hitbox , ylevel=0 , blueShot=visualizeBestShot , emceeOutput=bestShot)
    
    
    if shotsAllowed < start_MCing_at :
        # take best shot anyways, see what gives
        if (wasShotAHit(red_loc , bestShot , red_hitbox , shotFrom=blue_loc) and blueShotNotTaken==True):
            print("Even without his MCMC, Blue got lucky and lived. With those confidence intervals, that was almost like shooting blind. But his desperate shot worked")
        else:
            print("What a conundrum. Since Red ran out of ammo, Blue had to take the shot with the data he had, and missed."
                  +" After some discussion, it has been agreed that life and death shall now be decided by a game of rock-paper-scissors")
            print("")
            playRPS("Blue" , "Red")
    
    
    if not blueDied and shotsAllowed <= shotsFiredByRed and not blueShotNotTaken:
        print("What a conundrum. Both of you ran out of ammo."
              +" After some discussion, it has been agreed that life and death shall now be decided by a game of rock-paper-scissors")
        print("")
        playRPS("Blue" , "Red")
        
    
    print('')    
    print(str(shotsFiredByRed)+" shots were fired by Red")
    print(str(numMCMCs)+" MCMCs were executed")
    print("THE END")




"""
This function used to make the error boxes instead of error bars when plotting data
Taken from https://matplotlib.org/3.1.3/gallery/statistics/errorbars_and_boxes.html
"""
def make_error_boxes(xdata, ydata, xerror, yerror, facecolor='black', alpha=1.):

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = plt.Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha , zorder=15.)

    return pc


# dumb rock-papers-scissors game
def playRPS(name1 , name2):
    options = ['ROCK' , 'PAPER' , 'SCISSORS']
    while (True):
        
        i = np.random.randint(1,4 , 2) # 1=rock , 2=paper , 3=scissors
        print(name1 + " chooses " + options[i[0]-1] + ". " + name2 + " chooses " + options[i[1]-1])
        
        if   i[0]-i[1]==0 :
            print("Draw")
            continue
        elif i[0]-i[1]==-1 or i[0]-i[1]==2:
            print("FATALITY. " + name1 + " DIED. " + name2 + " WINS")
            return
        elif i[0]-i[1]==1 or i[0]-i[1]==-2:
            print("FATALITY. " + name2 + " DIED. " + name1 + " WINS")
            return
            
            
            
            