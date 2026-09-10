import numpy as np
import math
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

folder = os.path.exists('验证简单线性模型相关系数(OU噪声)')
if not folder:                                      # Check whether the folder exists; create it if it does not.
		os.makedirs('验证简单线性模型相关系数(OU噪声)') 
        
        
# Get the current time.
program_start=datetime.now()
print('\n开始运行:',program_start,'\n')

# Model parameters.
VL=-70      #mV
#gc=8      #mS/cm^2    2.1   coupling conductance
p=0.5       #0.5
Cm=5        #uF/cm^2    3
gL=8      #mS/cm^2    0.1

r1 = gL/Cm
a = 1/p/Cm
b = 1/(1-p)/Cm

# Time-step precision and simulation duration settings.
dt=0.1    #ms
time=600  #ms
n=round(time/dt)

# Noise parameter settings.
alpha=1

# Smoothed square-wave input settings.
Id_base = 0         
Is_base = 0
td0=200    #ms
low_ratio_d = 0.4
ts0=200   #ms
low_ratio_s = 0.5

# Correlation coefficient of the smoothed square waves (as a function of phase shift). This corresponds to the square-wave input settings; do not arbitrarily change the two low_ratio values above.
def Rou(phi): # Pearson correlation of the two deterministic square-wave inputs used in the theoretical calculation, written as ρ(Φ).
    if 0 <= phi < 0.4:
        return (0.2-phi)/0.24495
    elif 0.4 <= phi < 0.5:
        return -0.81649
    elif 0.5 <= phi <0.9:
        return (phi-0.7)/0.24495
    else:
        return 0.81649
    

# Calculate the theoretical correlation coefficient between Vs and Vd.
def theoy_Correlation(args):
    phi, sigma, Ad, As, gc= args
    
    # Calculate the group of coefficients used in the formula.

    r2 = r1 + (a+b)*gc
    A = a*b/r1 + a*a/r2
    B = a*b*(1/r1-1/r2)
    C1 = a*b/(r1-alpha)
    C2 = a*a/(r2-alpha)
    D1 = C1
    D2 = a*b/(r2-alpha)
    E = a*b/r1 + b*b/r2
    C3 = b*b/(r2-alpha)

    CX = (C1+C2)**2 + (D1-D2)**2 + (C1**2+D1**2)*alpha/r1 + (C2**2+D2**2)*alpha/r2 - 4*(C1*(C1+C2)+D1*(D1-D2))*alpha/(alpha+r1) - 4*(
        C2*(C1+C2)-D2*(D1-D2))*alpha/(alpha+r2) + 4*(C1*C2-D1*D2)*alpha/(r1+r2)

    CY = (C1+C3)**2 + (D1-D2)**2 + (C1**2+D1**2)*alpha/r1 + (C3**2+D2**2)*alpha/r2 - 4*(C1*(C1+C3)+D1*(D1-D2))*alpha/(alpha+r1) - 4*(
        C3*(C1+C3)-D2*(D1-D2))*alpha/(alpha+r2) + 4*(C1*C3-D1*D2)*alpha/(r1+r2)

    CXY = (2*C1+C2+C3)*(D1-D2) + 2*C1*D1*alpha/r1 - D2*(C2+C3)*alpha/r2 - 2*((2*C1+C2+C3)*D1+2*C1*(D1-D2))*alpha/(alpha+r1) - 2*(
        -(2*C1+C2+C3)*D2+(C2+C3)*(D1-D2))*alpha/(alpha+r2) + 2*(-2*C1*D2+D1*(C2+C3))*alpha/(r1+r2)


    x = 0.5*As
    y = 0.4899*Ad
    Var_X = (A*x)**2 + (B*y)**2 + 2*A*B*x*y*Rou(phi) + CX*sigma**2
    Var_Y = (B*x)**2 + (E*y)**2 + 2*B*E*x*y*Rou(phi) + CY*sigma**2
    Cov_XY = A*B*x**2 + B*E*y**2 + (A*E+B**2)*x*y*Rou(phi) + CXY*sigma**2
    
    return Cov_XY/math.sqrt(Var_X*Var_Y) # Directly calculate and return the final formula from the appendix; the coefficients above feed into this expression, which returns a scalar value such as 0.62.


# Numerically estimate the correlation coefficient. Each call to get_correlation(args) fixes one set of input, noise, and coupling parameters, runs one long simulation, and returns one correlation coefficient.
def get_correlation(args):
    phi, sigma, Id0, Is0, gc = args
    
    N = n*neuronNumber    # Extend a single time trajectory by a factor of 50 to increase the number of samples used to estimate the correlation.
    Id=np.zeros(N+1) # Create the dendritic input as an array of length N+1, then fill it point by point with the square wave.
    for i in range(N+1): # Here, i is the simulation time-step index.
        k=((i-td0/dt*phi)%(td0/dt))/(td0/dt) # Relative position within one dendritic-input period after applying the phase shift at the current time.
        if k<=low_ratio_d:
            Id[i]=Id_base
        else:
            Id[i]=Id0
    

    Is=np.zeros(N+1)
    for i in range(N+1):
        k=i%(ts0/dt)/(ts0/dt) # There is no phi here; the somatic input is treated as the phase reference.
        if k<=low_ratio_s:
            Is[i]=Is_base
        else:
            Is[i]=Is0
    
    correlation=0 # Initialize the variable.
    

    
   	# Generate OU noise; z1 and z2 are independent Gaussian random driving terms used to drive the OU process at each time step.
    z1=np.random.normal(0,sigma*math.sqrt(2*alpha),N) #scaled Gaussian random variables
    z2=np.random.normal(0,sigma*math.sqrt(2*alpha),N)
# noise1 and noise2 are OU-noise sequences.
    noise1=np.zeros(N+1) # Initialize OU noise.
    for i in range(N):
	    noise1[i+1]=noise1[i]-alpha*noise1[i]*dt+z1[i]*np.sqrt(dt) # Euler-Maruyama discretization; -alpha*noise1[i]*dt represents the mean-reversion term of the OU noise.
    noise2=np.zeros(N+1)
    for i in range(N):
	    noise2[i+1]=noise2[i]-alpha*noise2[i]*dt+z2[i]*np.sqrt(dt)
   
   
   	#Euler method
    Vs=np.zeros(N+1) # Why N+1 rather than N? There is an initial value V(0), and after N updates the trajectory reaches V_N.
    Vs[0]=VL # VL is the natural equilibrium potential when there is no external input.
    Vd=np.zeros(N+1)
    Vd[0]=VL

    
    for i in range(1,N+1,1): # range(start, stop, step): each iteration calculates the voltage at the next time step, V[i].
        
        Vs[i]=(-r1*(Vs[i-1]-VL) + a*gc*(Vd[i-1]-Vs[i-1]) + a*(Is[i-1]+noise1[i-1]))*dt + Vs[i-1]  # Vs[i] represents Vs at time t_i = i*dt.
        Vd[i]=(-r1*(Vd[i-1]-VL) + b*gc*(Vs[i-1]-Vd[i-1]) + b*(Id[i-1]+noise2[i-1]))*dt + Vd[i-1] 

    correlation = np.corrcoef(Vs[2000:], Vd[2000:])[0,1] # Calculate ρ(Vs,Vd), discarding the first 200 ms.

    return correlation # This could also be written in one line: return np.corrcoef(Vs[2000:], Vd[2000:])[0,1].


# Centralized parameter-settings section ******************************************************

neuronNumber = 10   #number of two-compartment neuron models
pointNum = 25        # Number of uniformly sampled points along one dimension; the total number of sampled points is its square.

# Of the following parameters, exactly two must be selected as the x- and y-axis variables by specifying a two-value range in brackets; the remaining parameters must be scalar values.
phi = 0.4   #phase shift 0~1  0.4 
sigma = [100,580]  #standard deviation(intensity) for noises  350
Id0 = [100,580]   #pA  340
Is0 = 450   #pA  600 
gc = 2.1 #mS/cm^2    2.1   coupling conductance

# End of parameter settings ***************************************************************

labels=['Phase Shift', 'Noise intensity(pA)', 'Dentritic Input(pA)', 'Somatic Input(pA)', 'coupling conductance(mS/cm^2)']
labels_short=['phi','sigma','Id0','Is0', 'gc']
parameters=[phi,sigma,Id0,Is0,gc]
variable=[]
index=[]
describtion=''
data = [[],[],[],[],[],[],[],[]]
args=[0,0,0,0,0] # This eventually becomes args=[Φ,σ,Id,Is,gc].
for i in range(5):
    if isinstance(parameters[i], list):
        index.append(i)
        variable.append(np.linspace(parameters[i][0], parameters[i][1], pointNum))
    else:
        describtion = describtion + labels_short[i] + '=' +str(parameters[i]) + ','
        data[i] = np.ones(pointNum*pointNum)*parameters[i]
        args[i] = parameters[i]
iy,ix = index
correlations = np.zeros([3, pointNum, pointNum])


for i in range(pointNum):
    args[iy] = variable[0][i] # Sweep σ.
    for j in range(pointNum):
        print("\r进度：{:.2f}%".format((i*pointNum+j+1)/pointNum**2*100),end="")
        args[ix] = variable[1][j] # Sweep Id0.
        
        correlations[0][i][j] = abs(theoy_Correlation(args)) # Correlation magnitude given by the theoretical formula.
        correlations[1][i][j] = abs(get_correlation(args)) # Correlation magnitude obtained from numerical simulation.
        correlations[2][i][j] = abs(correlations[1][i][j] - correlations[0][i][j]) # Difference between the two.
        
        data[iy].append(variable[0][i])
        data[ix].append(variable[1][j])
        data[5].append(correlations[0][i][j])
        data[6].append(correlations[1][i][j])
        data[7].append(correlations[2][i][j])
            

# Plot and save figures.
titles = ['Theory Correlation', 'Simulative Correlation', 'Error']
Vmax = [1, 1, 0.03]
letters = [r'$\rho$1', r'$\rho$2', r'$\Delta\rho=|\rho1-\rho2|$']
fig, axs = plt.subplots(2, 2, figsize=(18,14)) # Create a 2×2 figure.

for i,j in [(0,0), (0,1), (1,0)]: # Here (i,j) gives the subplot position: upper left, upper right, and lower left. The lower-right subplot (1,1) is not used.
    
    k = i*2 + j # Convert the two-dimensional subplot position to a single index corresponding to the three elements in titles.
    axs[i][j].set_xticks(np.linspace(0,pointNum-1,5), np.linspace(parameters[ix][0], parameters[ix][1], 5)) # Inside the image, index 0 is displayed as parameter value 100 and index 24 as 580, with several intermediate tick labels showing the actual parameter values.
    axs[i][j].set_yticks(np.linspace(0,pointNum-1,5), np.linspace(parameters[iy][0], parameters[iy][1], 5)) # As above, set the y-axis ticks to display the actual parameter values for the reader.
    axs[i][j].set_ylabel(labels[iy]) # The x- and y-axes are the two parameters selected for scanning.
    axs[i][j].set_xlabel(labels[ix]) # The x- and y-axes are the two parameters selected for scanning.
    axs[i][j].set_title(titles[k], weight='bold')
    ax = axs[i][j].imshow(correlations[k], cmap='Blues', origin = 'lower') # Each pixel corresponds to one parameter combination. origin='lower' places array element [0,0] at the lower left, so parameter values increase from 100 to 580 in the conventional mathematical directions: x left-to-right and y bottom-to-top.
    
    clb=fig.colorbar(ax, ax=axs[i][j], orientation='vertical') # Here, ax is the heatmap created by the previous line; add a colorbar for that heatmap.
    #clb.set_ticks(np.linspace(0,Vmax[k],11))
    #clb.update_ticks()
    clb.ax.set_title(letters[k]) # Add a title above the colorbar.

axs[1][0].set_xlabel(labels[ix] + '\n\nDefault: ' + describtion + '\nand ' + str(neuronNumber) + ' cells')
axs[1][1].axis('off') # Only three plots are needed, so turn off the lower-right subplot.
plt.show()
fig.savefig('验证简单线性模型相关系数(OU噪声)/image('+labels_short[ix]+','+labels_short[iy]+')_'+describtion+'.png',dpi = 300,bbox_inches='tight')


# Save to a CSV file.
dataframe = pd.DataFrame()
for i in range(5):
    dataframe[labels_short[i]]=data[i]
    
dataframe['theory_correlation']=data[5]
dataframe['simulative_correlation']=data[6]
dataframe['errors']=data[7]

dataframe.to_csv('验证简单线性模型相关系数(OU噪声)/Verify_Correlation'+'('+describtion+').csv',index=False,sep=',')

program_end=datetime.now()
print('\n\n计算结束:',program_end)
print('程序运行总时长:',program_end - program_start)









