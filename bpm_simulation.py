import numpy as np
from math import exp, sqrt
from datetime import datetime
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import pickle

EL=-70    #mV
tau_s=16  #ms
Cs=370    #pF
tau_ws=100  #100ms
b=200       #pA  strength of spike-triggered adaptation,
VT=-50    #mV

tau_d=7   #ms
#gd=1200   #pA
Cd=170    #pF
#cd=2600   #pA
tau_wd=30   #30ms
a=13        #nS=nA/V=pA/mV

Ed=-38    #mV
Dd=6      #mV

Vr=-70    #mV  reset voltage after a spike

# Noise parameter settings.
alpha=1 

# Noise-free signal-input settings.
td0=200    #ms
low_ratio_d=0.4
ts0=200   #ms
low_ratio_s=0.5

neuronNumber = 1   #number of two-compartment neuron models
pointNum = 9     # Number of sampled points for the independent variable.

# Of the following 11 parameters, only one can be selected as the control variable by specifying a two-value range in brackets; the remaining parameters must be scalar values.
phi = [0,1]   #phase shift 0~1  0.4
sigma = 300 #standard deviation(intensity) for noises  350
Id0 = 200   #pA  340
Is0 = 800   # pA  600; the value must be >500=baseline.
Id_base = 0         #0
Is_base = 500       #500
gs = 1300           #1300pA
cd = 2600       #2600pA
gd = 1200       #1200pA
gc = 0        # 2.1 mS; set this to 0 if the p and gc terms are not intended to be included.
p = 0.5         #0<p<1
#👆they are parameters in this model

#define the time scale👇
N = 6000 #total bins########
dt = 0.1 #time step in ms######
t0 = 0
time_total = N * dt + t0 #total simulation time#####
time = int(round(N * dt))  # Represents the number of 1 ms bins.
t = np.linspace(t0, time_total, N + 1) # Map array index i to the actual time t[i]. Dividing the interval into N equal segments means each step is theoretically dt. (start, stop, num)

labels = [
    "Phase Shift", "Noise intensity / pA", "Dendritic Input / pA", "Soma Input / pA",
    "Id_base / pA", "Is_base / pA", "gs / pA", "cd / pA", "gd / pA", "gc", "p"
]
labels_short = ["phi","sigma","Id0","Is0","Id_base","Is_base","gs","cd","gd","gc","p"]

parameters = [phi, sigma, Id0, Is0, Id_base, Is_base, gs, cd, gd, gc, p] # 11 parameters pack (use for FirstDepict scanning)
def FirstDepict(parameters, labels_short, pointNum):
  #parsing 11 parameters:
  #If an parameter is a list, such as [a, b], treat it as the "range of variables to be scanned".
  #Treat other parameters as fixed values, collect them into args, and generate a description string.
#1.prepare the container
    describtion_parts = [] #Collect string fragments using a list, and then join them
    args = [None] * 11 #fix parameters list,length 11, also put a value at variables,
 #(If any bit is still None, mathematical operations in the
 # simulation will result in an error, so we want args to always be "computable".)
    index = None # define a number. Default: No variables
    variable = np.array([]) # define a vertical list,string elements only. Default: Empty array

#2.Identify which parameters are list variables, and remember their position
    variable_indices = []

    for i, pval in enumerate(parameters):
        if isinstance(pval, list):
            variable_indices.append(i)
#i is the position of the parameters, p is the value of that parameter. "enumerate" produces
#(index, value) pairs while looping. "isinstance" determind whether p is a list or not, if
#a list, the list is a variable in the simulation, otherwise a constant.

#3.If there are multiple variable parameters, handle them according to the rule of "only taking the first one"

    if len(variable_indices) >= 1: # "len"means how many elements does this list have.In variable_indices,is there at
#least one varying parameters?
        index = variable_indices[0] #index is the first element of the list
        variable = np.linspace(parameters[index][0], parameters[index][1], pointNum)

#4.fill the arg with the string(string fit the describtion)
    for i in range(11): #Let i become0,1,2,3,4,5,6,7,8,9,10 in sequence, and process the 11 parameters one by one.
        pval = parameters[i] #Take the value of the i-th parameter from parameters and put it into the variable p

        if i == index: #Is the current parameter i the ‘variable parameter to be scanned’?
            args[i] = variable[0] if variable.size > 0 else None
        #The simplified if-else statement: if variable contains something (length greater than 0), its
        #first value variable[0] is added to args[i]; otherwise, None is added.
        else: #if the current parameter is a fixed constant
            args[i] = pval #put fixed parameters at corresponding location of args
            describtion_parts.append(f"{labels_short[i]}={pval}") #f"" format string, add fixed parameters into description_parts
    describtion = ", ".join(describtion_parts) #use "," as connector, add string in the list by ","
    return index, args, variable, describtion #return these four values for future use.
#index: Modify which parameter is changing. args:Fixed parameter table(variable positions filled with default values)
#variable: Scan array of values. description:Fixed parameter description string (used for figure titles, file names)

# Obtain the correlation coefficient of the input signals, with minimal safeguards against division by zero, NaN, and Inf.
def Correlation_IsId(args):
    phi, sigma, Id0, Is0, Id_base, Is_base = args # The commas here perform unpacking: assign the six elements of the list/tuple args to the six variables in order.

    # 1) Most common failure case: zero amplitude -> division by zero.
    amp_s = (Is0 - Is_base)
    amp_d = (Id0 - Id_base)
    if amp_s == 0 or amp_d == 0:
        return 0.0 # Exit the function immediately and return 0; 0.0 is a double or float value, while 0 is an int.

    # 2) sigma should theoretically be >=0; if a parameter sweep reaches a negative value, take the absolute value to avoid anomalous results.
    sigma = abs(sigma) # Absolute value. A noise standard deviation should not be negative; if the parameter sweep reaches a negative value, treat it as positive to avoid anomalous results.

    # 3) Protect the denominator against zero or non-finite values.
    denom_sq = (0.25 + (sigma / amp_s) ** 2) * (0.24 + (sigma / amp_d) ** 2)
    if denom_sq <= 0:
        return 0.0

    denominator = sqrt(denom_sq)
    if not np.isfinite(denominator) or denominator == 0: # Square root plus a check that the result is finite.
        return 0.0

    # 4) Original piecewise formula: keep it unchanged and only apply a finite-value guard at the end.
    if phi <= 0.4:
        corr_IsId = (0.2 - phi) / denominator
    elif phi <= 0.5:
        corr_IsId = -0.2 / denominator
    elif phi <= 0.9:
        corr_IsId = (phi - 0.7) / denominator
    else:
        corr_IsId = 0.2 / denominator

    if not np.isfinite(corr_IsId):
        return 0.0
    return abs(corr_IsId)   # Take the absolute value at the end.
#f function
f = lambda x: 1/(1 + exp((Ed-x)/Dd))
# Functions for calculating correlation coefficients.
def corr_abs(sig1, sig2):
    """abs(corrcoef) but safe: std=0 or empty -> 0 (只在退化情况避免 NaN)"""
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    # np.asarray(...) converts lists, tuples, and NumPy arrays into a common NumPy-array representation.
    # This enables subsequent NumPy operations such as .size, np.std, and np.corrcoef.
    if sig1.size < 2 or sig2.size < 2:
        return 0.0
    if np.std(sig1) == 0 or np.std(sig2) == 0:
       return 0.0
    # 1. Pearson correlation requires at least two data points. 2. np.std(x) is the standard deviation; if it is zero, all values in x are identical and the denominator of the correlation formula becomes zero.
    c = np.corrcoef(sig1, sig2)[0, 1]
    # np.corrcoef(sig1, sig2) returns a 2×2 correlation matrix; [0,1] selects row 0, column 1, which is ρ(sig1,sig2).
    if np.isnan(c): #avoid nan data here
        return 0.0
    return abs(c)

def debugprint(spike, i1, i2, Convolution_K_S, timeInterval1, timeInterval2):
    print(f"spike={spike}, i1={i1}, i2={i2}, Convolution_K_S={Convolution_K_S}, timeInterval1={timeInterval1}, timeInterval2={timeInterval2}")


# Core code: first run the model to generate the signals Vs/Vd/S/E/B/eventRate/burstFraction/Is/Id, then call corr_abs to calculate correlations between different quantities.
# Finally combine the results into Voltage_corr (Vs and Vd), Corr_1 (ER and Is), Corr_2 (BR and Id), and Multiplex_metric.
def get_correlation(args):
    phi, sigma, Id0, Is0, Id_base, Is_base, gs, cd, gd, gc, p = args

    # dendritic square wave with phase shift
    Id = np.zeros(N+1) # Initialize Id as an empty-valued array, then fill it with different values by position.
    for i in range(N+1):
        k = ((i - td0/dt*phi) % (td0/dt)) / (td0/dt) # First convert phi to a time offset, operate on it together with i, and then convert the overall expression into time-step units.
        if k <= low_ratio_d:
            Id[i] = Id_base
        else:
            Id[i] = Id0 # During the low-level phase, assign the baseline; otherwise assign the high input. This forms a square-wave input with period td0. If phi=0, the square wave starts at t=0; if phi=0.5, it starts at t=td0/2.

    # somatic square wave
    Is = np.zeros(N+1)
    for i in range(N+1):
        k = (i % (ts0/dt)) / (ts0/dt)# Recent position of i within the period; ts0/dt is the number of time steps in one period.
        if k <= low_ratio_s:
            Is[i] = Is_base
        else:
            Is[i] = Is0

    # Statistics: counting arrays for each 1 ms interval over the duration time. They are not reset at the start of each neuron. Unit: Hz = s^-1.
    fireRate = np.zeros(time)        # spike rate
    eventRate = np.zeros(time)       # single spike + burst
    burstRate = np.zeros(time)
    burstFraction = np.zeros(time)   # burst probability

    Voltage_corr = 0.0

    for counter in range(neuronNumber):# Counter. Because noise generation is inside the neuron loop, each simulated neuron regenerates two noise sequences: noise1 is added to the dendrite and noise2 to the soma.
        # Set up the noise.
        np.random.seed()
        z1, z2 = np.random.normal(0, sigma*sqrt(2*alpha), (2, N)) # Generate two independent Gaussian random-number sequences at once, with mean 0 and standard deviation σ√(2α).

        noise1 = np.zeros(N+1)
        noise2 = np.zeros(N+1)
        for i in range(N):
            noise1[i+1] = noise1[i] - alpha*noise1[i]*dt + z1[i]*sqrt(dt) # Noise increment term: dX_t = -X_t dt + dW_t, β=σ√(2α); next-step noise = decay of current noise + random perturbation.
            noise2[i+1] = noise2[i] - alpha*noise2[i]*dt + z2[i]*sqrt(dt)
            # noise[n+1] = noise[n] - alpha*noise[n]*dt + sigma*sqrt(2*alpha)*sqrt(dt)*xi[n]   # xi[n] ~ N(0,1)
            # sigma is the stationary standard deviation of the OU noise, alpha is the OU-noise decay rate, dt is the time step, and z1[i] is a standard normal random number. This is the Euler-Maruyama discretization of OU noise.
        # Euler method
        Vs = np.zeros(N+1) # Create the Vs and Vd time-series arrays, each with length N+1 because the indices run from 0 through N.
        Vd = np.zeros(N+1)
        Vs[0] = EL# Initial potential is the resting potential.
        Vd[0] = EL

        ws = np.zeros(N+1) # Create the somatic-adaptation and dendritic-adaptation time-series arrays, each with length N+1.
        wd = np.zeros(N+1)

        S = [] # Record only the spikes generated by the current neuron; 10,000 neurons would generate 10,000 independent spike trains.
        spikeNumber = 0 # Continuously increases.

        # convolution replacement
        i1 = 0 # Only increases and usually eventually catches up with spikeNumber.
        i2 = 0
        Convolution_K_S = 0 # Number of spikes occurring within the interval (t-2.5, t-0.5].
        timeInterval1 = int(round(2.5/dt)) # Convert to time-step counts.
        timeInterval2 = int(round(0.5/dt))

        print("timeInterval1=", timeInterval1)
        print("timeInterval2=", timeInterval2)

        for i in range(1, N+1):
            # dendrite
            Vd[i] = (
                -(Vd[i-1]-EL)/tau_d*dt
                + (1/Cd) * (
                    gd*f(Vd[i-1])
                    + (gc/(1-p))*(Vs[i-1]-Vd[i-1])     # ✅ No if statement is needed: when gc=0, this term naturally becomes 0.
                    + Id[i-1]
                    + noise1[i-1]
                    - wd[i-1]
                    + cd*Convolution_K_S
                ) * dt
                + Vd[i-1]
            )
            wd[i] = (-wd[i-1] + a*(Vd[i-1]-EL))/tau_wd*dt + wd[i-1]

            # soma + spike
            if Vs[i-1] >= VT:
                Vs[i] = Vr
                S.append(t[i-1])
                spikeNumber += 1
                ws[i] = (-ws[i-1]/tau_ws)*dt + b + ws[i-1]
            else:
                Vs[i] = (
                    -(Vs[i-1]-EL)/tau_s*dt
                    + (1/Cs) * (
                        gs*f(Vd[i-1])
                        + (gc/p)*(Vd[i-1]-Vs[i-1])      # Same as above: when gc=0, this term naturally vanishes.
                        + Is[i-1]
                        + noise2[i-1]
                        - ws[i-1]
                    ) * dt
                    + Vs[i-1]
                )
                ws[i] = (-ws[i-1]/tau_ws)*dt + ws[i-1]

            # convolution counter update
# After a spike occurs, these counters decrease by 1 at each subsequent step: 5 → 4 → 3 → 2 → 1 → 0 (then +1 at the boundary); 25 → 24 → … → 0 (then -1 at the boundary). After reaching a boundary they are reset to the number of steps remaining until the corresponding boundary of the next spike, so they act as dynamically updated countdown timers.
            if i1 != spikeNumber: # There are still unprocessed window-closing events.
                timeInterval1 -= 1 # Decrease the countdown by 1 for each dt step.
                if timeInterval1 == 0:
                    Convolution_K_S -= 1 # One spike's influence window has ended, so the number of currently active windows decreases by 1. The convolution contribution does not necessarily become zero.
                    i1 += 1
                    if i1 == spikeNumber:
                        timeInterval1 = int(round(2.5/dt))
                    else:# Return to the original condition if i1 != spikeNumber. This occurs just after one window-closing event has been processed, while preparing to process the next spike's window closing.
                        timeInterval1 = int(round((S[i1] - (t[i]-2.5)) / dt))
                        if timeInterval1 <= 0: # Prevent invalid countdown values (zero or negative) from breaking the event-trigger logic.
                            timeInterval1 = 1

                if i2 != spikeNumber:
                    timeInterval2 -= 1
                    if timeInterval2 == 0:
                        Convolution_K_S += 1
                        i2 += 1
                        if i2 == spikeNumber:
                            timeInterval2 = int(round(0.5/dt))
                        else:
                            timeInterval2 = int(round((S[i2] - (t[i]-0.5)) / dt))
                            if timeInterval2 <= 0:
                                timeInterval2 = 1
            
            debugprint(spikeNumber, i1, i2, Convolution_K_S, timeInterval1, timeInterval2)

        # voltage correlation
        Voltage_corr += corr_abs(Vs[1000:-1000], Vd[1000:-1000]) # Remove the first and last 1000 steps, leaving the middle 500 ms (dt=0.1 ms, so 1000 steps=100 ms).

        # Burst + event detection (original logic).
        B = []
        E = []
        i = 0
        if spikeNumber >= 2:
            while i < (spikeNumber - 1): # i is the spike index; continue looping as long as there is a following spike.
                s = S[i] # Time of the current spike.
                E.append(s) # Add the spike at time s to E (the event list).
                if (S[i+1] - s) <= 15: # If the next spike occurs within 15 ms, classify it as a burst.
                    B.append(s)
                    i = i + 2 # Continue searching through subsequent spikes in this burst until the burst ends.
                    while i < (spikeNumber - 1): 
                        if (S[i] - s) > 15: # If the next spike is more than 15 ms away, treat the burst as ended and break out of the while loop.
                            break
                        i = i + 1
                    continue # Skip the remaining code below and return to the outermost while loop.
                i = i + 1

        # Place the events into 1 ms bins. After each neuron is simulated, add the points in S, E, and B to these arrays; within each 1 ms window, fireRate/eventRate/burstRate contain the total spike/event/burst counts across the neuron population.
        for point in S: # For example, if S=[0.1, 0.2, 1.3, 1.4, 1.5], then fireRate[0]=2, fireRate[1]=3, and the other bins are 0.
            idx = int(point)
            if 0 <= idx < time:
                fireRate[idx] += 1
        for point in E:
            idx = int(point)
            if 0 <= idx < time:
                eventRate[idx] += 1
        for point in B:
            idx = int(point)
            if 0 <= idx < time:
                burstRate[idx] += 1

    for i in range(time):
        if eventRate[i] != 0:
            burstFraction[i] = burstRate[i] / eventRate[i] # BF does not need to be divided by the number of neurons again because the numerator and denominator come from the same neuron population. Dividing both by 10,000 would leave the ratio unchanged.

    Corr_1 = corr_abs(eventRate[100:-100], Is[1005:-1000:10])
    Corr_2 = corr_abs(burstFraction[100:-100], Id[1005:-1000:10])
    Multiplex_metric = (Corr_1 + Corr_2) / 2.0
    Voltage_corr = abs(Voltage_corr / neuronNumber)

    return Voltage_corr, Corr_1, Corr_2, Multiplex_metric


# Run the model and generate the required data, following the args/variable structure from FirstDepict.
def ModelProcess_Generate_Data(index, args, variable):
    Voltage_corr = np.zeros(pointNum)
    Corr_1 = np.zeros(pointNum)
    Corr_2 = np.zeros(pointNum)
    Multiplex_metric = np.zeros(pointNum)

    process_pool = Pool()
    Args = []
    for i in range(pointNum):
        args[index] = variable[i]
        Args.append(args.copy())


    print("Args=", Args)
    results = process_pool.map(get_correlation, Args)
    process_pool.close()
    process_pool.join()

    for i, result in enumerate(results):
        Voltage_corr[i], Corr_1[i], Corr_2[i], Multiplex_metric[i] = result

    # Save data to a dictionary (original style).
    dataframe = {}
    dataframe['index'] = index
    dataframe['variable_Name'] = labels_short[index]
    dataframe['variable'] = variable
    dataframe['Vs_Vd'] = Voltage_corr
    dataframe['EventRate_Is'] = Corr_1
    dataframe['BurstFraction_Id'] = Corr_2
    dataframe['Multiplex_metric'] = Multiplex_metric

    # If the independent variable is one of the input parameters (index<6), additionally calculate the theoretical Is-Id correlation.
    if index < 6:
        IsId_corr = np.zeros(len(variable))
        for i in range(pointNum):
            args[index] = variable[i]
            IsId_corr[i] = Correlation_IsId(args[:6])
        dataframe['Is_Id'] = IsId_corr

    return dataframe


# Save data to files and generate/save plots (original logic: save figures + save pkl).
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

def DataSave_LineGraph(path, dataframe, describtion):
    short_describtion = dataframe['variable_Name'] + '_' + str(neuronNumber) + 'cells'

    variable = dataframe['variable']
    index = dataframe['index']

    # First figure: multiplexing metric + two correlation terms.
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.plot(variable, dataframe['Multiplex_metric'], color='blue', label='Metric of Multiplexing', marker='s')
    axs.plot(variable, dataframe['EventRate_Is'], color='black', label=r'$\rho$' + '(Event Rate, Is)', linestyle=':')
    axs.plot(variable, dataframe['BurstFraction_Id'], color='brown', label=r'$\rho$' + '(Burst Fraction, Id)', linestyle='--')
    axs.legend()
    axs.set_xlabel(labels[index], font)

    title = describtion + ' neuronNumber=' + str(neuronNumber)
    fig.suptitle(title)
    fig.savefig(path + '/image_' + short_describtion + '多路复用.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Second figure: voltage correlation + multiplexing.
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    axs[0].plot(variable, dataframe['Vs_Vd'], color='red', label=r'$\rho$' + '(Vs, Vd)', marker='o')
    axs[0].plot(variable, dataframe['Multiplex_metric'], color='blue', label='Metric of Multiplexing', marker='s')
    axs[0].legend()
    axs[0].set_xlabel(labels[index], font)

    axs[1].scatter(dataframe['Vs_Vd'], dataframe['Multiplex_metric'], marker='^', s=40, color='black')
    axs[1].set_xlabel(r'$\rho$' + '(Vs, Vd)', font)
    axs[1].set_ylabel('Metric of Multiplexing', font)

    fig.suptitle(title)
    fig.savefig(path + '/image_' + short_describtion + '多路复用与电压相关系数.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Third figure: if Is_Id is available, add the input-correlation plot.
    if 'Is_Id' in dataframe:
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))
        axs[0].plot(variable, dataframe['Is_Id'], color='darkgreen', label=r'$\rho$' + '(Is, Id)', marker='p')
        axs[0].plot(variable, dataframe['Vs_Vd'], color='red', label=r'$\rho$' + '(Vs, Vd)', marker='o')
        axs[0].plot(variable, dataframe['Multiplex_metric'], color='blue', label='Metric of Multiplexing', marker='s')
        axs[0].legend()
        axs[0].set_xlabel(labels[index], font)

        axs[1].scatter(dataframe['Is_Id'], dataframe['Multiplex_metric'], marker='^', s=40, color='black')
        axs[1].set_xlabel(r'$\rho$' + '(Is, Id)', font)
        axs[1].set_ylabel('Metric of Multiplexing', font)

        fig.suptitle(title)
        fig.savefig(path + '/image_' + short_describtion + '多路复用与input相关系数.png', dpi=300, bbox_inches='tight')
        plt.show()

    dataframe['neuronNumber'] = neuronNumber
    dataframe['pointNum'] = pointNum

    # Save to pkl.
    with open(path + '/数据_' + short_describtion + '.pkl', 'wb') as file:
        pickle.dump(dataframe, file)


# Main function.
if __name__=='__main__':
    out_path = '多路复用度量与电压相关系数'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    index, args, variable, describtion = FirstDepict(parameters, labels_short, pointNum)
    if index is None:
        raise ValueError("你需要把 11 个参数中的某一个写成 [min,max]，其余写成单值。")

    # Print the fixed parameters, excluding the independent-variable entry.
    s1 = labels_short.copy()
    s2 = parameters.copy()
    s1.pop(index)
    s2.pop(index)

    print('\n参数:')
    print('-'*110, '\n [{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}]'.format(*s1))
    print('=[{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}]'.format(*s2))
    print('-'*110, f"\n自变量选择：{labels_short[index]}, 范围：[{variable[0]}, {variable[-1]}], 点数：{pointNum}\n")

    start = datetime.now()
    print("开始:", start)

    dataframe = ModelProcess_Generate_Data(index, args, variable)
    DataSave_LineGraph(out_path, dataframe, describtion)

    end = datetime.now()
    print("结束:", end)
    print("耗时:", end - start)