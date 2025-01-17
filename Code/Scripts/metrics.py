import numpy as np
from statistics import median

#def single function for general case of n situations
def PerturberSample(sample_num, y_test, y_pred_wop, y_pred_wp, y_pred_wpl, std_pred_wop, std_pred_wp, std_pred_wpl):
    """ Calculuate the median error, median absolute error, and the median standard deviation for the Wop, WP, and WPL cases.

    Parameters
    ----------
    sample_num : int
        Number of mock sample lenses.
    y_test : np.array
        Known values of the lens parameters.
    y_pred_wop, y_pred_wp, y_pred_wpl : np.array
        Predicted values of the lens parameters for the wop, wp, and wpl cases.
    std_pred_wop, std_pred_wp, std_pred_wpl : np.array
        Standard deviations of the lens parameter predicted distributions.

    Returns
    -------

    """

    ##ME - Median Error
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wop[i]-y_test[i])
    #print(param_diff)
    sum=0
    for i in param_diff:
        #print(i)
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wop = list(np.round(mean,2))
    #print(ME_wop)

    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wp[i]-y_test[i])
    sum=0
    for i in param_diff:
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wp = list(np.round(mean,2))
    #print(ME_wp)

    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wpl[i]-y_test[i])
    sum=0
    for i in param_diff:
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wpl = list(np.round(mean,2))
    #print(ME_wpl)

    ##MAE - Median Absolute Error
    param_diff_abs=[]
    for i in range(sample_num-1):
        param_diff_abs.append(abs(y_pred_wop[i]-y_test[i]))
    #print(param_diff_abs)
    sum=0
    for i in param_diff_abs:
        sum=sum+i
    mean=(sum/(sample_num-1))
    MAE_wop = list(np.round(mean,2))
    #print(MAE_wop)

    param_diff_abs=[]
    for i in range(sample_num-1):
        param_diff_abs.append(abs(y_pred_wp[i]-y_test[i]))
    sum=0
    for i in param_diff_abs:
        sum=sum+i
    mean=(sum/(sample_num-1))
    MAE_wp = list(np.round(mean,2))
    #print(MAE_wp)

    param_diff_abs=[]
    for i in range(sample_num-1):
        param_diff_abs.append(abs(y_pred_wpl[i]-y_test[i]))
    sum=0
    for i in param_diff_abs:
        sum=sum+i
    mean=(sum/(sample_num-1))
    MAE_wpl = list(np.round(mean,2))
    #print(MAE_wpl)

    ##Msigma - Median Standard Deviation
    #print(std_pred_wop)
    sum=0
    for i in std_pred_wop:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wop = list(np.round(mean,2))
    #print(Mstd_wop)

    ##Msigma - Median Standard Deviation
    sum=0
    for i in std_pred_wp:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wp = list(np.round(mean,2))
    #print(Mstd_wp)

    ##Msigma - Median Standard Deviation
    sum=0
    for i in std_pred_wpl:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wpl = list(np.round(mean,2))
    #print(Mstd_wpl)

    metrics = [ME_wop, ME_wp, ME_wpl, MAE_wop, MAE_wp, MAE_wpl, Mstd_wop, Mstd_wp, Mstd_wpl]

    return metrics

def PerturberSampleTrunc(sample_num, param_num, y_test, y_pred_wop, y_pred_wp, y_pred_wpl, std_pred_wop, std_pred_wp, std_pred_wpl):

    #WoP
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wop[i]-y_test[i])
    #print(param_diff)
    
    ##ME - Mean Error
    sum=0
    for i in param_diff:
        #print(i)
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wop = list(np.round(mean,2))
    #print(ME_wop)
    
    ##MDE - Median Error
    #for i in range(param_num):
     #   param_i = []
    #for row in param_diff:
     #   for i in range(param_num):
      #      param_i.append(row[i])
    #for i in param_num: 
     #   print(param_i)
            
            
    #Median_wop = list(np.round(median(param_diff),2))

     #WP
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wp[i]-y_test[i])
    #print(param_diff)
    ##ME - Mean Error
    sum=0
    for i in param_diff:
        #print(i)
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wp = list(np.round(mean,2))
    
    #WPL
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_wpl[i]-y_test[i])
    ##ME - Mean Error
    sum=0
    for i in param_diff:
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_wpl = list(np.round(mean,2))
    #print(ME_wpl)
    ##MDE - Median Error
    #Median_wpl = list(np.round(median(param_diff),2))

    ##Msigma - Mean Standard Deviation
    sum=0
    for i in std_pred_wop:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wop = list(np.round(mean,2))
    #print(Mstd_wop)
    ##MDsigma - Median Standard Deviation
    #MDstd_wop = list(np.round(median(std_pred_wop),2))

    ##Msigma - Mean Standard Deviation
    sum=0
    for i in std_pred_wp:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wp = list(np.round(mean,2))
    #print(Mstd_wp)
    ##MDsigma - Median Standard Deviation
    #MDstd_wp = list(np.round(median(std_pred_wp),2))
    
    ##Msigma - Median Standard Deviation
    sum=0
    for i in std_pred_wpl:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_wpl = list(np.round(mean,2))
    #print(Mstd_wpl)
    ##MDsigma - Median Standard Deviation
    #MDstd_wpl = list(np.round(median(std_pred_wpl),2))

    mean_metrics = [ME_wop, ME_wp, ME_wpl, Mstd_wop, Mstd_wp, Mstd_wpl]
    #median_metrics = [Median_wop, Median_wpl, MDstd_wop, MDstd_wpl]

    return mean_metrics

def PerturberSampleTrunc_Substructure(sample_num, param_num, y_test, y_pred_WoS, y_pred_WS, std_pred_WoS, std_pred_WS):

    #WoS
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_WoS[i]-y_test[i])
    #print(param_diff)
    
    ##ME - Mean Error
    sum=0
    for i in param_diff:
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_WoS = list(np.round(mean,2))
    #print(ME_wop)

     #WS
    param_diff=[]
    for i in range(sample_num-1):
        param_diff.append(y_pred_WS[i]-y_test[i])
    #print(param_diff)
    
    ##ME - Mean Error
    sum=0
    for i in param_diff:
        sum=sum+i
    mean=(sum/(sample_num-1))
    ME_WS = list(np.round(mean,2))

    ##Msigma - Mean Standard Deviation
    sum=0
    for i in std_pred_WoS:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_WoS = list(np.round(mean,2))
    #print(Mstd_wop)

    ##Msigma - Mean Standard Deviation
    sum=0
    for i in std_pred_WS:
        sum=sum+i
    mean=(sum/(sample_num-1))
    Mstd_WS = list(np.round(mean,2))
    #print(Mstd_wp)

    mean_metrics = [ME_WoS, ME_WS, Mstd_WoS, Mstd_WS]
    #median_metrics = [Median_wop, Median_wpl, MDstd_wop, MDstd_wpl]

    return mean_metrics