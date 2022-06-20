import functools
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import find_peaks

# arr: trial of magn of shank ACC
# events: idx of peaks of arr (only looking for pairs of peaks)
def gen_gait_cycle_events(arr, events, viz=False):
    gait_cycles_ev=[] # arr of tuples(len:2) denoting beginning and end of gait cycles
    last_g_ev=(0,0)
    for i in range(0,len(events)):
        if (i-1>=0 and i+1 < len(events) and 
            (arr[events[i]] < arr[events[i+1]] and 
             arr[events[i]] < arr[events[i-1]] and
             arr[events[i-1]] >= arr[events[i+1]]
            )):
            g_ev = events[i]
            if last_g_ev!=(0,0) and i-last_g_ev[0]==3:
                gait_cycles_ev.append((last_g_ev[1],g_ev))
            last_g_ev = (i,g_ev)
                                        
    return gait_cycles_ev

magn = lambda arr: np.sqrt(np.sum(np.square(arr)))

def gen_gait_cycles(trial, viz=False):
    magn_trial = np.apply_along_axis(magn, 1,trial[:,3,0:3]) # get [full length of trial,shankL,Acc_X:Acc_Z]
    peaks, _ = find_peaks(magn_trial, prominence=0.4)
    
    if viz:
        print('ACC Magn Trial with peaks')
        plt.plot(magn_trial);plt.plot(peaks, magn_trial[peaks], "xr")
        plt.show()

    g_events = gen_gait_cycle_events(magn_trial, peaks)
    
    if viz:
        print('ACC Magn Trial with gait cycle')
        plt.plot(magn_trial)
        plt.plot([x[0] for x in g_events], magn_trial[[x[0] for x in g_events]], "xr")
        plt.plot([x[1] for x in g_events], magn_trial[[x[1] for x in g_events]], "ob")
        plt.show()
    
    return g_events #array of tuple of gait cycle event idx (begin,end)

def normalized_gait(A):
	x=np.arange(A.shape[-1])
	A=np.where(np.isnan(A)==1,0,A)
	Y= interpolate.interp1d(x,A, kind='cubic')(np.linspace(x.min(), x.max(), 101))
	return Y

# entry: entry from retrieved dataset; shape: (patientNum, feature_trial, label)
# out: array of entry with gait cycles for trials
#    ; shape: [(patientNum, feature_gait_cycle_1, label), ... , (patientNum, feature_gait_cycle_N, label)]
def trial2gcycles(entry,viz=False,i=None):
    if i is not None:
        print('tria2gcycles for i:'+str(i))
    gc_entries = []
    
    g_events = gen_gait_cycles(entry[1],viz)
    
    for b_i, e_i in g_events:
        new_entry = np.copy(entry)
        un_norm_gc = entry[1][b_i:e_i,:,:] # replace trial with normalized gait cycle
        
        norm_gc = np.apply_along_axis(normalized_gait, 0,un_norm_gc) # (101,5,6)
        
        new_entry[1] = norm_gc
        gc_entries.append(new_entry)
    
    return gc_entries

tmp_len = 0
progress = lambda tmp_len,total_len:print('Progress: ['+'='*int(tmp_len/total_len*50)+'>'+'-'*int((total_len-tmp_len)/total_len*50)+'] '+str(int(tmp_len/total_len*100))+'%'+' '*10, end='\r')

def join(ar1,ar2, total_len):
    arr = ar1
    
    for e in ar2:
        arr.append(e)

    global tmp_len

    if tmp_len==0:
        tmp_len=2
    else:
        tmp_len+=1
    
    progress(tmp_len,total_len)

    return arr


if __name__ == '__main__':
    gois_dataset = np.load('../GoIS_dataset.npy', allow_pickle=True)
    
    print('Loaded Gait on Irregular Surface(GoIS) dataset')
    print('='*25)
    print('Applying trial2gcycles on each entry')
    print('trial2gcycles:')
    print('-find start of gait cycles in trial')
    print('-split gait cycles')
    print('-normalize gait cycles to 101 x 5 x 6')

    tmp_l = 0

    t2g_p = lambda x: progress();

    total_l = len(gois_dataset)

    def t2g_p(x):
        global tmp_l
        tmp_l+=1
        progress(tmp_l,total_l)

        return trial2gcycles(x)

    ngois_d = list(map(t2g_p,gois_dataset))

    print('')
    print('='*25)
    print('Folding array of array of entries => into => an array of normalized gait entries')
    
    ngois_dataset = functools.reduce(lambda x,y: join(x,y, len(ngois_d)), ngois_d)

    print('')
    print('='*25)
    print('normalized Gait on Irregular Surface dataset is ready! Saving it...!')
    
    np.save('../nGoIS_dataset', ngois_dataset)