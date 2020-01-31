# THIS CODE WILL DETERMINE WHETHER THE TRIAL HAS A PREY CAPTURE LIKE KINEMATICS
# BASED ON THE ONSET OF THE CONTRA-LATERAL EYE CONVERGENCE, TAIL KINEMATICS
# The onset is defined as the point between the min and max peaks of eye velocities
# and the start of the tail bout.
# UPDATES: 27 JUNE 2019, Ivan
#          31 JANUARY 2020, UPDATES FOR MACX DATA, Ivan
# COMPATIBLE ONLY FOR PYTHON 2.7

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter

from csvfiles_reader import *
from extract_taileyebouts import *
from eye_movement_filters import *
from get_preylocation import *
import operator
from itertools import izip_longest
import seaborn as sns
from detect_preycapture import *

plt.style.use(['dark_background'])

# ============================SET PARAMETERS=======================================================

# ---------------------VIDEO RECORDING PARAMETERS------------------------------------------------
Fs = 300 # Sampling frequency used for the video
tfactor = Fs/1000.0  # convert fps to fpms
# ---------------------PARAMETERS FOR EXTRACTING THE BOUTS------------------------------------------------
bout_thresh = 0.40  # 0.00 - 1.00 threshold value for extracting bouts, higher more bouts
peakthres = 4  # 0.00 - 20.00 lower vale more peaks, for calculating tail beat frequency
# ---------------------PARAMETERS FOR DETECTING THE PREY CAPTURE BOUTS, MINIMUM VALUES-----------------------------------
filter_avg_tail = 5.0 # threshold for average tail bout angle
filter_avg_binocular = 17.0 # threshold for eye angle binocular convergence, unit in degrees
filter_length_bout = 0 # length of bout based on frame number, unit in frame number
filter_eye_vel = -0.1 # diverging eye velocity in degrees/ms, unit in deg/ms
filter_eye_diverge = -1.0 # eye divergence by degrees, unit in deg
thresh_saccade_speed = 0.2 # the onset of eye movement should be greater than saccade threshold, unit in deg/ms
# ---------------------PARAMETERS FOR EXTRACTING PRE AND POST SACCADIC BOUTS------------------------------------------------
pre_post_saccade_win = [0.5,0.5] # pre and post saccadic threshold in seconds
# ---------------------VISUAL PREY PARAMETERS------------------------------------------------
preypoints = [10, 70] # direction of prey in visual angle, e.g. [10,70] means going from 10deg to 70 deg
sensory_delay = 0.1 # how many frames to consider for sensory delay in seconds
sensory_delay = int(Fs*float(sensory_delay)) # convert in # frames
delay = 0.2 # time window to the tail onset for finding the minimum eye onset in seconds
delay = int(Fs*float(delay)) # convert in # frames
speed = 40.0 # prey speed in degrees per second
padding_nonstimulus = [0, 0] # to add the frames as a waiting time to determine prey location, visual stimulus appears usually 3 s after trial
# ---------------------LOW PASS FILTER------------------------------------------------
order = 3
fs = Fs  # sample rate, Hz
cutoff = 10  # desired cutoff frequency of the filter, Hz

# ============================ GENERATE ALL THE FILES =================================================================
# THIS PART IS ONLY FOR COMPILING ALL THE INPUT FILES. THIS IS SUBJECTIVE, IT DEPENDS ON HOW YOU ORGANIZE YOUR DATA
# I organize it this way home/experiment/fish/data (eye, tail), This part will combine all the experiments I have
# Directories
maindir = 'E:\\Macx\\PC analysis\\'
dir_output = maindir

expts = list()
expts += [xp for xp in os.listdir(maindir) if ".csv" not in xp]  # store the fish filenames

# dir = 'C:\\Users\\Semmelhack Lab\\Desktop\\analysis\\'

eye_files = [] # DECLARE A LIST TO STORE THE EYE ANGLE FILES
tail_files = []

fishIDs = list() # LIST FOR STORING EACH FISH NAME
fishIDs_dir = list() # LIST FOR SOME INFO FOR EACH FISH

for exp in expts:
    fishIDs += [exp+ '\\' + fish for fish in os.listdir(maindir + exp + '\\') if ".csv" not in fish]  # store the fish filenames
    # STORE [BIG PREY OR LEFT PREY FOR EACH FISH, RESPONSE TIME OF THE PREY CAPTURE BOUT]
    fishIDs_dir += [{str(fish): [0, 0], str(fish) +'_bigorsmall': [], str(fish) +'_resptime': [] } for fish in fishIDs]

for fish in fishIDs: # READ EACH FISH EYE AND TAIL ANGLES

    eyefile = maindir + fish + '\\eye_angle\\'
    tailfile = maindir + fish

    for file in os.listdir(eyefile):
        if file.endswith(".csv"):
            # store the [file directory, directory output, prey direction, fish name, prey speed, filename of eye angles]
            eye_files.append([os.path.join(eyefile, file), dir_output, preypoints, fish, speed, file])
    for file in os.listdir(tailfile):
        if file.endswith(".shv"):
            tail_files.append([os.path.join(tailfile, file), dir_output, preypoints, fish, speed, file])

print(len(eye_files), len(tail_files))
print(len(fishIDs))

# ============================MACRO FOR PREY SELECTION ANALYSIS=======================================================

eye_diffs = []
tails = []
rtime = []
one_bout = 0
multi_bout = 0
nbouts = 0
eye_onsets = []
fish_movements = []
right_prey = 0
left_prey = 0
PC = 0
fish_PC = []
response_time = []
duration = []
prey_size = []
sample_fish = []
pre_saccade_left = []
post_saccade_left = []

pre_saccade_right = []
post_saccade_right = []

pc_onset = []
avg_tail = []
for j in range(0, len(eye_files)):  # READ EACH TRIAL

    # Display which file is currently being processed
    print('PROCESSING: ')
    trial = eye_files[j][5].split('_')[0]
    trial_name = trial.split('-')[0] + '-' + trial.split('-')[1] + '-' + trial.split('-')[3]
    print(trial_name)
    if 'Ctrl' in eye_files[j][0]:
        # SKIP THE CONTROL DATA
        continue

    dir_output = str(eye_files[j][1])  # output for saving some info

    # 0e ---------------------------------------------------------------------------------------

    # smaller = None
    # preysize = int(eye_files[j][5][eye_files[j][5].find("@") + -1])

    # 1s -----------------read the eye and tail angles------------------------------------------

    eyes = eye_reader(str(eye_files[j][0]))
    eyes[0]['LeftEye'] = butter_freq_filter(eyes[0]['LeftEye'], cutoff, fs, order)  # low pass filter
    eyes[0]['RightEye'] = butter_freq_filter(eyes[0]['RightEye'], cutoff, fs, order)
    # tailangle = tail_reader(str(tail_files[j][0]))

    # 1e ----------------------------------------------------------------------------------------

    # 2s -----------------compute the velocity------------------------------------------

    LeftVel = savgol_filter(eyes[0]['LeftEye'], 3, 2, 1, mode='nearest')
    RightVel = savgol_filter(eyes[0]['RightEye'], 3, 2, 1, mode='nearest')
    eyes_vel = [{'LeftVel': LeftVel, 'RightVel': RightVel}]

    # 2e -------------------------------------------------------------------------------

    # 3s ----------------- DETECT THE TAIL BOUTS AND CORRESPONDING EYE MOVEMENTS -------------------------------

    TailEye = extract_tail_eye_bout(eyes, eyes_vel, tail_files[j][0], Fs, bout_thresh, peakthres,
                                    delay)  # extract the tail bouts

    if not TailEye:  # if there's no bout detected
        print('WARNING!! TailEye empty: ', eye_files[j])
        continue

    if len(TailEye) == 1:  # determine whether it is a single bout trial or a multi-bout
        one_bout += 1
    else:
        multi_bout += 1

    # 3e --------------------------------------------------------------------------------------------------

    # 4s --------------------------------- SET TIME FOR EACH TRIAL TO seconds --------------------------------------------
    time = range(0, len(TailEye[0]['left']))
    time = [t / tfactor for t in time]
    # 4e ---------------------------------------------------------------------------------

    first_pc_bout = 1  # an ID for first prey capture bout
    pc_per_trial = []
    resp_time_per_trial = []

    for i in range(0, len(TailEye)):  # READ THROUGH EACH BOUT
        print('============= BOUT #', i, 'of ', TailEye[i]['filename'], '==================')
        t1 = TailEye[i]['frames'][0]
        t2 = TailEye[i]['frames'][1]

        if first_pc_bout == 0:  # Check whether 1st prey capture bout has already been found
            print("First prey capture bout already found")
            continue

        # 5s ---------------------- DETECT PREY CAPTURE -------------------------------------------------------------
        bout_candid = {'frames': TailEye[i]['frames'], 'bout_angles': TailEye[i]['bout_angles'],
                       'sum_eyeangles': TailEye[i]['sum_eyeangles'],
                       'right_eyeangles': TailEye[i]['right_eyeangles'], 'left_eyeangles': TailEye[i]['left_eyeangles']}

        PC_threshold = {'sensory_delay': sensory_delay, 'bout_angles': filter_avg_tail,
                        'bout_length': filter_length_bout,
                        'sum_eyeangles': filter_avg_binocular, 'eye_vel': filter_eye_vel,
                        'eye_mag': filter_eye_diverge, 'tail_meanangle': 0.0}

        isPC = ispreycap(bout_candid, PC_threshold, tfactor)
        isPC = all(1 == pc for pc in
                   isPC.values())  # check whether this bout satisfied all the filters set on the PC_threshold

        if not isPC:
            # if isPC is false, this bout is not prey capture, move to next one
            continue
        # 5e ---------------------------------------------------------------------------------------------------------

        print("DETECTED PREY CAPTURE BOUT")
        print(eye_files[j][0])
        print(tail_files[j][0])
        print('Tail bout', TailEye[i]['frames'])
        print('Mean binocular angle', np.mean(TailEye[i]['sum_eyeangles']))
        print('TAIL BOUT FREQ', TailEye[i]['tailfreq'])

        if i == 0:
            rtime.append((TailEye[0]['frames'][0] / tfactor))

        tail = np.mean(TailEye[i]['bout_angles'])

        # get the index, and value of the saccade onset based on velocity
        # Get the velocity maximum peak
        print(TailEye[i]['frames'][0], TailEye[i]['frames'][1])
        r_maxpeak, r_max = max(enumerate(TailEye[i]['right_vel_delay']), key=operator.itemgetter(1))
        l_maxpeak, l_max = max(enumerate(TailEye[i]['left_vel_delay']), key=operator.itemgetter(1))

        # Get the minimum peak between the start of the bout to the peak velocity
        # You only want the minimum peak BEFORE the maximum peak
        # There are cases when the maximum peak is found within the first two points
        # one reason is the starting point of the detected tail bout is greatly delayed compared to
        # its corresponding eye movement

        if r_maxpeak == 0:
            r_min = TailEye[i]['right_vel_delay'][0]
            r_minpeak = 0
        else:
            r_minpeak, r_min = min(enumerate(TailEye[i]['right_vel_delay'][0:r_maxpeak]), key=operator.itemgetter(1))

        if l_maxpeak == 0:
            l_min = TailEye[i]['left_vel_delay'][0]
            l_minpeak = 0
        else:
            l_minpeak, l_min = min(enumerate(TailEye[i]['left_vel_delay'][0:l_maxpeak]), key=operator.itemgetter(1))

        r_sac_on = TailEye[i]['frames'][0] - delay + int(math.ceil((r_maxpeak + r_minpeak)/2))
        l_sac_on = TailEye[i]['frames'][0] - delay + int(math.ceil((l_maxpeak + l_minpeak)/2))

        onset = min(r_sac_on, l_sac_on)
        #if (TailEye[i]['frames'][0] - padding_nonstimulus[0]) < 0:
        #    print("NO STIMULUS YET")
        #    continue

        if first_pc_bout == 1:  # first bout
            PC += 1  # is there a prey capture in this trial?
            fish_PC.append(trial_name)
            sample_fish.append(eye_files[j][3])

            pre_saccade_left.append(eyes[0]['LeftEye'][l_sac_on - int((pre_post_saccade_win[0] * float(Fs)))])
            post_saccade_left.append(eyes[0]['LeftEye'][l_sac_on + int((pre_post_saccade_win[1] * float(Fs)))])

            pre_saccade_right.append(eyes[0]['RightEye'][r_sac_on - int((pre_post_saccade_win[0] * float(Fs)))])
            post_saccade_right.append(eyes[0]['RightEye'][r_sac_on + int((pre_post_saccade_win[1] * float(Fs)))])

            response_time.append((TailEye[i]['frames'][0] - padding_nonstimulus[0]) / tfactor)
            duration.append(TailEye[i]['frames'])

            pc_onset.append(onset)
            avg_tail.append(tail)
            first_pc_bout = 0  # set to 0 to notify that first PC bout is found

    print(TailEye[0]['filename'])
    print('done')

print('FISH WITH PC', set(sample_fish))
print('# OF FISH WITH PC', len(set(sample_fish)))
print('TOTAL # OF PREY CAPTURES', PC)
# print(np.mean(response_time), np.std(response_time))
results = izip_longest(fish_PC, pc_onset, duration, avg_tail, pre_saccade_left, post_saccade_left, pre_saccade_right, post_saccade_right)
header = izip_longest(['Fish'], ['PC Onset'], ['Bout duration'], ['Mean Tail'], ['Pre-saccade left'], ['Post-saccade left'], ['Pre-saccade right'], ['Post-saccade right'])

with open(maindir + "Summary" + '.csv', 'wb') as myFile:
    # with open(dir_output + 'Velocity_Acceleration_' + filename + '.csv', 'wb') as myFile:
    wr = csv.writer(myFile, delimiter=',')
    for head in header:
        wr.writerow(head)
    for rows in results:
        wr.writerow(rows)
