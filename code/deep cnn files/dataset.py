
### gloabl one
import logging
import re
import numpy as np
import glob
import os
import os.path
import torch#for testing gpu 
import pandas as pd
import math
import mne



log = logging.getLogger(__name__)


def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)


def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id


def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)
    return file_paths     #RETURN TO STOP SORTING

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)

def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        if 'eval' in file_path:
            edf_file = mne.io.read_raw_edf(file_path, montage = None, eog = ['FP1', 'FP2', 'F3', 'F4',
                                                                             'C3', 'C4',  'P3', 'P4','O1', 'O2','F7', 'F8',
                                                                             'T3', 'T4', 'T5', 'T6','PZ','FZ', 'CZ','A1', 'A2'], verbose='error')
            
            print(edf_file)
        else:
            edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()
  

    return int(header[236:244].decode('ascii'))


def load_data(fname,typee, preproc_functions, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    log.info("Load data..."+fname)
    ##edit to get on gpu device
    torch.cuda.set_device(1)
    print("--------------------------------" + torch.cuda.get_device_name(0))

    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2',
                        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if ((' ' + wanted_part + '-' in ch_name) or (wanted_part == ch_name)):#if ' ' + wanted_part + '-' in ch_name:
                    wanted_found_name.append(ch_name)
            print(wanted_found_name)####Comment out
            assert len(wanted_found_name) == 1
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names)
    global matrix_label1
    global matrix_data1
    global matrix_f
    #assert np.array_equal(cnt.ch_names, selected_ch_names)
    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names)  == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # change from volt to mikrovolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    #print(fname)
    #print("****************************************************************")
    matrix_label=[]
    matrix_data=[]
    matrix_data2=[]
    f_edf=pd.DataFrame()
 
    if data.shape[1] < 120000:
        return None, None
    for fn in preproc_functions:
        log.info(fn)
        print(data.shape)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    filesedf=[]
    filesedf.append(fname)
    f_edf['files']=filesedf
   
    #df = pd.DataFrame(eval(fname))
    
    prt=fname.split('/')[-1]
    print('********************sp value')
    print(prt)
    x=['file']
    print(x)
    arr=np.array(prt)
    sp=pd.DataFrame(columns = x)
    sp.loc[0]=prt
    print(sp)

    print()
    sp1=sp.file.str.split(".",expand=True)
    path1='v2.0.0/Annotations'
    sp2=sp1[0].astype('int') # numbers like csv files names
    k='.csv'
    h=sp2.astype('str')
    csvfiles=h+k

    for file2 in os.listdir(path1):  
      chk2=1
      if csvfiles[0] in file2:
        f2=os.path.join(path1,file2)
       # print('csv file found')
        chk2=2
        file_found=1
        break 
      if chk2==1: 
        #print('csv file not found for')
        #print(fname)
        file_found=0
        chk2=0
    

        
      
    if file_found==1: 
     print(chk2)
     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")  
     print(matrix_data)
     chk2=1
     ann=pd.read_csv(f2)
     cols=['Start time','End time', 'File Start']
     annotation= ann[cols]
     #data=annotation['Start time']
     t=['strt_time','end_time','file_strt']
     datat=pd.DataFrame()
     datat[t]=pd.DataFrame(annotation)
     dataf=datat.iloc[0:1,2]
     dataff=pd.DataFrame()
     dataff=pd.DataFrame(dataf)
     ff= datat.iloc[0,2]
     st=datat.strt_time.str.split(":",expand=True)
     et=datat.end_time.str.split(":",expand=True)
     f_strt=dataff.file_strt.str.split(":",expand=True)
     k=st.astype('int')
     j=et.astype('int')
     f=f_strt.astype('int')
     s_time=fs*(k[1]*60+k[2]) # conversion into seconds
     e_time=fs*(j[1]*60+j[2]) # conversion into seconds
     f_time=fs*(f[1]*60+f[2])
     xx=f_time.at[0] # getting single value of file strt
     fs_time=s_time-xx
     fe_time=e_time-xx
     i = 0;
     j = 6000;
     
     
     ann_row=s_time.size
     e,file_length=data.shape
     print('e is',e)
     print('file_length is', file_length)
     loop_count=file_length/6000
     loop_count=math.ceil(loop_count)
     for y in range(loop_count-1):
       chck=0
       print(i, j)
       for x in range(ann_row-1):
          
         if not data[:,i:j].any():
             continue  
         if (fs_time[x] or fe_time[x]) in range(i,j):
           chck=1
           if typee==1 :
             example_abn=data[:,i: j]
             print('*************************ndarray or not')
             print(isinstance(example_abn, (np.ndarray)))
             matrix_data.append(example_abn)
             matrix_data2.append((example_abn))
             matrix_data1=matrix_data1+list(np.array(example_abn).reshape(21,-1))
             matrix_label.append(0)
             break
           if typee==0:
             example_nor=data[:,i: j]
             #print('*************************ndarray or not')
             #print(isinstance(example_abn, (np.ndarray)))

             matrix_data.append(example_nor)
             matrix_data2.append((example_nor))

             matrix_data1=matrix_data1+list(np.array(example_nor).reshape(21,-1))

             matrix_label.append(1)
             break
           if y == 0:
            
 
             example_1 = data[:,0: 6000]
             print('*************************ndarray or not')
             print(isinstance(example_abn, (np.ndarray)))

             matrix_data.append(example_1)
             matrix_data1=matrix_data1+list(np.array(example_1).reshape(21,-1))
             matrix_data2.append((example_1))

             matrix_label.append(typee)
           elif j < file_length:
             example = data[:,i: j]
             matrix_data.append(example)
             matrix_data2.append((example))

             matrix_data1=matrix_data1+list(np.array(example).reshape(21,-1))
 
             matrix_label.append(typee)
           elif y==n-1:
            example = data[:,-6000: ]
            matrix_data.append(example)
            matrix_data2.append((example))

            matrix_data1=matrix_data1+list(np.array(example).reshape(21,-1))
            matrix_label.append(typee)
            break
         i = int(j )
         j = int(j + 6000)
         
    if file_found==0: 
     print(chk2)
     
     chk2=1
     
     i = 0;
     j = 6000;
     e,file_length=data.shape

     #print(file_length)
     loop_count=file_length/6000
     loop_count=math.ceil(loop_count)
     for y in range(loop_count-1):
       if not data[:,i:j].any():
             continue 
       chck=0
       print(i, j)
       if y == 0:
         example_1 = data[:,0: 6000]
         matrix_data.append(example_1)
         matrix_label.append(typee)
         matrix_data2.append((example_1))

         matrix_data1=matrix_data1+list(np.asarray( example_1).reshape(21,-1))

         
       elif j < file_length:
         example = data[:,i: j]
         matrix_data.append(example)
         matrix_label.append(typee)
         matrix_data2.append((example))

         matrix_data1=matrix_data1+list((example))
       elif y==n-1:
         example = data[:,-6000: ]
         matrix_data.append(example)
         matrix_label.append(typee)  
         matrix_data2.append((example))

         matrix_data1=matrix_data1+list(np.asarray(example).reshape(21,-1))

         break
       i = int(j )
       j = int(j + 6000)
       print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")  
       #print(matrix_data)
      
    matrix_data=np.array(matrix_data)
    matrix_label1=matrix_label+matrix_label1
    
    matrix_label=np.array(matrix_label)
    matrix_f=matrix_f+matrix_data2
    print('*****************************shape of matrix_f', len(matrix_f),len(matrix_f[0]),len(matrix_f[1]))
    return matrix_data2, matrix_label


def get_all_sorted_file_names_and_labels(train_or_eval, folders):
    all_file_names = []
    for folder in folders:
        full_folder = os.path.join(folder, train_or_eval) + '/'
        log.info("Reading {:s}...".format(full_folder))
        this_file_names = read_all_file_names(full_folder, '.edf', key='time')
        log.info(".. {:d} files.".format(len(this_file_names)))
        all_file_names.extend(this_file_names)
    log.info("{:d} files in total.".format(len(all_file_names)))
    #all_file_names = sorted(all_file_names, key=time_key)
    #COMMENT OUT TO STOP SORTING

    labels = ['/abnormal/' in f for f in all_file_names]
    labels = np.array(labels).astype(np.int64)
    return all_file_names, labels


class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 data_folders,                 train_or_eval='train', sensor_types=['EEG']):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types
        self.data_folders = data_folders
	

    def load(self, only_return_labels=False):
        global matrix_label1
        matrix_label1=[]
        global matrix_data1
        matrix_data1=[]
        global matrix_f
        matrix_f=[]

        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval,
            folders=self.data_folders,)

        if self.max_recording_mins is not None:
            log.info("Read recording lengths...")
            assert 'train' == self.train_or_eval
            # Computation as:

            lengths = [get_recording_length(fname) for fname in all_file_names] 
            #print('before np array', lengths)
            #print('after np array', lengths)
            
            lengths = np.array(lengths)
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels
        if only_return_labels:
            return cleaned_labels
        X = []
        y = []
        n_files=0
        #for ff in enumerate(cleaned_file_names[:self.n_recordings]):
            #n_files=(get_recording_length(ff)/6000)+n_files
        n_files = len(cleaned_file_names[:self.n_recordings])
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            #i_fname=(get_recording_length(i_fname)/6000)
            log.info("Load {:d} of {:d}".format(i_fname + 1,n_files))
            print('*******************************************************************************************************************************************************')
            print(fname)
            print(n_files)
            print(cleaned_labels[i_fname])
            
            x,yy = load_data(fname,cleaned_labels[i_fname], preproc_functions=self.preproc_functions,
                          sensor_types=self.sensor_types)
            print('shape of x', yy)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            if x is None:
              continue

                
            
            
            #assert x is not None
            #MYEDIT
            #xx=x.reshape(21,-1)
           
            #print('shape of xx', x.shape)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #X.append(x)
            #y.append(yy)
        
        print(isinstance(y, (list,np.ndarray)))
        
        print(isinstance(X, np.ndarray))
        
        #print(len(X),len(X[0]),len(X[1]))
        y_arr=np.array(matrix_label1)
        print(y_arr)
        print(len(matrix_data1),len(matrix_data1[0]),len(matrix_data1[1]))
        print(len(y_arr))
        X=matrix_f
        
        XX=matrix_data1
        #X=np.array(matrix_f)
        #print(np.asarray(X).shape)
        X=matrix_f
        #print(matrix_f)
        
        
        return matrix_f, y_arr
