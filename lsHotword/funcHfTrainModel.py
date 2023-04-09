from numpy.random import randint
from numpy import zeros,array
from numpy import save as npsave
from pydub import AudioSegment
import os
from scipy.io import wavfile
try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam
print("Full Hotword Data Generator and Trainer")
import argparse
from matplotlib.pyplot import specgram
from sklearn.model_selection import train_test_split
from . import hotword as ht


# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(pathD):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(pathD + "/positives/"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(pathD + "/positives/"+filename)
            activates.append(activate)
    for filename in os.listdir(pathD+"/backgrounds/"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(pathD+"/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir(pathD + "/negatives/"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(pathD + "/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds

def load_raw_audio_from_diff_path(posPath,negPath,bgPath):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(posPath):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(os.path.join(posPath,filename))
            activates.append(activate)
    for filename in os.listdir(bgPath):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(os.path.join(bgPath,filename))
            backgrounds.append(background)
    for filename in os.listdir(negPath):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(os.path.join(negPath,filename))
            negatives.append(negative)
    return activates, negatives, backgrounds


 
def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

 
def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    

    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False
    
    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
   

    return overlap

 
def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
   
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)
    
    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)

    
    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time

 
def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
   
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
 
    
    return y

 
def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Set the random seed
   
    
    # Make background quieter
    background = background - 20

  
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []
 
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = randint(0, 5)
    random_indices = randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
  
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)
    

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = randint(0, 3)
    random_indices = randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

   
    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
   
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = ht.graph_spectrogram("train.wav")
    print(y)
    return x, y

 

def main():
    global Ty
    parser = argparse.ArgumentParser(description='Dir For Dataset e.g. neg an pos')
    parser.add_argument('--input', action='store', type=str, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--tx', action='store', type=int, default=5511) # default for 44100Hz 10sec audio file
    parser.add_argument('--nsamp', action='store', type=int, default=32) 
    parser.add_argument('--nf', action='store', type=int, default=101)  # default for 44100Hz audio file
    parser.add_argument('--ty', action='store', type=int, default=1375) # default for 44100Hz audio file
    parser.add_argument('--bsize', action='store', type=int, default=10) # default for 44100Hz audio file
    args = parser.parse_args()
    Ty = args.ty # The number of time steps in the output of our model
    TrainHotwordModel(pathD=args.input,epochs=args.epochs,Tx=args.tx,n_freq=args.nf,nsamples=args.nsamp,ty=args.ty,batch_size=args.bsize)
    

def TrainHotwordModel(pathD,epochs,Tx=5511,n_freq=101,nsamples=30,ty=1375,batch_size=10):
    global Ty
    Ty = ty
    activates, negatives, backgrounds = load_raw_audio(pathD=pathD)
    X = []
    Y = []
    for i in range(0, nsamples):
        if i%10 == 0:
            print(i)
        x, y = create_training_example(backgrounds[i % 2], activates, negatives)
        X.append(x.swapaxes(0,1))
        Y.append(y.swapaxes(0,1))
    X=array([X])
    X=X[0]
    Y=array([Y])
    Y=Y[0]
    print(X.shape)
    print(X.ndim)
    print(Y.shape)
    print(Y.ndim)
    npsave('./X.npy',X)
    npsave('./Y.npy',Y)
    print("Successfull!!")
    assert X.ndim == 3, "Error: X not have correct dimentions"
    assert Y.ndim == 3, "Error: Y not have correct dimentions"
    assert Y.shape[1] == Ty, "Error: Y not have correct dimentions"
    assert X.shape[1] == Tx, "Error: X not have correct dimentions"
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=37)
    model = ht.Hmodel(input_shape = (Tx, n_freq))
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(X,Y,batch_size=batch_size,epochs=epochs)
    model.save("model.h5")
    print("Model Saved !")