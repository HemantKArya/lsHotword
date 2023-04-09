from __future__ import print_function
from . import lsrc
from sys import argv,exit as sysExit
from PyQt6.QtWidgets import QApplication,QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import qRegisterResourceData,qUnregisterResourceData
app = QApplication(argv)
splash_object = QSplashScreen(QPixmap(":/icons/assets/lshotword_banner.png"))
splash_object.show()

from PyQt6.QtWidgets import QSpinBox,QProgressBar,QWidget,QFileDialog,QSpacerItem,QSizePolicy,QTextBrowser,QVBoxLayout,QHBoxLayout,QFrame,QLabel,QPushButton,QTextEdit
from PyQt6.QtGui import QKeySequence,QShortcut,QIcon,QFont,QDesktopServices,QFontDatabase,QPixmap
from PyQt6.QtCore import QSize,QMetaObject,QCoreApplication,QThread, pyqtSignal, QObject,QUrl,Qt

# from nmtrain import Ui_lsForm

from numpy.random import randint
from numpy import zeros,array
from numpy import save as npsave
from pydub import AudioSegment
import sys
import os
from scipy.io import wavfile
from tensorflow.keras.callbacks import Callback
try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam
print("Full Hotword Data Generator and Trainer")
# import argparse
from matplotlib.pyplot import specgram
from sklearn.model_selection import train_test_split
from . import hotword as ht


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

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
    x = graph_spectrogram("train.wav")
    print(y)
    return x, y


    
class UpdateProgressBar(QObject,Callback):
    progress_updated = pyqtSignal(int)

    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps

    def on_train_begin(self, logs=None):
        self.progress_updated.emit(0)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('loss')
        accuracy = logs.get('accuracy')
        steps = (epoch + 1) * self.total_steps // self.params['epochs']
        progress = int(steps * 100 / self.total_steps)
        self.progress_updated.emit(progress)


class TrainThread(QThread):
    output_log = pyqtSignal(str)
    progressUpdate = pyqtSignal(int)

    def __init__(self,posPath,negPath,bgPath,outPath,epochs,Tx=5511,n_freq=101,nsamples=30,ty=1375,batch_size=10,Kcallback = None):
        super().__init__()
        global Ty
        Ty = ty
        self.posPath = posPath
        self.negPath = negPath
        self.bgPath = bgPath
        self.outPath = outPath
        self.epochs = epochs
        self.batch_size = batch_size
        self.Tx = Tx
        self.Kcallback = Kcallback
        self.n_freq =n_freq
        self.nsamples = nsamples
    def run(self):
        update_progress_bar = UpdateProgressBar(self.nsamples)
        update_progress_bar.progress_updated.connect(self.progressUpdate.emit)
        callbacks = [update_progress_bar]
        self.output_log.emit("Prepairing data...")
        activates, negatives, backgrounds = load_raw_audio_from_diff_path(self.posPath, self.negPath, self.bgPath)
        X = []
        Y = []
        for i in range(0, self.nsamples):
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
        self.output_log.emit("Data loaded successfully...")
        self.output_log.emit("Shape of X= "+str(X.shape))
        self.output_log.emit("Shape of Y= "+str(Y.shape))
        npsave('./X.npy',X)
        npsave('./Y.npy',Y)
        print("Successfull!!")
        self.output_log.emit("X.npy and Y.npy saved...")
        try:
            assert X.ndim == 3, "Error: X not have correct dimentions"
            assert Y.ndim == 3, "Error: Y not have correct dimentions"
            assert Y.shape[1] == Ty, "Error: Y not have correct dimentions"
            assert X.shape[1] == self.Tx, "Error: X not have correct dimentions"
        except Exception as e:
            print(e)
            self.output_log.emit(str(e))
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=37)
        model = ht.Hmodel(input_shape = (self.Tx, self.n_freq))
        model.summary()
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        self.output_log.emit("Starting to train model with {} epochs...".format(self.epochs))
        model.fit(X,Y,batch_size=self.batch_size,epochs=self.epochs,callbacks=callbacks)
        self.output_log.emit("Model Trained Successfully...")
        model.save(os.path.join(self.outPath,"model.h5"))
        print("Model Saved !",os.path.join(self.outPath,"model.h5"))
        self.output_log.emit("Model Saved -> "+str(os.path.join(self.outPath,"model.h5")))

class Ui_lsForm(object):
    def setupUi(self, lsForm):
        lsForm.setObjectName("lsForm")
        lsForm.resize(869, 555)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/icons/assets/lshotword_logo.png"), QIcon.Mode.Normal, QIcon.State.Off)
        lsForm.setWindowIcon(icon)
        lsForm.setWindowOpacity(1.0)
        lsForm.setStyleSheet("QToolTip{\n background-color:rgb(2,2,10);border:none;color:rgb(232, 244, 254);}\nQPushButton{\n"
"background-color:none;\n"
"color: rgb(232, 244, 254);\n"
"border-radius:15px;\n"
"font-size:18px;\n"
"}\n"
"QPushButton::hover{\n"
"background-color: rgba(104, 144, 255,100);\n"
"}\n"
"\n"
"QLabel{\n"
"color: rgb(232, 244, 254);\n"
"font-size:18px;\n"
"}\n"
"\n"
"QSpinBox {\n"
"    padding-right: 15px; /* make room for the arrows */\n"
"    \n"
"    background-color: rgb(137, 150, 166);\n"
"    border-width: 3;\n"
"    font-size:18px;\n"
"    font-weight:30px;\n"
"    color: rgb(223, 233, 242);\n"
"    border-radius:10px;\n"
"}\n"
"QSpinBox::up-button {\n"
"   border:None;\n"
"image: none;\n"
"  background:Transparent;\n"
"width: 0px;\n"
"}\n"
"QSpinBox::down-button {\n"
"   border:None;\n"
"image: none;\n"
"width: 0px;\n"
"background:Transparent;\n"
"}")
        self.horizontalLayout = QHBoxLayout(lsForm)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.title_frame = QFrame(lsForm)
        self.title_frame.setStyleSheet("QFrame{\n"
"\n"
"    background-color: rgb(37, 49, 64);\n"
"}")
        self.title_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.title_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.title_frame.setObjectName("title_frame")
        self.horizontalLayout_2 = QHBoxLayout(self.title_frame)
        self.horizontalLayout_2.setContentsMargins(2, 0, 2, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.title = QLabel(self.title_frame)
        self.title.setMinimumSize(QSize(0, 30))
        self.title.setMaximumSize(QSize(16777215, 40))
        self.title.setStyleSheet("QLabel{\n"
"color:qlineargradient(spread:pad, x1:0.308458, y1:0.693, x2:0.756, y2:0, stop:0.233831 rgba(255, 255, 255, 255), stop:1 rgba(164, 217, 242, 255));\n"
"font-size:24px;\n"
"}")
        self.title.setScaledContents(False)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("title")
        self.verticalLayout_2.addWidget(self.title)
        self.tool_frame = QFrame(self.title_frame)
        self.tool_frame.setMinimumSize(QSize(0, 50))
        self.tool_frame.setMaximumSize(QSize(16777215, 50))
        self.tool_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.tool_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.tool_frame.setObjectName("tool_frame")
        self.horizontalLayout_4 = QHBoxLayout(self.tool_frame)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(4)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.browse_pos_audio = QPushButton(self.tool_frame)
        self.browse_pos_audio.setMinimumSize(QSize(40, 40))
        self.browse_pos_audio.setMaximumSize(QSize(40, 40))
        self.browse_pos_audio.setText("")
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/icons/assets/posIcons.png"), QIcon.Mode.Normal, QIcon.State.Off)
        self.browse_pos_audio.setIcon(icon1)
        self.browse_pos_audio.setObjectName("browse_pos_audio")
        self.horizontalLayout_3.addWidget(self.browse_pos_audio)
        self.browse_neg_audio = QPushButton(self.tool_frame)
        self.browse_neg_audio.setMinimumSize(QSize(40, 40))
        self.browse_neg_audio.setMaximumSize(QSize(40, 40))
        self.browse_neg_audio.setText("")
        icon2 = QIcon()
        icon2.addPixmap(QPixmap(":/icons/assets/negIcon.png"), QIcon.Mode.Normal, QIcon.State.Off)
        self.browse_neg_audio.setIcon(icon2)
        self.browse_neg_audio.setObjectName("browse_neg_audio")
        self.horizontalLayout_3.addWidget(self.browse_neg_audio)
        self.browse_bg_audio = QPushButton(self.tool_frame)
        self.browse_bg_audio.setMinimumSize(QSize(40, 40))
        self.browse_bg_audio.setMaximumSize(QSize(40, 40))
        self.browse_bg_audio.setText("")
        icon3 = QIcon()
        icon3.addPixmap(QPixmap(":/icons/assets/bgIcon.png"), QIcon.Mode.Normal, QIcon.State.Off)
        self.browse_bg_audio.setIcon(icon3)
        self.browse_bg_audio.setObjectName("browse_bg_audio")
        self.horizontalLayout_3.addWidget(self.browse_bg_audio)
        self.browse_output_dir = QPushButton(self.tool_frame)
        self.browse_output_dir.setMinimumSize(QSize(40, 40))
        self.browse_output_dir.setMaximumSize(QSize(40, 40))
        self.browse_output_dir.setStyleSheet("")
        self.browse_output_dir.setText("")
        icon4 = QIcon()
        icon4.addPixmap(QPixmap(":/icons/assets/exportIcon.png"), QIcon.Mode.Normal, QIcon.State.Off)
        self.browse_output_dir.setIcon(icon4)
        self.browse_output_dir.setObjectName("browse_output_dir")
        self.horizontalLayout_3.addWidget(self.browse_output_dir)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.toogle_train = QPushButton(self.tool_frame)
        self.toogle_train.setMinimumSize(QSize(40, 40))
        self.toogle_train.setMaximumSize(QSize(40, 40))
        self.toogle_train.setStyleSheet("")
        self.toogle_train.setText("")
        icon5 = QIcon()
        icon5.addPixmap(QPixmap(":/icons/assets/trainIcon.png"), QIcon.Mode.Normal, QIcon.State.Off)
        icon5.addPixmap(QPixmap(":/icons/assets/waitIcon.png"), QIcon.Mode.Normal, QIcon.State.On)
        icon5.addPixmap(QPixmap(":/icons/assets/waitIcon.png"), QIcon.Mode.Disabled, QIcon.State.On)
        icon5.addPixmap(QPixmap(":/icons/assets/trainIcon.png"), QIcon.Mode.Active, QIcon.State.Off)
        icon5.addPixmap(QPixmap(":/icons/assets/trainIcon.png"), QIcon.Mode.Selected, QIcon.State.Off)
        icon5.addPixmap(QPixmap(":/icons/assets/trainIcon.png"), QIcon.Mode.Selected, QIcon.State.On)
        self.toogle_train.setIcon(icon5)
        self.toogle_train.setObjectName("toogle_train")
        self.horizontalLayout_3.addWidget(self.toogle_train)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.tool_frame)
        self.frame = QFrame(self.title_frame)
        self.frame.setMinimumSize(QSize(0, 36))
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_8 = QHBoxLayout(self.frame)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.progressBar = QProgressBar(self.frame)
        self.progressBar.setMinimumSize(QSize(60, 0))
        self.progressBar.setMaximumSize(QSize(16777215, 30))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"text-align:centre;\n"
"color:rgb(203, 223, 255);\n"
"background-color: #253140;\n"
"/*border: 2px solid #3DBFF2;\n"
"border-radius:15px;*/\n"
"border:none;\n"
"font-size:18px;\n"
"}\n"
"QProgressBar::chunk {\n"
"\n"
"    background-color: qlineargradient(spread:pad, x1:0.652, y1:0.0852269, x2:1, y2:1, stop:0.20398 rgba(97, 136, 255, 255), stop:1 rgba(54, 83, 242, 255));\n"
" border-radius: 10px;\n"
"\n"
"}")
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_7.addWidget(self.progressBar)
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.l_samples = QLabel(self.frame)
        self.l_samples.setObjectName("l_samples")
        self.horizontalLayout_7.addWidget(self.l_samples)
        self.spin_nsamples = QSpinBox(self.frame)
        self.spin_nsamples.setMinimumSize(QSize(0, 29))
        self.spin_nsamples.setMaximumSize(QSize(50, 16777215))
        self.spin_nsamples.setStyleSheet("QSpinBox {\n"
"    padding-right: 15px; \n"
"    background-color: rgb(137, 150, 166);\n"
"    border-width: 3;\n"
"    font-size:18px;\n"
"    font-weight:30px;\n"
"    color: rgb(223, 233, 242);\n"
"    border-radius:10px;\n"
"}\n"
"")
        self.spin_nsamples.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_nsamples.setMinimum(10)
        self.spin_nsamples.setMaximum(5000)
        self.spin_nsamples.setProperty("value", 30)
        self.spin_nsamples.setObjectName("spin_nsamples")
        self.horizontalLayout_7.addWidget(self.spin_nsamples)
        self.l_epochs = QLabel(self.frame)
        self.l_epochs.setObjectName("l_epochs")
        self.horizontalLayout_7.addWidget(self.l_epochs)
        self.spin_epochs = QSpinBox(self.frame)
        self.spin_epochs.setMaximumSize(QSize(50, 16777215))
        self.spin_epochs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_epochs.setMinimum(1)
        self.spin_epochs.setMaximum(10000)
        self.spin_epochs.setObjectName("spin_epochs")
        self.horizontalLayout_7.addWidget(self.spin_epochs)
        self.l_bsize = QLabel(self.frame)
        self.l_bsize.setObjectName("l_bsize")
        self.horizontalLayout_7.addWidget(self.l_bsize)
        self.spin_bsize = QSpinBox(self.frame)
        self.spin_bsize.setMaximumSize(QSize(50, 16777215))
        self.spin_bsize.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_bsize.setMinimum(2)
        self.spin_bsize.setMaximum(1000)
        self.spin_bsize.setProperty("value", 10)
        self.spin_bsize.setObjectName("spin_bsize")
        self.horizontalLayout_7.addWidget(self.spin_bsize)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_2.addWidget(self.frame)
        self.cmd_logs = QTextBrowser(self.title_frame)
        self.cmd_logs.setMinimumSize(QSize(0, 100))
        font = QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.cmd_logs.setFont(font)
        self.cmd_logs.setStyleSheet("QTextBrowser{\n"
"color:#DFE9F2;\n"
"border:2px solid #DFE9F2;\n"
"border-radius: 15px\n"
"}")
        self.cmd_logs.setObjectName("cmd_logs")
        self.verticalLayout_2.addWidget(self.cmd_logs)
        self.frame_3 = QFrame(self.title_frame)
        self.frame_3.setMinimumSize(QSize(0, 25))
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_6 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.info = QLabel(self.frame_3)
        self.info.setStyleSheet("QLabel{\n"
"color:qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0.233831 rgba(240, 244, 255, 255), stop:1 rgba(189, 228, 242, 255));\n"
"font-weight:29px;\n"
"font-size:18px;\n"
"border:none;\n"
"}")
        self.info.setObjectName("info")
        self.horizontalLayout_5.addWidget(self.info)
        self.github_btn = QPushButton(self.frame_3)
        self.github_btn.setMinimumSize(QSize(40, 40))
        self.github_btn.setMaximumSize(QSize(40, 40))
        self.github_btn.setText("")
        icon6 = QIcon()
        icon6.addPixmap(QPixmap(":/icons/assets/gitIcon.png"), QIcon.Mode.Normal, QIcon.State.On)
        self.github_btn.setIcon(icon6)
        self.github_btn.setCheckable(False)
        self.github_btn.setObjectName("github_btn")
        self.horizontalLayout_5.addWidget(self.github_btn)
        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.addWidget(self.frame_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout.addWidget(self.title_frame)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(lsForm)
        QMetaObject.connectSlotsByName(lsForm)

    def retranslateUi(self, lsForm):
        _translate = QCoreApplication.translate
        lsForm.setWindowTitle(_translate("lsForm", "lsHotword"))
        self.title.setText(_translate("lsForm", "Model Trainer"))
        self.l_samples.setText(_translate("lsForm", "Samples"))
        self.l_epochs.setText(_translate("lsForm", "Epochs"))
        self.l_bsize.setText(_translate("lsForm", "Batch Size"))
        self.cmd_logs.setHtml(_translate("lsForm", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.info.setText(_translate("lsForm", "@iamhemantindia"))




class window(QWidget,Ui_lsForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        font_id = QFontDatabase.addApplicationFont(':fonts/fonts/ubuntuBold.ttf')
        font_family = QFontDatabase.applicationFontFamilies(font_id)
    
        self.title.setFont(QFont(font_family[0]))
        self.cmd_logs.setFont(QFont(font_family[0],13))
        self.l_samples.setFont(QFont(font_family[0]))
        self.l_epochs.setFont(QFont(font_family[0]))
        self.l_bsize.setFont(QFont(font_family[0]))
        self.info.setFont(QFont(font_family[0]))
        #set tooltips
        self.browse_bg_audio.setToolTip('Browse Background audio folder')
        self.browse_neg_audio.setToolTip('Browse Negative audio folder')
        self.browse_output_dir.setToolTip('Browse Output folder')
        self.browse_pos_audio.setToolTip('Browse Positive audio folder')
        self.toogle_train.setToolTip('Start training hotword model')
        #set icons
        # self.browse_bg_audio.setIcon(QIcon('./assets/bgIcon.png'))
        self.browse_bg_audio.setIconSize(self.browse_bg_audio.size())
        # self.browse_pos_audio.setIcon(QIcon('./assets/posIcons.png'))
        self.browse_pos_audio.setIconSize(self.browse_pos_audio.size())
        # self.browse_neg_audio.setIcon(QIcon('./assets/negIcon.png'))
        self.browse_neg_audio.setIconSize(self.browse_neg_audio.size())
        # self.browse_output_dir.setIcon(QIcon('./assets/exportIcon.png'))
        self.browse_output_dir.setIconSize(self.browse_output_dir.size())
        # self.toogle_train.setIcon(QIcon('./assets/trainIcon.png'))
        self.toogle_train.setIconSize(self.toogle_train.size())
        # self.github_btn.setIcon(QIcon('./assets/gitIcon.png'))
        self.github_btn.setIconSize(self.github_btn.size())
        #set handles
        self.browse_pos_audio.clicked.connect(self.onBrowsePos)
        self.browse_neg_audio.clicked.connect(self.onBrowseNeg)
        self.browse_bg_audio.clicked.connect(self.onBrowseBg)
        self.browse_output_dir.clicked.connect(self.onBrowseOutput)
        self.toogle_train.clicked.connect(self.trainBtn)
        self.github_btn.clicked.connect(self.openGithub)
        self.progressBar.setVisible(False)
        self.got_allDir = [0,0,0,0]
        self.trainBtnState = True

    def openGithub(self):
        QDesktopServices.openUrl(QUrl('www.github.com/HemantKArya/lsHotword'))
    def updateLogs(self,new_log):
        self.cmd_logs.setText(self.cmd_logs.toPlainText()+new_log+'\n')
    def UpdateProgressValue(self,value):
        self.progressBar.setVisible(True)
        self.progressBar.setValue(value)
        if value==100:
            # self.toogle_train.setEnabled(True)
            self.toogleTrainBtn()
    def toogleTrainBtn(self):
        if not self.trainBtnState:
            self.trainBtnState = True
            self.toogle_train.setIcon(QIcon(':/icons/assets/trainIcon.png'))
        else:
            self.trainBtnState = False
            self.toogle_train.setIcon(QIcon(':/icons/assets/waitIcon.png'))
    def trainBtn(self):
        if sum(self.got_allDir) ==4 and self.trainBtnState:
            self.trainModel = TrainThread(self.posFolderPath.toLocalFile(), self.negFolderPath.toLocalFile(), self.bgFolderPath.toLocalFile(), self.outFolderPath.toLocalFile(), epochs=self.spin_epochs.value(), nsamples=self.spin_nsamples.value(),batch_size=self.spin_bsize.value())
            self.trainModel.output_log.connect(self.updateLogs)
            self.trainModel.progressUpdate.connect(self.UpdateProgressValue)
            self.toogleTrainBtn()
            # self.toogle_train.setDisabled(True)
            self.trainModel.start()
    def onBrowsePos(self):
        self.posFolderPath = QFileDialog.getExistingDirectoryUrl(self)
        if not(self.posFolderPath.isEmpty()):
            self.got_allDir[0] = 1
            print(self.posFolderPath.toLocalFile())
            self.updateLogs("Positive Audio folder -> "+self.posFolderPath.toLocalFile())
        else:
            self.got_allDir[0] = 0
            print("Empty Directory!")
    def onBrowseNeg(self):
        self.negFolderPath = QFileDialog.getExistingDirectoryUrl(self)
        if not(self.negFolderPath.isEmpty()):
            self.got_allDir[1] = 1
            print(self.negFolderPath.toLocalFile())
            self.updateLogs("Negative Audio folder -> "+self.negFolderPath.toLocalFile())
        else:
            self.got_allDir[1] = 0
            print("Empty Directory!")
    def onBrowseBg(self):
        self.bgFolderPath = QFileDialog.getExistingDirectoryUrl(self)
        if not(self.bgFolderPath.isEmpty()):
            self.got_allDir[2] = 1
            print(self.bgFolderPath.toLocalFile())
            self.updateLogs("Background Audio folder -> "+self.bgFolderPath.toLocalFile())
        else:
            self.got_allDir[2] = 0
            print("Empty Directory!")
    def onBrowseOutput(self):
        self.outFolderPath = QFileDialog.getExistingDirectoryUrl(self)
        if not(self.outFolderPath.isEmpty()):
            self.got_allDir[3] = 1
            print(self.outFolderPath.toLocalFile())
            self.updateLogs("Model Output folder -> "+self.outFolderPath.toLocalFile())
        else:
            self.got_allDir[3] = 0
            print("Empty Directory!")


def main():
    splash_object.close()
    win = window()
    win.show()
    sysExit(app.exec())