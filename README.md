# lsHotword
![lshotword-banner](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/lshotword_banner.png)

[![Github](	https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/HemantKArya/lsHotword)

**lsHotword** is Wake Word detector and Easy to use Module Which is open-Source and **Free License**.If you face any problem you can contact me on my E-mail at the last of this Document. For any Help we also have YouTube channel link is at the last of this file.

# Install lsHotword using pip
 To install lsHotword open cmd and type-
 ```
 pip install lsHotword --upgrade
 ```
 Make sure your python should be on path.

# Training Your Own Model
## Create Dataset
To train your own Model you have to create your Dataset.
Record 10 audio with voice **Activate** and place it under "Positives folder" and record 10 **Non-Activate Word ** Which are not Activate and place it under negatives folder. And like that record 2 or more than 2 background noises in different environments of 10 seconds. Make sure to record these audios in 44100 Hz sample rate, either will you have to change too many parameters.You can use free software and tools like-

1. [Audacity](https://www.audacityteam.org/download/) Edit audio clips to select only 10 sec of background noise or exact part where you said you hotword in audio.
2. [FFMpeg](https://ffmpeg.org/) for converting the sample rate to 44100Hz

 [Examples are provided on Github for audio](https://github.com/HemantKArya/lsHotword/tree/main/Examples/data) (from deeplearning.ai's deep learning program).
Your Directory should look like this-
- data/
    - background/
        - file1.wav
        - file2.wav
        - file3.wav
    - positives/
        - file4.wav
        - file5.wav
        - file6.wav
        - .
        - .
    - negatives/
        - file7.wav
        - file8.wav
        - file9.wav
        - .
        - .
        

![audioexample](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/Q4tnfi3E.png)

Then open command prompt here (eg. outside "data" folder) and type-.

```
lshUITrainer
```
Press enter and you will see this window-
![lshotwordtrainerwindow](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/UsGpiupQt1.png)

1. Import **positives** audio folder.
2. Import **negative** audio folder.
3. Import **background** noise samples folder.
4. Import Output directory where you want to save model after training.
5. No. of training examples to generate keep it 30 and increase it if you have more data and for more better accuracy in result.
6. No. of **Epochs** (How much times you want to train your model eg 100-400).
7. Batch size increase it if you have more GPU power or keep it same.
![after-t-step](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/3AwBBo2nOR.png)
8. When 1-7 all steps are done then start training by clicking on this button.

And you will see something like that-
![starttraining](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/0mySroWeGr.png)
When training finishes  you see the output directory where wake word model is saved.
![training-complete](https://github.com/HemantKArya/lsHotword/raw/main/docfiles/q34bpEOMao.png)

## Without GUI (Optional)
This is an alternative way, if GUI Trainer is not working or you want to do it step by step.

Type this commmand to generate training examples from raw data-
```
lsHDatagen --input ./data --nsamp 32
```
Here **data** is the folder where both folders **"positives and negatives"** are located and **nsamp** are number of training examples you want to generate. After finishing this process you will see two files 'X.npy and Y.npy' outside data folder.
Now its time to train our Hotword Model open cmd again here and type-
```
lsHTrainer --inX X.npy --inY Y.npy --epochs 600
```
and then after few minutes you will get your model  with name **model.h5**, Hurray!! you just created your own hotword or wake word model. 

## Test Hotword Model live
Now test it using this command-

'lsHTestModel --model < path to model >'
```
lsHTestModel --model ./model.h5
```
and then you will see a text like **<< Waiting for Hotword >>** when you see this text then try to speak your wake word and see a chime sound will beep!!

# Using Trained Model

After installing **lsHotword** and training your own model e.g **model.h5** then you are ready to use it any program where you want to use it. Example-

```
from lsHotword.ls import Hotword

path_to_model = "./model.h5"          # path to model where it is located
hotword = Hotword(path_to_model)      # create object of Hotword

#Now call HotwordLoop function
if hotword.HotwordLoop():
    print('Wake word Detected!!')    # print when hotword is detected.

```
And thats all you are ready to go use it in any program you want to make. If you want to contribute to this little project then feel free to make push request on dev branch.

*This module is created with the help of **Deeplearning.ai 's Deep Learning Program**.*
# For More Information

For more information or send your query at:
iamhemantindia@protonmail.com

Or

 [![](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/iamhemantindia)

Or Checkout Our Youtube Channel Logical Spot (Hemant Kumar)

[![youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/c/LogicalSpot)

Feel free to **Contribute** at - 

[![Github](	https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/HemantKArya/lsHotword)