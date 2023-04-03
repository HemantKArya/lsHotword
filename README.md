# lsHotword 🤖
[![Github](	https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/HemantKArya)

**lsHotword** detector is Easy to use Module Which is open-Source and **Free License**.This module is created with the help of **Deeplearning.ai 's Deep Learning Program**. If you have any problem you can contact me on my E-mail at the last of this Document. For any Help we also have YouTube channel link is at the last of this file.

# Install lsHotword using pip ✌
 To install lsHotword open cmd and type-
 ```
 pip install lsHotword
 ```
 make sure your python should be on path.

# Training Your Own Model 😊
## Create Dataset
To train your own Model you have to create your Dataset.
Record 10 audio with voice **Activate** and place it under "Positives folder" and record 10 **Non-Activate Word ** Which are not Activate and place it under negatives folder. And like that record 2 or more than 2 background noises in different environments of 10 seconds. Make sure to record these audios of in 44100 Hz sample rate, either will you have to change too many parameters. Examples are provided on Github(from coursera's deep learning program).
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

Then open command prompt here (eg. outside "data" folder) and type-.
```
lsHDatagen --input ./data --nsamp 32
```
Here **data** is the folder where both folders **"positives and negatives"** are located and **nsamp** are number of training examples you want to generate. After finishing this process you will see two files 'X.npy and Y.npy' outside data folder.
Now its time to train our Hotword Model open cmd again here and type-
```
lsHTrainer --inX X.npy --inY Y.npy --epochs 600
```
and then after few minutes you will get your model  with name **model.h5**, Hurray!! you just created your own hotword or wake word model. Now test it using this command-
```
lsHTestModel --model ./model.h5
```
and then you will see a text like **<<Waiting for Hotword>>** when you see this text then try to speak your wake word and see a chime sound will beep!!

# Using Trained Model 😎

After installing **lsHotword** and training your own model e.g **model.h5** then you are ready to use it any program where you want to use it. Example-

```
from lsHotword.ls import Hotword

path_to_model = "./model.h5"          # path to model where it is located
hotword = Hotword(path_to_model)      # create object of Hotword

#Now call HotwordLoop function
if hotword.HotwordLoop():
    print('Wake word Detected!!')    # print when hotword is detected.

```

# For More Information 😻

For more information or send your query at:
iamhemantindia@protonmail.com

or Checkout Our Youtube Channel

[![youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/c/LogicalSpot)