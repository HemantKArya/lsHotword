# lsHotword 🤖
[![Github](	https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/HemantKArya)

**lsHotword** detector is an easy-to-use module which is open-source and comes with a **Free License**. This module was created with the help of **Deeplearning.ai 's Deep Learning Program**. If you have any problems, you can contact me via my email, which is provided at the end of this document. For further assistance, we also have a YouTube channel - the link to which is also provided below.

# Install lsHotword using pip ✌
 To install lsHotword open cmd and type-
 ```
 pip install lsHotword
 ```
 make sure your python should be on path.

# Training Your Own Model 😊
## Create Dataset
To train your own Model you have to create your Dataset.
Record 10 audio with voice **Activate** and place it under "Positives folder" and record 10 **Non-Activate Word** Which are not Activate and place it under negatives folder.Finally, record two or more background noises in different environments, each of 10 seconds duration. Ensure that these audio files are recorded in a sample rate of 44100 Hz; otherwise, you will have to change too many parameters. Examples are provided on Github (from Coursera's Deep Learning Program).
Your directory structure should look like this:
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

Audio files info-

**positives/** -> 44100Hz, length may vary, but it should be under 10 seconds. Audio files should contain only the sound of the keyword (Hotword). See an example [Here.](https://github.com/HemantKArya/lsHotword/tree/main/Examples/data/positives)

**negetives/** -> 44100Hz, length may vary, but it should be under 10 seconds. Audio files should contain only the sound of the keyword opposite to the hotword. See an example [here.](https://github.com/HemantKArya/lsHotword/tree/main/Examples/data/negatives)

**background/** -> 44100Hz, lenght should exactly be 10 second. e.g [See here.](https://github.com/HemantKArya/lsHotword/tree/main/Examples/data/backgrounds)



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
and then you will see a text like **<< Waiting for Hotword >>** when you see this text then try to speak your wake word and see a chime sound will beep!!

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