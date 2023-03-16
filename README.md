# Tortoise-TTS Local Installation Guide

#### Step 1: Setup Miniconda
##### Follow the installation guide for your specific system:

[Windows](https://conda.io/projects/conda/en/stable/user-guide/install/windows.html)
[Mac](https://conda.io/projects/conda/en/stable/user-guide/install/macos.html)
[Linux](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html)

##### Verify that your installation was successful by typing:
```
conda list
```

#### Step 2: Create Conda environment
```
conda create -n tortoise python=3.8
conda activate tortoise
```

#### Step 3: Install PyTorch
##### Install PyTorch on your local computer using the following instructions (use Conda as package):
https://pytorch.org/get-started/locally/

##### Check if PyTorch is using your GPU:
```
python

>>> import torch

>>> torch.cuda.is_available()
True
```

#### Step 4: Setup the Tortoise-TTS model
```
git clone https://github.com/neonbjb/tortoise-tts

cd tortoise-tts

pip install numpy==1.23.0

python setup.py install
```


#### Step 5: Generate speech
```
# tts.py
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

# This is the text that will be spoken.
text = "Thanks for reading this article. I hope you learned something."

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "fast"

CUSTOM_VOICE_NAME = "tom"

voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)

gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)

torchaudio.save(f'generated-{CUSTOM_VOICE_NAME}.wav', gen.squeeze(0).cpu(), 24000)
```


#### Step 6: Get creative & have fun!
