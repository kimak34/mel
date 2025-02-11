import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from numba import errors
#import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)
import pyttsx3
import speech_recognition as sr
import mel
model = mel.Mel(200, 512, 79, 20)
model.load("model")

r = sr.Recognizer()

tag = ""
while tag != "goodbye":
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=0.2)
        print("Please speak now:")
        print("Your prompt: ", end="")
        audio2 = r.listen(source2)
        prompt = r.recognize_whisper(audio2).lower()
        print(prompt)
    
    response, tag = model.query(prompt)
    print("Mel's answer: " + response)
    print()
    
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()