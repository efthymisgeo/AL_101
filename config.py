import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
# BASE_PATH = '/home/chris/Dropbox'
EMOBASE_PATH = os.path.join(BASE_PATH, 'AL_101/data/Emo-DB')
print(EMOBASE_PATH)
NL_FEATURE_PATH = '/home/chris/Dropbox/AL_101/extracted_features/utterance'