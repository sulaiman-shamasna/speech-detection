import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'vad_webrtc',
    version= '1.0',
    author= 'Sulaiman Shamasna',
    author_email= 'sulaiman.shamasna@gmail.com',
    url= 'https://github.com/sulaiman-shamasna/speech-detection-WebRTC/tree/main',
    description= 'WebRTC-Vad implementation for speech detection',
    packages=['vad_webrtc'],
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': [
            'live_prediction=vad_webrtc.live_prediction:main',
            'generate_plot=vad_webrtc.generate_plot:main'
        ]
    },
    install_requires=[
        'webrtcvad',
        'numpy',
        'matplotlib',
        'librosa',
        'sounddevice'
    ]

)