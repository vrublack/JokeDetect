import speech_recognition as sr
import combined

def record_phrase():
    # Record Audio
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    # write audio to a WAV file
    wav_data = audio.get_wav_data()
    wav_fname = 'mic_recording.wav'
    with open(wav_fname, 'wb') as f:
        f.write(wav_data)

    print('Processing...' + '\n')

    # Speech recognition using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        transcribed = r.recognize_google(audio)
        print("Transcription: " + transcribed + '\n')

        word_pred = combined.word_predict(transcribed)
        audio_pred = combined.audio_predict(wav_fname)
        print('Word pred: {}, audio pred: {}'.format(word_pred, audio_pred))


    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == '__main__':
    while True:
        record_phrase()