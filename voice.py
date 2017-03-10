import speech_recognition as sr
#import combined

prev_wav = []
prev_transcription = []


def voice_callback(recognizer, audio):
    # write audio to a WAV file
    wav_data = audio.get_wav_data()
    prev_wav.append(wav_data)
    if len(prev_wav) >= 3:
        combined_wav = prev_wav[-3] + prev_wav[-2][40:] + prev_wav[-1][40:]
        with open("mic_recording.wav", "wb") as f:
            f.write(combined_wav)

    print('Processing...')

    # Speech recognition using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        transcribed = r.recognize_google(audio)
        prev_transcription.append(transcribed)
        if len(prev_transcription) >= 3:
            combined = prev_transcription[-3] + prev_transcription[-2] + prev_transcription[-1]
            print("Transcription: " + combined)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


if __name__ == '__main__':
    # Record Audio
    r = sr.Recognizer()
    source = sr.Microphone()
    print("Say something!")
    stop_listening = r.listen_in_background(source, voice_callback)

    input('Listening... Press any key to stop')
