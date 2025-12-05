import whisper

model = whisper.load_model("small")
result = model.transcribe("test.wav")
print(result["text"])