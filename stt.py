import whisper, srt, datetime
from typing import List

# `whisper._MODELS`
model = whisper.load_model("medium", 'cuda', download_root='./models')
result = model.transcribe(
    "./example.m4a", 
    language='en',
    initial_prompt='Tyson Fury: If you want to blame me, blame me!'
)

subtitles: List[srt.Subtitle] = []
for idx, segment in enumerate(result["segments"]):
    print("[%.2fs -> %.2fs] %s" % (segment["start"], segment["end"], segment["text"]))
    subtitles.append(srt.Subtitle(idx, datetime.timedelta(seconds=segment["start"]), datetime.timedelta(seconds=segment["end"]), segment["text"]))

with open("./whisper_result.srt", 'w', encoding='utf-8') as f:
    f.write(srt.compose(subtitles))