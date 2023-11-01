from faster_whisper import WhisperModel
import json

# https://huggingface.co/guillaumekln/faster-whisper-large-v2
model_path = "./faster-whisper-large-v2/"
model = WhisperModel(model_path, device="cuda", compute_type='float16')

segments, info = model.transcribe(
    "./example.m4a",
    language='en',
    initial_prompt='Tyson Fury: If you want to blame me, blame me!'
)

result = {"language": info.language, "segments": [], "text": ""}

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    list.append(result["segments"], {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })
result["text"] = ' '.join(segment["text"] for segment in result["segments"])

with open("fast_whisper_result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)