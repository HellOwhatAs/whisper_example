import whisper, json

# `whisper._MODELS`
model = whisper.load_model("medium", 'cuda', download_root='./models')
result = model.transcribe(
    "./example.m4a", 
    language='en',
    initial_prompt='Tyson Fury: If you want to blame me, blame me!'
)

with open("whisper_result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

for segment in result["segments"]:
    print("[%.2fs -> %.2fs] %s" % (segment["start"], segment["end"], segment["text"]))