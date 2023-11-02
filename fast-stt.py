from faster_whisper import WhisperModel
from typing import List
import srt, datetime

# https://huggingface.co/guillaumekln/faster-whisper-large-v2
model_path = "./faster-whisper-large-v2/"
model = WhisperModel(model_path, device="cuda", compute_type='float16')

segments, info = model.transcribe(
    "D:/Downloads/红警对手飞机这么多基本成型！提前防空拦截，马上被说暗盟！_哔哩哔哩bilibili_红色警戒2_游戏集锦.m4a",
    language='zh',
    initial_prompt='红警对手飞机这么多基本成型！提前防空拦截，马上被说暗盟！'
)

subtitles: List[srt.Subtitle] = []
for idx, segment in enumerate(segments):
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    subtitles.append(srt.Subtitle(idx, datetime.timedelta(seconds=segment.start), datetime.timedelta(seconds=segment.end), segment.text))

with open("./fast_whisper_result.srt", 'w', encoding='utf-8') as f:
    f.write(srt.compose(subtitles))