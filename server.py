from typing import Union

from fastapi import FastAPI, HTTPException, Response

import sys
import io
import numpy as np
import torch
from scipy.io.wavfile import write
from text import text_to_sequence

app = FastAPI()

tacotron2 = torch.load('./outdir/tacotron.pt')
tacotron2.cuda().eval().half()

sys.path.insert(0, './waveglow')
waveglow = torch.load('./waveglow/checkpoints/waveglow_256channels.pt')['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval().half()

@app.get("/")
def read_root(text: Union[str, None] = None):
    if text is None:
        raise HTTPException(status_code=400, detail="text parameter required")

    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    audio = audio * 32768.0
    audio = audio.squeeze().cpu().numpy().astype(np.int16)
    with io.BytesIO() as buf:
        write(buf, 22050, audio)
        im_bytes = buf.getvalue()
    return Response(im_bytes, media_type='audio/x-wav')


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
