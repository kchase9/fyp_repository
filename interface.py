
import numpy as np
from transformers import pipeline
import gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load custom model but use OpenAI's tokenizer
model = WhisperForConditionalGeneration.from_pretrained("kchase9/whisper-tiny-creolese-finetuned")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# pipe = pipeline(
#     model="openai/whisper-medium"
# )

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer
)

def transcribe(audio):
    sr, audio_data = audio
    text = pipe(audio_data)["text"]  
    return text

input_audio = gr.Audio(
    sources=["upload"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)
demo = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources="upload"), 
    outputs="text",
    title="Whisper Small Creolese",
    description="Demo for Creolese speech recognition using a fine-tuned Whisper tiny model.",
)

if __name__ == "__main__":
    demo.launch()
