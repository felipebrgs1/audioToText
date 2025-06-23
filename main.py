import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Caminho para a pasta de áudios
audio_folder = "audio"
output_folder = "transcricoes"
os.makedirs(output_folder, exist_ok=True)

# Configuração do dispositivo e tipo de dado
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Percorre todos os arquivos de áudio na pasta
audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith(".mp3")]

for audio_file in audio_files:
    audio_path = os.path.join(audio_folder, audio_file)
    print(f"Transcrevendo: {audio_file}")
    result = pipe(audio_path, return_timestamps=True)
    texto = result["text"]
    # Salva a transcrição em um arquivo de texto
    nome_txt = os.path.splitext(audio_file)[0] + ".txt"
    with open(os.path.join(output_folder, nome_txt), "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"Transcrição salva em: {nome_txt}")

print("Transcrição concluída!")
