import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

reference_speaker_wav = "audio.wav"

text_french = "Bonjour, ceci est un test de génération de voix en français à partir d'un texte en français."

output_path_fr = "output_from_fr_in_fr.wav"

print("\nInicializando o modelo XTTS v2...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"\nGerando áudio em francês a partir do texto em francês...")
    tts.tts_to_file(
        text=text_french,
        speaker_wav=reference_speaker_wav,
        language="fr",
        file_path=output_path_fr
    )
    print(f"Áudio gerado e salvo em: {output_path_fr}")

    print("\nProcesso concluído.")

except FileNotFoundError:
    print(f"\nErro: O arquivo de áudio de referência '{reference_speaker_wav}' não foi encontrado ou não pôde ser lido.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")
