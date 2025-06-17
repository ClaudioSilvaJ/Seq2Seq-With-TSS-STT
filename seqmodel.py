import re
import time
import math
import random
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import matplotlib.pyplot as plt
import speech_recognition as sr
import time
import requests
import argparse
import time
import random
import speech_recognition as sr
import difflib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

def replace_unknown_words(sentence, lang):
    words = sentence.split()
    known_words = lang.word2index.keys()
    new_words = []
    for word in words:
        if word in known_words:
            new_words.append(word)
        else:
            closest = difflib.get_close_matches(word, known_words, n=1)
            if closest:
                new_words.append(closest[0])
            else:
                print(f"Palavra desconhecida ignorada: {word}")
    return ' '.join(new_words)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def prepare_dataset(path, reverse=False):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize(s) for s in line.split('\t')] for line in lines if len(line.split('\t')) >= 2]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
    return pairs

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.word2count = {}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def sentence_to_tensor(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split()] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)

def build_dataloader(pairs, input_lang, output_lang, batch_size=32):
    X, Y = [], []
    for inp, out in pairs:
        in_ids = [input_lang.word2index[w] for w in inp.split()] + [EOS_token]
        out_ids = [output_lang.word2index[w] for w in out.split()] + [EOS_token]
        X.append(in_ids[:MAX_LENGTH] + [0]*(MAX_LENGTH - len(in_ids)))
        Y.append(out_ids[:MAX_LENGTH] + [0]*(MAX_LENGTH - len(out_ids)))
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        return self.gru(x)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        score = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(2).unsqueeze(1)
        weights = F.softmax(score, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, hidden, target=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, device=device)
        outputs, attns = [], []

        for _ in range(MAX_LENGTH):
            embedded = self.embedding(decoder_input)
            context, attn_weights = self.attn(hidden.permute(1,0,2), encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
            out, hidden = self.gru(rnn_input, hidden)
            output = self.out(out)
            outputs.append(output)
            attns.append(attn_weights)
            if target is not None:
                decoder_input = target[:, _].unsqueeze(1)
            else:
                decoder_input = output.argmax(-1)

        outputs = torch.cat(outputs, dim=1)
        return F.log_softmax(outputs, dim=-1), hidden, torch.cat(attns, dim=1)


def train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        enc_opt.zero_grad(), dec_opt.zero_grad()

        enc_out, enc_hidden = encoder(x)
        dec_out, _, _ = decoder(enc_out, enc_hidden, y)

        loss = criterion(dec_out.view(-1, dec_out.size(-1)), y.view(-1))
        loss.backward()
        enc_opt.step(), dec_opt.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(encoder, decoder, dataloader, n_epochs=80, lr=1e-3):
    enc_opt = optim.Adam(encoder.parameters(), lr)
    dec_opt = optim.Adam(decoder.parameters(), lr)
    criterion = nn.NLLLoss()
    for epoch in range(1, n_epochs+1):
        loss = train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion)
        print(f"[{epoch}/{n_epochs}] Loss: {loss:.4f}")

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        sentence = replace_unknown_words(sentence, input_lang)
        input_tensor = sentence_to_tensor(input_lang, sentence)
        enc_out, enc_hidden = encoder(input_tensor)
        dec_out, _, _ = decoder(enc_out, enc_hidden)
        topi = dec_out.argmax(-1)
        words = []
        for idx in topi[0]:
            if idx.item() == EOS_token:
                break
            words.append(output_lang.index2word.get(idx.item(), '?'))
        return ' '.join(words)


def save_model(encoder, decoder, input_lang, output_lang, path="model.pth"):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'input_lang': input_lang,
        'output_lang': output_lang
    }, path)

def load_model(path="model.pth", hidden_size=128):
    checkpoint = torch.load(path, map_location=device)

    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, input_lang, output_lang

def translate_sentence(sentence, model_path="model.pth"):
    encoder, decoder, input_lang, output_lang = load_model(model_path)
    norm = normalize(sentence)
    return evaluate(encoder, decoder, norm, input_lang, output_lang)

def enviar_texto(texto, idioma):
    url = "http://localhost:8000/generate"
    payload = {
        "text": texto,
        "language": idioma
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True)
    args = parser.parse_args()

    if args.type == "test":
        encoder, decoder, input_lang, output_lang = load_model("model.pth")

        r = sr.Recognizer()
        mic = sr.Microphone()

        print("Ajustando ruído ambiente... Aguarde um momento.")
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
        print("Pronto! Diga algo em Português (ou diga 'sair' para encerrar).")

        while True:
            try:
                with mic as source:
                    audio = r.listen(source, timeout=5, phrase_time_limit=25)

                texto = r.recognize_google(audio, language='en-US')
                print(f"Você disse: {texto}")

                if texto.lower().strip() == "sair":
                    print("Encerrando...")
                    break

                norm = normalize(texto)
                traducao = evaluate(encoder, decoder, norm, input_lang, output_lang)
                print("Tradução:", traducao)
                enviar_texto(traducao, "fr")

            except sr.WaitTimeoutError:
                print("Nenhuma fala detectada. Tentando novamente...")
                continue
            except sr.UnknownValueError:
                print("Não entendi o que você disse. Tente novamente.")
                continue
            except sr.RequestError as e:
                print(f"Erro ao requisitar resultados da API do Google; {e}")
                time.sleep(2)
                continue
            except KeyboardInterrupt:
                print("\nEncerrando o programa.")
                break
            except Exception as e:
                print(f"Ocorreu um erro inesperado: {e}")
                break

        exit(0)
    pairs = prepare_dataset("data\\eng-fra.txt")
    input_lang = Lang("eng")
    output_lang = Lang("fran")

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    dataloader = build_dataloader(pairs, input_lang, output_lang)
    hidden_size = 128
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(encoder, decoder, dataloader, n_epochs=30)
    save_model(encoder, decoder, input_lang, output_lang)
    print("Model trained and saved.")

    for _ in range(5):
        pair = random.choice(pairs)
        print("> ", pair[0])
        print("= ", pair[1])
        print("< ", evaluate(encoder, decoder, pair[0], input_lang, output_lang))
        print()



