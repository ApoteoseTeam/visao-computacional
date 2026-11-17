# Reali — contagem de pessoas

Aplicativo Python com detecção ONNX, rastreamento, calibração de portão e exportação MP4.

## Instalação

Use Python 3.11 ou 3.12 de 64 bits. É necessário um ambiente gráfico para abrir a janela do OpenCV. A base usa CPU, sem exigir placa de vídeo dedicada; a velocidade depende do computador e da resolução do modelo.

```sh
python -m venv .venv
```

Ative o ambiente no Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```sh
source .venv/bin/activate
```

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app --help
python -m app
```

Os padrões são `videos/pessoas3.mp4` e `models/yolo11n_960.onnx`, relativos à pasta do projeto. Copie esses arquivos junto com o projeto. O modelo precisa ser YOLO de detecção COCO, ONNX, entrada NCHW float32, saída sem NMS; a classe pessoa é 0. Não há download automático.

Para escolher os arquivos e rodar de qualquer pasta:

```sh
python -m app --video /caminho/video.mp4 --model /caminho/modelo.onnx --provider cpu
```

Execute o módulo na pasta do projeto; fora dela, use o caminho absoluto para `app/main.py`. Caminhos fornecidos nas opções são relativos à pasta de execução. Também estão disponíveis `--config`, `--output-dir`, `--confidence` e `--display-width`.

## Aceleração opcional

`--provider auto` procura CUDA, DirectML ou CoreML entre os provedores instalados; sem aceleração utiliza CPU. Se a inicialização acelerada falhar, tenta CPU. Uma escolha explícita indisponível gera erro. A instalação padrão só inclui ONNX Runtime para CPU.

Para DirectML no Windows, use um ambiente compatível com o pacote:

```sh
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-directml
python -m app --provider directml
```

Para NVIDIA, substitua o runtime por `onnxruntime-gpu` e instale as bibliotecas CUDA/cuDNN exigidas por ele. Instale apenas uma distribuição do ONNX Runtime por ambiente. A presença de GPU não garante compatibilidade com o runtime.

## Uso

C: calibrar; arraste os quatro pontos. S: salvar. P: pausar. R: reiniciar. E: reprocessar e exportar. Q: sair.

A configuração guarda coordenadas normalizadas. A exportação usa MP4V; se FFmpeg estiver no PATH, também tenta gerar uma cópia H.264. FFmpeg é opcional.

`USBCamera` seleciona o backend padrão do sistema (`AUTO`). A interface principal processa arquivos de vídeo.

## Verificação

```sh
python -m unittest discover -s tests -p "test_portability.py"
```

Os demais scripts em `tests` são diagnósticos manuais e podem exigir câmera, modelos ou aceleração específicos. Windows, Linux e macOS dependem de Python, bibliotecas nativas e drivers compatíveis; não há garantia de funcionamento em todo computador.
