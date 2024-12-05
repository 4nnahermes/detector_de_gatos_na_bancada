Discente: Anna Carolina Hermes

## Detector de Gatos na Bancada

Este projeto foi desenvolvido como parte da disciplina de Fundamentos de Inteligência Artificial, ministrada pelo docente Pablo Chiaro e tem como objetivo implementar um sistema de detecção de gatos posicionados em uma bancada.

Utilizando o modelo YOLOv4 em conjunto com a biblioteca OpenCV e desenvolvido em Python, o sistema emprega uma rede neural profunda (DNN) para identificar gatos de maneira eficiente e precisa. Este trabalho demonstra a aplicação de técnicas de inteligência artificial para resolver desafios específicos da visão computacional.

## Configuração do ambiente virtual

1. Criar o ambiente virtual

```bash
python -m venv env-visao  
```

2. Ativar o ambiente virtual:

No macOS e Linux:

```bash
source ./env-visao/bin/activate
```

No Windows:

```bash
source ./env-visao/bin/activate
```

## Instalação de dependências

Certifique-se de que seu ambiente virtual esteja ativado. Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.1.3
opencv-python==4.10.0.84
```

### Verificação da instalação

Para verificar se as bibliotecas foram instaladas corretamente, você pode executar o seguinte comando em um terminal Python:

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Links para download de modelos YOLO
Caso não encontre todos os arquivos listados abaixo na pasta raiz do projeto, faça o download pelos links fornecidos a seguir:

- Arquivo de configuração: [yolov4-tiny.cfg](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/cfg/yolov4-tiny.cfg)
- Arquivo de pesos: [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
- Arquivo de nomes das classes: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


## Rodar o script

Para executar o rastreio de pessoas, simplesmente execute o script `main.py` com Python. Certifique-se de que todos os arquivos necessários estão na mesma pasta que o script.

```bash
python main.py
```

## Encerrar o script

Pressionar 'q' para sair do aplicativo.

## Desativar o ambiente virtual

```bash
deactivate
````

## Alterações para caso de uso de outros vídeos

- Caso deseje utilizar outro vídeo para fazer a detecção, certificar-se que ele está na mesma pasta que o script e altere este trecho no script (substituindo "gato.mp4" pelo nome do seu vídeo):
```bash
caminho_video = "gato.mp4"
```
- Caso faça esta alteração de vídeo, o ROI_BANCADA deverá ser atualizado para a sua necessidade no script:
```bash
ROI_BANCADA = (90, 180, 250, 180)
```

## AVISO:

Nenhum felino sofreu maus tratos no vídeo, apenas encontra-se em confusão mental visto que não pode subir na bancada e foi colocado ali para o desenvolvimento da tarefa. Sofrimento apenas humano a partir de então.
