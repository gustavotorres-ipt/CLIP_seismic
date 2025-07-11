# CLIP para sísmica

## Dados

- Os dados estão disponíveis na pasta:
    - `datasets/Dados_CLIP_sismico` do Teams.

- Salve os arquivos:
    - `seismic_distilbert.pt`
    - `resnet18_text_encoder.pth`
    - `customized_clip.pth`

    na raiz do projeto.

- Baixe também o arquivo

    `imagens_e_legendas_sismicas_sinteticas.zip`
    e descompacte.

    - Dentro da pasta `classes_balanceadas` copie as pastas `captions` e `images` na raiz do projeto.

## Instalação

- Crie um ambiente virtual utilizando `conda`:
    - `conda create -n CLIP`

- Ative o ambiente utilizando o comando:
    - `conda activate CLIP`

- Instale os pacotes com o comando:
    - `pip install -r requirements.txt`

## Treinamento

### CLIP
- Para fazer o treinamento do modelo, rode o arquivo `clip_training.py`:
    - `python clip_training.py`

### Decoder
- EM CONSTRUÇÂO

## Teste

- Para testar a busca por imagens, rode o arquivo `search_image.py`:

    - `python search_image.py`

    - Em seguida digite uma **descrição da imagem que procura em inglês**.

    - **Exemplo**: "*an image with a fault tilting right.*"

- A busca por imagens também pode ser feita utilizando interface gráfica. Rode o arquivo `interface.py` seguindo as mesmas instruções do `search_image.py`.
