# 🐶 Classificador de Imagens: Qual é o Dog?
Este projeto é um classificador de imagens de cães que utiliza TensorFlow Lite para prever a raça do cão com base na imagem carregada. Ele foi implementado com uma interface amigável utilizando Tkinter, permitindo aos usuários carregar imagens e ver os resultados de predição de forma simples e interativa.

![Texto alternativo](/imagens/lulu_exemplo.png)

# ✨ Funcionalidades
Carregar imagens: O usuário pode selecionar imagens do seu computador para análise.
Redimensionamento dinâmico: As imagens são ajustadas para caber na interface e exibidas em um Canvas com barras de rolagem para facilitar a navegação.
Predição rápida: Com apenas um clique, o modelo retorna a raça do cão e o nível de confiança na predição.
Interface intuitiva: Desenvolvida com Tkinter, a interface é simples e funcional.

# 🛠️ Tecnologias e Bibliotecas Utilizadas
Python 3.7+
Tkinter: Para a criação da interface gráfica.
Pillow: Para manipulação e redimensionamento de imagens.
NumPy: Para processamento e normalização de dados.
tflite_runtime: Para carregar e executar modelos TensorFlow Lite de forma leve.

# 📁 Estrutura do Projeto

```
.
├── modelo.tflite         # Modelo treinado no formato TensorFlow Lite
├── labels.txt            # Rótulos das classes previstas pelo modelo
├── classificador.py      # Código principal do aplicativo
├── README.md             # Este arquivo
├── image.png             # Captura de tela da interface

```

# 🚀 Como Executar o Projeto
Siga os passos abaixo para executar o aplicativo em sua máquina:

1️⃣ Pré-requisitos

Certifique-se de que você tem Python 3.7+ instalado. Em seguida, instale as bibliotecas necessárias:

``` pip install tflite-runtime Pillow numpy  ```

2️⃣ Estrutura dos Arquivos
Certifique-se de que os seguintes arquivos estão na mesma pasta:

modelo.tflite: Modelo TensorFlow Lite treinado.
labels.txt: Arquivo de texto com os rótulos das classes (uma por linha).

3️⃣ Execute o Aplicativo
Basta rodar o script Python:

```python classificador.py``` 

Ou caso esteja utilizando o VS Code, apertar o play!

# 🧠 Como o Classificador Funciona?
Carregar Modelo e Rótulos:

O modelo modelo.tflite é carregado utilizando o tflite_runtime.
Os rótulos associados às classes do modelo são lidos a partir do arquivo labels.txt.
Preprocessamento da Imagem:

A imagem selecionada é convertida para RGB, redimensionada para o tamanho esperado pelo modelo e normalizada para valores entre 0 e 1.
Inferência:

A imagem preprocessada é passada para o modelo, que retorna as probabilidades para cada classe.
O rótulo com a maior probabilidade é exibido junto com o nível de confiança.

# 🐾 Exemplo de Execução
Carregue uma imagem: Clique no botão "Carregar Imagem" e selecione a imagem do seu cão.

Faça a predição: Clique no botão "Diz aí, qual o dog?" para ver a raça do cão e o nível de confiança.

# 🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.


