# Perceptron_example_1
Este projeto implementa um **Perceptron** utilizando uma função de ativação hiperbólica e treinamento supervisionado. O objetivo é demonstrar como o algoritmo aprende a partir de um conjunto de dados e como ele pode prever saídas com base nas entradas.

## Índice

- [1. Introdução](#1-introdução)
- [2. Bibliotecas Utilizadas](#2-bibliotecas-utilizadas)
- [3. Estrutura do Código](#3-estrutura-do-código)
  - [3.1. Classe Perceptron](#31-classe-perceptron)
  - [3.2. Função de Ativação](#32-função-de-ativação)
  - [3.3. Treinamento do Modelo](#33-treinamento-do-modelo)
  - [3.4. Teste do Modelo](#34-teste-do-modelo)
- [4. Execução](#4-execução)
- [5. Conclusão](#5-conclusão)

## 1. Introdução

O **Perceptron** é um modelo básico de rede neural que realiza classificações. Neste projeto, uma função de ativação hiperbólica é utilizada, permitindo que as saídas sejam normalizadas entre -1 e 1. O objetivo é treinar o perceptron com um conjunto de dados simples e testar suas previsões.

## 2. Bibliotecas Utilizadas

O código utiliza as seguintes bibliotecas:
- **NumPy**: Para operações numéricas e manipulação de arrays.
- **Math**: Para funções matemáticas, como a função exponencial.

## 3. Estrutura do Código

### 3.1. Classe Perceptron

A classe **Perceptron** é definida para implementar o algoritmo. Ela contém os seguintes métodos:

- **`__init__(self, input_size, learning_rate=0.5, lamb=0.1)`**: Inicializa o perceptron com um tamanho de entrada especificado, uma taxa de aprendizado (`learning_rate`) e um parâmetro (`lamb`) que ajusta a função de ativação. Os pesos são inicializados aleatoriamente.

- **`predict(self, inputs)`**: Este método calcula a saída do perceptron para um conjunto de entradas. Ele utiliza a soma ponderada das entradas multiplicadas pelos pesos e aplica a função de ativação.

### 3.2. Função de Ativação

- **`activation(self, u)`**: Implementa a função de ativação hiperbólica, que normaliza a saída entre -1 e 1, definida como:

  \[
  f(u) = \frac{2}{1 + e^{-u}} - 1
  \]

### 3.3. Treinamento do Modelo

- **`train(self, training_data, targets, error_threshold=0.01, max_iterations=1000)`**: Este método treina o perceptron com os dados de treinamento. Para cada época, calcula-se o erro total e atualiza-se os pesos com base no erro de predição. O treinamento continua até que o erro total fique abaixo de um limite especificado ou até que um número máximo de iterações seja atingido.

  - A atualização dos pesos é feita com a seguinte fórmula:

  \[
  w_j = w_j + \eta \times \delta \times x_{ij}
  \]

  Onde \(\delta\) é calculado como:

  \[
  \delta = \frac{error \times (1 - output^2)}{2}
  \]

  O progresso do treinamento é exibido no console, mostrando o erro total e os pesos atuais a cada época.

### 3.4. Teste do Modelo

- **`test(self, test_data)`**: Este método avalia o perceptron em um conjunto de dados de teste e retorna as previsões para as entradas fornecidas.

## 4. Execução

Para executar o código, assegure-se de que a biblioteca NumPy esteja instalada. Em seguida, execute o script em um ambiente Python. O perceptron será treinado usando um conjunto de dados simples, e as previsões para os mesmos dados de entrada serão exibidas.

## 5. Conclusão

Este projeto fornece uma introdução ao algoritmo de perceptron com uma função de ativação hiperbólica. O modelo aprende a partir de um pequeno conjunto de dados, e sua capacidade de previsão é demonstrada através de testes. A implementação ilustra os conceitos fundamentais por trás do aprendizado supervisionado em redes neurais.




----



# Perceptron_example_2

Este projeto implementa um **Perceptron**, um modelo básico de aprendizado de máquina para classificação binária, utilizando o famoso dataset Iris. O objetivo é demonstrar como o algoritmo funciona e como visualizá-lo.

## Índice

- [1. Introdução](#1-introdução)
- [2. Bibliotecas Utilizadas](#2-bibliotecas-utilizadas)
- [3. Estrutura do Código](#3-estrutura-do-código)
  - [3.1. Classe Perceptron](#31-classe-perceptron)
  - [3.2. Treinamento do Modelo](#32-treinamento-do-modelo)
  - [3.3. Visualização dos Dados e Regiões de Decisão](#33-visualização-dos-dados-e-regiões-de-decisão)
  - [3.4. Gráfico de Erros](#34-gráfico-de-erros)
- [4. Execução](#4-execução)
- [5. Conclusão](#5-conclusão)

## 1. Introdução

O **Perceptron** é um tipo de neurônio artificial que realiza classificações binárias. Ele aprende a partir de exemplos de treinamento e ajusta seus pesos com base nos erros cometidos durante as previsões.

## 2. Bibliotecas Utilizadas

O código utiliza as seguintes bibliotecas:
- **NumPy**: Para operações numéricas e manipulação de arrays.
- **Pandas**: Para leitura e manipulação de dados do dataset Iris.
- **Matplotlib**: Para geração de gráficos e visualizações.

## 3. Estrutura do Código

### 3.1. Classe Perceptron

A classe **Perceptron** é definida para implementar o algoritmo. Ela contém os seguintes métodos:

- **`__init__(self, eta=0.01, n_iter=10)`**: Inicializa o perceptron com uma taxa de aprendizado (`eta`) e um número de iterações (`n_iter`).

- **`fit(self, X, y)`**: Este método treina o perceptron. Ele inicializa um vetor de pesos, que é ajustado a cada iteração com base no erro de predição. Os pesos são atualizados usando a fórmula:

  \[
  w_j = w_j + \eta \times (target - \text{predict}(x_i)) \times x_{ij}
  \]

- **`predict(self, X)`**: Realiza a predição das classes para as entradas fornecidas.

- **`net_input(self, X)`**: Calcula a soma ponderada dos inputs com os pesos.

### 3.2. Treinamento do Modelo

Após a definição da classe, o modelo é treinado utilizando o dataset Iris, que contém informações sobre flores de diferentes espécies. As duas classes de flores utilizadas são **Iris-setosa** e **Iris-versicolor**. Os dados são filtrados para incluir apenas essas duas classes.

### 3.3. Visualização dos Dados e Regiões de Decisão

Os dados são visualizados utilizando um gráfico de dispersão, onde as flores da classe **setosa** são representadas por círculos vermelhos e as da classe **versicolor** por cruzes azuis. Após o treinamento, as regiões de decisão do perceptron são plotadas no gráfico, permitindo visualizar como o modelo separa as duas classes.

### 3.4. Gráfico de Erros

Um gráfico é gerado para mostrar o número de classificações incorretas ao longo das épocas de treinamento. Isso ajuda a entender como o modelo está aprendendo e se está convergindo para uma solução.

## 4. Execução

Para executar o código, basta garantir que as bibliotecas necessárias estejam instaladas. Em seguida, execute o script em um ambiente Python que suporte visualização gráfica, como Jupyter Notebook ou um script Python.

## 5. Conclusão

Este projeto fornece uma introdução ao algoritmo de perceptron e como ele pode ser aplicado a um problema de classificação simples. A visualização das regiões de decisão e dos erros ao longo das épocas ajuda a entender melhor o processo de aprendizado do modelo.


ref => https://medium.com/@urapython.community/perceptron-com-python-uma-introdu%C3%A7%C3%A3o-f19aaf9e9b64
