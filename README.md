# bert-applications:  Estudo IA Quinzenal 01 - BERT
LLM BERT architecture studies and real-world application

## I - BERT - Definição geral

**O que é o BERT?**:

Desenvolvido em 2018 pela Google, o BERT (Bidirectional Encoder Representations from Transformers) é uma arquitetura de Deep Neural Networks do tipo LLM (Large Language Models), sendo uma das primeiras LLM's criadas.

Teve origem a partir do artigo "*Attention is all you need*" de 2017 para realizar tarefas com NLP e revolucionou a área de NLP, pois no momento haviam RNN's e CNN's, porém eram obtinham parcial êxito no cumprimento das tarefas de NLP.

O que diferencia o BERT das redes neurais anteriores é a tecnologia de **transformadores**, pois antes utilizava-se uma arquitetura de *encoder-decoder*, cujo poder de processamento consumia muitos recursos e tempo, pois não permitia computação paralela.

> **O que são transformadores?**
>
> Transformadores são a arquitetura principal utilizada no BERT para gerar o contexto bidirecional (entende posterior e anterior) através de um **mecanismo de atenção** aperfeiçoado da arquitetura de **encoder e decoder** visto em modelos anteriores como RNN, CNN ou LSTM.
> Nos transformadores, além do encoding e decoding, foi adicionada uma área de estágios de pré-processamento, o que permite paralelizar o processo.
> 1. Arquitetura *encoding-decoding* (Utilizada em RNN's, CNN's, LSTM's...):
> ![image](https://hackmd.io/_uploads/HyGmvGl71g.png)
> Com essa arquitetura, torna-se impossível paralelizar a sua implementação devido à necessidade de passar pelas sentenças de forma ordinal termo a termo.
>
> 2. Arquitetura *Transformers* (BERT):
> ![image](https://hackmd.io/_uploads/SJIhdGlmke.png)
> Os estágios cruciais que permitiram a melhoria do modelo são:
>    - **Preprocessing Stages**: a criação dos **embeddings** para os inputs, termo a termo; e a computação do vetor posicional em cada termo nos inputs (que são as nossas frases). Esses estágios são realizados simultaneamente nas fontes (*sources*) e nos alvos (*targets*)
>    - **Embedding**: é o estágio de transformar variáveis categóricas e/ou vetores de altas dimensões (características com muitas dimensões) em vetores densos e contínuos e de menor dimensão. Ela favorece a achar, numericamente, a similaridade das palavras, portanto, vetores mais próximos têm maior similaridade.
>    - **Positional encodings**: cria um vetor de contexto para cada termo, o que permite encontrar o ciclo natural dos termos e suas relações.
>    - **Encoder Block**: esse bloco unifica o vetor de saída do embedding com o vetor de contexto, termo a termo.
>    - **Multi-head Attention**: essa camada de atenção de múltiplas "referências" busca todas as relações entre os termos, dando mais contexto entre os **termos do input** (pois no decoding é o masked). Ao final ela cria um novo vetor para cada termo, o Attention Vector.
>    - **Position-wise feed-forward net (FFN)**: Etapa de "ajuste" dos vetores dos termos para ser o esperado no **bloco decoder**. 
>    - **Decoder block**: consiste em 3 camadas mais importantes: **masked multi-head attention, multi-head attention e position-wise feed-forward network**, onde o FFN e Multi-head attention são o mesmo processo que no encoder block. Recebe 2 inputs: os **vetores de atenção do encoder block** e os **inputs target traduzidos**.
>    >    - **Masked multi-head attention layer**: é uma das 3 camadas contidas no decoder block. Funciona, como o nome já diz, como um mascaramento dos inputs posteriores (transformam os vetores em 0's) para forçar o aprendizado do modelo ao tentar prevê-los sequencialmente. Na prática, dessa forma, os vetores passarão pela etapa de multi-head attention como sendo iguais a zero, o que impede que gere-se viés de modelo e force o aprendizado real deste.

##  II - BERT - Arquitetura
Diferente das RNN's e CNN's que identificam os tokens de forma sequencial, e, portanto, prevêem as palavras seguintes, o BERT prevê as palavras anteriores e as seguintes, fornecendo **maior contexto** à interpretação do texto, por isso são chamados de arquiteturas **bidirecionais** (pra frente e pra trás).

> Em resumo, o BERT:
> - Baseia-se na arquitetura de encoding-decoding.
> - Entende o contexto de forma bidirecional (termos posteriores e anteriores, palavra a palavra).

As arquiteturas originais do BERT são:
1. **BERT Base**:
    *  Camadas: 12
    *  Tamanho das camadas escondidas: 768
    *  Attention heads: 12
    *  Número de parâmetros: 110M 
2. **BERT Large**:
    *  Camadas: 24
    *  Tamanho das camadas escondidas: 1024
    *  Attention heads: 16
    *  Número de parâmetros: 340M

> **OBS.:** Há diversas versões hoje em dia de BERT pré-treinado de acordo com cada área de atuação, tais como **bioBERT**, **RoBERTa**, dentre outros encontrados no [**Hugging Face**](https://huggingface.co/models?sort=trending&search=BERT).

## III - BERT - Aplicações
- **Resposta a perguntas**: foi um dos primeiros chatbots potencializados por transformadores, o que gerou resultados impressionantes.
- **Análise de sentimento**: identificar positividade ou negatividade de opiniões, tais como análise de notas clínicas, detecção de comentários mau intencionados, etc.
- **Geração de texto**: BERT possibilitou gerar textos longos com comandos simples, facilitando tarefas repetitivas como documentação (geração de relatórios, resumos e outros documentos baseados em dados), bem como geração de conteúdos para artigos, etc.
- **Sintetizar textos**: resume textos complexos e de difícil entendimento, poupando tempo para seus usuários.
- **Tarefas de autocompletude**: sugestões de preenchimento de palavras, serviços de mensagem ou e-mails.

## IV - Vantagens e desvantagens em relação ao RNN, CNN e LSTM (arquitetura anterior ao BERT)

1. **Vantagens:**
    - Complexidade computacional inferior, portanto menos oneroso na inferência.
    - Permite paralelização computacional (bidirecional e em blocos).
    - Permite trabalhar com grandes frases e textos por conta dos sinais de rede trabalhando bidirecionalmente.
3. **Desvantagens:**
    - Exige grandes quantidades de dados para o problema alvo.
    - Mais oneroso no treinamento devido à necessidade de utilizar GPU.

> **OBS.:** Tais desvantagens são eliminadas ao passo que utilizamos **TRANSFER LEARNING**, mais conhecido por **modelos pré-treinados**.

## V - BERT - Projetos pessoais
1. []()
2. []()

# Referências
1. [Datacamp - O que é o BERT? Introdução aos modelos BERT'0](https://www.datacamp.com/pt/blog/what-is-bert-an-intro-to-bert-models)
2. [Attention is all you need - BERT](https://arxiv.org/abs/1706.03762)
3. [An introduction to using transformers and hugging face](https://www.datacamp.com/pt/tutorial/an-introduction-to-using-transformers-and-hugging-face)
4. [BERT Training and Fine tuning](https://medium.com/@0192.mayuri/bert-training-and-fine-tuning-c49718d639ba)
