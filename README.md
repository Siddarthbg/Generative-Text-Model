# Generative-Text-Model

# Name : Siddarth B.G

# INTERN ID : CT08WU25

# DOMAIN: Artificial Intelligence

# DURATION: 8 WEEKSW

# MENTOR: NEELA SANTOSH

# DESCRIPION:
This project is a **Generative Transformer Model (GTM)** implemented in PyTorch, designed to perform character-level text generation using a custom-built transformer architecture. Inspired by the mechanisms behind modern language models like GPT, this script is a self-contained framework for training and generating text using attention-based sequence modeling.
At its core, the model is based on the **Transformer architecture**, which consists of components such as **multi-head self-attention**, **positional encoding**, and **feed-forward networks**. The attention mechanism allows the model to weigh relationships between different parts of a sequence, while the positional encoding ensures the model is aware of token order—a crucial factor in language understanding.
The model begins with token embedding, transforming characters into dense vectors. Then, positional encodings are added to these embeddings, helping the model distinguish between different positions in a sequence. The embeddings are processed through several layers of **Transformer blocks**, each consisting of attention and feed-forward sub-layers with residual connections and normalization for stability. The final layer produces logits used for predicting the next character.
Text is tokenized using a simple **character-level tokenizer** that assigns an index to each unique character. Special tokens like `<PAD>`, `<EOS>`, and `<UNK>` are included to handle padding, sequence termination, and unknown characters, respectively. The tokenizer includes methods for encoding raw text into tokens and decoding tokens back into readable text.
The **training loop** uses a custom `TextDataset` and `DataLoader`, with sequences padded to equal length in each batch. A mask is created to ignore padding during attention calculations. During training, sequences are shifted so that the model learns to predict the next character. The **CrossEntropyLoss** function is used for optimization, ignoring padding tokens. Gradient clipping is employed to stabilize training and prevent exploding gradients.
One of the notable features of this script is its ability to **generate text autoregressively**. Given a prompt, the model generates characters one at a time, appending each new character to the sequence and feeding it back into the model. It supports **top-k sampling**, where only the top-k most probable next tokens are considered during sampling, and **temperature scaling** to control randomness and creativity.
The script is also designed with **command-line flexibility**, allowing users to specify training or generation modes, model parameters, prompt input, and output behavior. It can use sample texts or load data from a user-provided file. Trained models can be saved and reused, making the system practical for experimentation and reuse.
Overall, this project offers a clear, modular implementation of a transformer-based generative model suitable for educational purposes, small-scale experiments, and as a foundation for more advanced NLP applications. It demonstrates key principles of modern language modeling, including sequence modeling, attention, autoregressive generation, and custom tokenization.

# OUTPUT:![Image](https://github.com/user-attachments/assets/421eb7f1-b0f2-42d8-84e5-bc7263252224)
