import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm

# Define the transformer model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1024):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved in state_dict)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and split into multiple heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads, then apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)  # Using GELU activation as in newer transformers
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class GenerativeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.init_weights()
        
    def init_weights(self):
        # Initialize embeddings and linear layers
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x, mask=None):
        # Get token embeddings and add positional encoding
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        # Final linear layer to get logits for next token prediction
        output = self.output_layer(x)
        
        return output
    
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=None, device='cpu'):
        self.eval()
        
        # Move start tokens to device
        current_tokens = start_tokens.to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Create masks for padded positions
                seq_len = current_tokens.size(1)
                mask = torch.ones((1, seq_len, seq_len), device=device).triu(1) == 0
                
                # Get predictions for next token
                logits = self(current_tokens, mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    # Get top k values and their indices
                    values, indices = torch.topk(next_token_logits, top_k)
                    # Zero out values not in top k
                    next_token_logits = torch.zeros_like(next_token_logits).scatter_(1, indices, values)
                
                # Apply softmax to convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to current tokens
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # Check if EOS token was generated
                if next_token.item() == 1:  # Assuming 1 is the EOS token
                    break
                    
        return current_tokens

# Text Dataset for training
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenized_texts = []
        
        for text in texts:
            # Tokenize and truncate to max_length
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_texts[idx], dtype=torch.long)

# Simple tokenizer implementation
class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2}
        self.idx_to_char = {0: '<PAD>', 1: '<EOS>', 2: '<UNK>'}
        self.vocab_size = 3  # Start with special tokens
    
    def fit(self, texts):
        # Build vocabulary from texts
        for text in texts:
            for char in text:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = self.vocab_size
                    self.idx_to_char[self.vocab_size] = char
                    self.vocab_size += 1
    
    def encode(self, text):
        # Convert text to token indices
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text] + [self.char_to_idx['<EOS>']]
    
    def decode(self, indices):
        # Convert token indices back to text
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices if idx not in [0, 1]])  # Skip PAD and EOS

# Collate function for batching with padding
def collate_batch(batch):
    batch_size = len(batch)
    max_length = max([len(item) for item in batch])
    
    # Initialize padded tensor
    padded_batch = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    # Fill in the tensor
    for i, item in enumerate(batch):
        padded_batch[i, :len(item)] = item
    
    # Create attention mask (1 for actual tokens, 0 for padding)
    mask = (padded_batch != 0).unsqueeze(1).repeat(1, max_length, 1)
    
    return padded_batch, mask

# Training function
def train_model(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as progress_bar:
            for batch_idx, (input_batch, mask) in enumerate(progress_bar):
                input_batch = input_batch.to(device)
                mask = mask.to(device)
                
                # Shift inputs and targets for next token prediction
                x = input_batch[:, :-1]
                y = input_batch[:, 1:]
                mask = mask[:, :-1, :-1]
                
                # Forward pass
                logits = model(x, mask)
                
                # Reshape for cross entropy loss
                logits = logits.reshape(-1, logits.size(-1))
                y = y.reshape(-1)
                
                # Calculate loss ignoring padding
                loss = criterion(logits, y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
                
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# Main function to run the training and generation
def main():
    parser = argparse.ArgumentParser(description='Train and generate text with a transformer model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate text')
    parser.add_argument('--model_path', type=str, default='generative_model.pt', help='Path to save/load model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='Prompt for text generation')
    parser.add_argument('--max_gen_length', type=int, default=200, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sample text data if no data file provided
    sample_texts = [
        "Once upon a time, there was a little girl who lived in a village near the forest.",
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
    ]
    
    # Load data from file if provided
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = sample_texts
        print("Using sample texts for training")
    
    # Create and fit tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create or load model
    if args.train or not os.path.exists(args.model_path):
        model = GenerativeTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_length,
            dropout=args.dropout
        ).to(device)
        
        print(f"Created new model with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        model = torch.load(args.model_path, map_location=device)
        print(f"Loaded model from {args.model_path}")
    
    # Train model if requested
    if args.train:
        # Create dataset and dataloader
        dataset = TextDataset(texts, tokenizer, args.max_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Train model
        print("Starting training...")
        train_model(model, dataloader, optimizer, criterion, device, args.epochs)
        
        # Save model
        torch.save(model, args.model_path)
        print(f"Model saved to {args.model_path}")
    
    # Generate text if requested
    if args.generate:
        # Encode prompt
        prompt_tokens = torch.tensor([tokenizer.encode(args.prompt)[:-1]], dtype=torch.long)  # Remove EOS
        
        print("\nGenerating text from prompt:", args.prompt)
        
        # Generate text
        generated_tokens = model.generate(
            prompt_tokens, 
            args.max_gen_length, 
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        # Decode and print
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main()