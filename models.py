import numpy as np 
import tensorflow as tf 
from keras.models import Model, Sequential
from keras.layers import Layer, Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, Add
from typing import Tuple

def positional_encoding(length: int, depth: int) -> tf.Tensor:
    depth        = depth/2
    positions    = np.arange(length)[:, np.newaxis]
    depths       = np.arange(depth)[np.newaxis, :]/depth
    angle_rates  = 1/(10000*depths)
    angle_rads   = positions*angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(Layer):

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(PositionalEmbedding, self).__init__()
        self.d_model      = d_model
        self.embedding    = Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs) -> tf.Tensor:
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        length = tf.shape(x)[1]
        x      = self.embedding(x)
        x     *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x      = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(Layer):

    def __init__(self, **kwargs) -> None:
        super(BaseAttention, self).__init__()
        self.mha       = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add       = Add()

class CrossAttention(BaseAttention):

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        self.last_attn_scores    = attn_scores
        x                        = self.add([x, attn_output])
        x                        = self.layernorm(x)
        return x
    

class GlobalSelfAttention(BaseAttention):

    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_output = self.mha(query=x, value=x, key=x)
        x           = self.add([x, attn_output])
        x           = self.layernorm(x)
        return x
    
class CausalSelfAttention(BaseAttention):

    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x           = self.add([x, attn_output])
        x           = self.layernorm(x)
        return x
    
class FeedForward(Layer):

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.seq        = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model),
            Dropout(dropout_rate)
        ])
        self.add        = Add()
        self.layer_norm = LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    
class EncoderLayer(Layer):

    def __init__(self, *, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn            = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(Layer):

    def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: float = 0.1) -> None:
        self(Encoder, self).__init__()
        self.d_model       = d_model
        self.num_layers    = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers    = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.dropout       = Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
    
class DecoderLayer(Layer):

    def __init__(self, *, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.cross_attention       = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn                   = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x                     = self.causal_self_attention(x)
        x                     = self.cross_attention(x, context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x                     = self.ffn(x)
        return x
    
class Decoder(Layer):

    def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: float = 0.1) -> None:
        super(Decoder, self).__init__()
        self.d_model          = d_model
        self.num_layers       = num_layers
        self.pos_embedding    = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout          = Dropout(dropout_rate)
        self.dec_layers       = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x
    
class Transformer(Model):

    def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int, target_vocab_size: int, dropout_rate: float = 0.1) -> None:
        super(Transformer, self).__init__()
        self.encoder     = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate)
        self.decoder     = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        context, x = inputs
        context    = self.encoder(context)
        x          = self.decoder(x, context)
        logits     = self.final_layer(x)
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits