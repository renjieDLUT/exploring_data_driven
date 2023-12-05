import torch.nn as nn
import torch

print("*" * 30, nn.TransformerEncoderLayer.__name__, "*" * 30)
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
src = torch.rand(10, 32, 512)
encoder_out = transformer_encoder_layer(src)
print(f"encoder_out.shape:{encoder_out.shape}")

print("*" * 30, nn.TransformerDecoderLayer.__name__, "*" * 30)
transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
tgt = torch.rand(10, 10, 512)
decoder_out = transformer_decoder_layer(tgt, encoder_out)
print(f"decoder_out.shape:{decoder_out.shape}")

print("*" * 30, nn.TransformerEncoder.__name__, "*" * 30)
transformer_encoder=nn.TransformerEncoder(transformer_encoder_layer,num_layers=6)
transformer_encoder_out=transformer_encoder(src)
print(f"transformer_encoder_out:{transformer_encoder_out.shape}")

print("*" * 30, nn.TransformerDecoder.__name__, "*" * 30)
transformer_decoder=nn.TransformerDecoder(transformer_decoder_layer,num_layers=6)
transformer_decoder_out=transformer_decoder(tgt,transformer_encoder_out)
print(f"transformer_decoder_out:{transformer_decoder_out.shape}")

print("*" * 30, nn.Transformer.__name__, "*" * 30)
transformer=nn.Transformer(d_model=512,nhead=16,num_decoder_layers=6,num_encoder_layers=6,dim_feedforward=2048,dropout=0.1,batch_first=True)
transformer_out=transformer(src,tgt)
print(f"transformer_out:{transformer_out.shape}")


