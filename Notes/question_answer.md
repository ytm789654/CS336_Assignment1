### tokenization
![Problem (unicode1)](image.png)  
a) '\x00'  
b) __repr__() is a special function in a class to return a user defined string while directly call print(obj), the __str()__ is used  
c)  
![answer for Q_c](image-1.png)  
Seems chr(0) is \x00 in string but will print nothing when call function print.  

![Problem (unicode2)](image-2.png)  
a)  
![print for different utf](image-3.png)  
The output is more readable in utf-8 while the input is within ASCII.  
![mem used](image-4.png)  
The utf-8 will use less mem.  
b)Not all character is encoded into one byte. So decode byte by byte can cause crash like:  
![crash with simplified Chinese](image-5.png)  
c) b'\xc0\x80' this is valid in all three encoding format.  


### Transformer resource accounting
A transformer model including  
- embedding layer
  - weight is vocab_size * embedding_size(embedding_size = d_model), mem vocab_size * d * mem(float)
  - no matmul op
- num_layers * transformer_blocks, each block including:
    - RMS norm layer
      - weight is 1 * d, memory is d * mem(float)
      - no matmul op
    - multihead attention QKV with pos encoding
      - QKV mapping:
        - W_Q, W_K, W_V, W_O weight are all d_model * d_model,  mem 4 * d * d * mem(float) = $4d^2$mem(float)
        - total matmul $4 * 2 * n * d * d = 8nd^2$,
      - pos encoding:
        - mem is common used, zero.
        - we can treat the pos encoding as n * d/2 * 1 * 2 tensor batch mat d/2 * 2 * 2  tensor, reshape as nd/2 * 2 matmul 2 * d, matmul is $2nd^2$, performed to QK, so totally $4nd^2$, 
      - QKV attention:
        - no weight.
        - Q matmul K_T 2 * n * d * d, (QK_T) matmul V 2 * n * d * d, totally $4nd^2$
    - another norm layer, 
      - memory is d * mem(float)
      - no matmul op 
    - ff:
      - W1, W3 d * d_ff, W2 d_ff * d_m, total 3 * d * d_ff * mem(float) = $3d * d_{ff}$
      - matmul including W1 @ X, W3 @ X, this part 2 * 2 * n * d * d_ff, and W2 matmul, this part 2 * d_ff * d * n, totally $6nd * d_{ff}$
  - totally, one block need $(d + 4d^2 + d + 3d * d_{ff})mem(float) = (2d + 4d^2 + 3d*d_{ff})mem(float)$
  - totally, one block need $8nd^2 + 4nd^2 + 4nd^2 + 6nd * d_{ff} = 16nd^2 + 6nd * d_{ff}$FLOPs
- norm layer:
  - weight is 1 * d, memory is d * mem(float)
  - no matmul op
- output_embbeding layer:
  - weight is d * vocab_size, d * vocab_size * mem(float)
  - matmul 2 * n * d * vocab_size

so totally, a transformer LM with num_layers blocks will need:
- $(vocab_size * d + num\_layers * (d + 4d^2 + d + 3d * d_{ff}) + d + d * vocab_size) * mem(float) = (2 * vocab_size * d + num\_layers * (2d + 4d^2 + 3d * d_{ff}) + d)mem(float)$
- $(num\_layers * (16nd^2 + 6nd * d_{ff}) + 2 * n * d * vocab_size)FLOPs$

### cross entropy
$
P = -log\frac{exp(x_{target}- x_{max})}{\sum exp(x_a - x_{max})} = log\sum exp(x_a - x_{max}) - log(exp(x_{target} - x_{max})) = log\sum exp(x_a - x_{max}) + x_{max} - x_{target}
$