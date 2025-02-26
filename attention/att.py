import jax 
import jax.numpy as jnp
from jax import lax


@jax.jit
def attention_forward(k, q, v): 
    '''
    q : query vector
    k : key 
    v : value vector
    '''

    batch_size, num_heads, seq_len, head_size = q.shape

    k_T = jnp.swapaxes(k, -1, -2)
    attention_scores = jnp.matmul(q, k_T) * (1.0 / jnp.sqrt(head_size))

    attention_probs = jax.nn.softmax(attention_scores, axis=-1)

    return jnp.matmul(attention_probs, v)


# XLA optimized forward attention implementation
@jax.jit
def xla_attention_forward(q, k, v): 

    batch_size, num_heads, seq_len, head_size = q.shape
    
    scale = 1.0 / jnp.sqrt(head_size)

    k_T =  jnp.swapaxes(k, -1, -2)
    attention_scores =lax.dot_general(
        q.astype(jnp.float32), 
        k_T.astype(jnp.float32),
        dim_nums = ( 
        ((3,), (2,)),
        ((0, 1), (0, 1)) 
    ) * scale

    #---- Softmax implementation 

    max_scores = jnp.max(attention_scores, axis=-1, keepdims=True)

    exp_scores = jnp.exp(attention_scores  - max_scores)

    sum_scores = jnp.sum(exp_scores, axis=-1, keepdims=True)

    attention_probs = exp_scores / sum_scores 

    #---- Output attention_probs (or attention weights) @ v 


    output = lax.dot_general( 
        attention_probs.astype(jnp.float32),
        v.astype(jnp.float32), 
        dim_nums = (
        ((3,), (2,)),
        ((0, 1), (0, 1))
        ) 
    )

    return output