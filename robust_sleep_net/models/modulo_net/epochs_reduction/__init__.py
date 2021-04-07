from .attention_reducer import AttentionReducer
from .attention_reducer_with_reduction import AttentionReducer as AttentionWithReduction
from .pool_reducer import PoolReducer

epoch_reducers = {
    "Attention": AttentionReducer,
    "AttentionWithReduction": AttentionWithReduction,
    "Pooling": PoolReducer,
}
