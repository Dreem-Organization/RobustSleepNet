from .gru_sequence_encoder import GRUSequenceEncoder, LSTMSequenceEncoder
from .residual_sequence_encoder import ResidualGRUSequenceEncoder

sequence_encoders = {
    "GRUSequenceEncoder": GRUSequenceEncoder,
    "LSTMSequenceEncoder": LSTMSequenceEncoder,
    "ResidualGRUSequenceEncoder": ResidualGRUSequenceEncoder,
}
