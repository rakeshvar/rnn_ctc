from lstm import LSTM, BDLSTM
from reccurent import RecurrentLayer, BiRecurrentLayer

configs = (
    # Configurations
    # USAGE: train.py data.pkl config#
    # Add more configurations here. Or edit the existing ones.
    #
    # Serial number = Line number - 10
    (RecurrentLayer, {"nunits": 9}),
    (RecurrentLayer, {"nunits": 9, "conv_sz": 3}),
    (RecurrentLayer, {"nunits": 5, "learn_init_state":False}),
    (BiRecurrentLayer, {"nunits": 3}),
    (BiRecurrentLayer, {"nunits": 3, "conv_sz": 3}),
    (LSTM, {"nunits": 9}),
    (LSTM, {"nunits": 9, "forget": True}),
    (LSTM, {"nunits": 9, "actvn_pre": "relu10"}),
    (LSTM, {"nunits": 9, "forget": True, "actvn_post": "relu50", "actvn_pre": "relu10"}),
    (LSTM, {"nunits": 9, "learn_init_states": False}),
    (BDLSTM, {"nunits": 9}),
    (BDLSTM, {"nunits": 3}),
    (BDLSTM, {"nunits": 5}),
    (RecurrentLayer, {"nunits": 90}),
)
