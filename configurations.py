from lstm import LSTM
from reccurent import RecurrentLayer, BiRecurrentLayer

configs = (
    # Config 0
    (RecurrentLayer, {"conv_sz": 1, "nunits": 9}),

    # Config 1
    (RecurrentLayer, {"conv_sz": 3, "nunits": 9}),

    # Config 2
    (RecurrentLayer, {"conv_sz": 1, "nunits": 5,  "learn_init_state":False}),

    # Config 3
    (RecurrentLayer, {"conv_sz": 3, "nunits": 5}),

    # Config 4
    (LSTM, {"nunits": 9}),

    # Config 5
    (LSTM, {"nunits": 9, "forget": True}),

    # Config 6
    (LSTM, {"nunits": 9, "forget": True, "actvn_pre": "relu"}),

    # Config 7
    (LSTM, {"nunits": 9, "forget": False, "actvn_pre": "relu10"}),

    # Config 8
    (LSTM, {"nunits": 9, "forget": True,
            "actvn_post": "relu50", "actvn_pre": "relu10"}),

    # Config 9
    (LSTM, {"nunits": 9, "learn_init_states":False}),

    # Config 10
    (BiRecurrentLayer, {"nunits": 5}),
)
