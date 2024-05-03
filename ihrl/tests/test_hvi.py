from ihrl.hvi import val_true
from ihrl.taxicab import TaxiMDP, Root, taxi_state

def test_hvi_initial_value():
    layout_str = """
    A--B
    ----
    ----
    C--D 
    """
    mdp = TaxiMDP(layout_str)
    init_subtask = Root(mdp)
    init_state = taxi_state(0, 0, mdp.width-1, mdp.height-1)
    val = val_true(init_subtask, init_state, max_horizon=10)
    expected_value = 21 #??? TODO: CHECK
    assert val == expected_value