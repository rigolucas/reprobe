from reprobe import Monitor


def test_monitor_score():
    monitor = Monitor(None, [])
    
    #{layer: prob}
    monitor.history = [
        {0: 0.9, 1: 0.2},  # means = 0.55
        {0: 0.1, 1: 0.1},  # means = 0.1
    ]
    assert monitor.score("max_of_means", flush_buffer=False) == 0.55
    assert monitor.score("mean_of_means", flush_buffer=False) == 0.325
    assert monitor.score("max_absolute", flush_buffer=False) == 0.9