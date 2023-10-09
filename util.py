TRUNCATE2, TRUNCATE5 = 100, 600
CHANNELS = [2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 5180, 5200, 5220, 5240, 5260, 5280, 5300,
            5320, 5745, 5765, 5785, 5805, 5825, 5845]
FEATURES = ['noise', 'max_magnitude', 'total_gain_db', 'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db']
STAT_FEATURES = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'mad']
LABELS = {'floor': 0, 'inter_mid': 1, 'inter_high': 2, 'jamming': 3}
