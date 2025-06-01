from risk_model.risk_model import RiskModel
import numpy as np
import matplotlib.pyplot as plt
import pathlib, os

# Specify Path to JSON file and load into the class.
config_name = "ship_config.json"
config_path = os.path.join(pathlib.Path(__file__).parent, config_name)
print(config_path)

risk_model = RiskModel(config_path)


case_example = 3 #1, 2, 3
""" 
    case 1: For getting risk for a single ttg value
    case 2: For getting total risk given a couple of ttg values like in case of a trajectory
    case 3: To display the Risk profile given a large enough timeframe until the decline phases off
"""

match case_example:
    case 1:
        # Risk value for just a time
        ttg = 20
        mode = risk_model.select_mso_mode(ttg)
        risk = risk_model.compute_total_risk(ttg, mode)
        print(mode, risk)

    case 2:
        # Gets total risk for a couple of TTGs - like risk for a trajectory
        ttgs = np.random.randint(0, 100, 50) #replace with TTGs of trajectory/path
        total_risk = 0

        for ttg in ttgs:
            mode = risk_model.select_mso_mode(ttg)
            risk = risk_model.compute_total_risk(ttg, mode)
            # print(mode, risk)
            total_risk += risk
            
        print(f"Total Risk: {total_risk}")

    case 3:
        # Risk value for a timeframe, it shows the risk profile across all time
        ttgs = np.linspace(0, 150, 300)
        risk_list = []

        for ttg in ttgs:
            # mode = risk_model.select_mso_mode(ttg)
            mode = "MEC"
            risk = risk_model.compute_total_risk(ttg, mode)
            # print(mode, risk)
            risk_list.append(risk)


        plt.plot(ttgs, risk_list)
        plt.ylabel('Risk')
        plt.xlabel('TTG (seconds)')
        plt.show()