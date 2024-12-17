"""
What we want to save:

States from the whole environment:
- Initial objects and shore
- States evolution of all the objects (own_ships, target_ships, obstacles)
- risk measured in the monitor


NO FOR NOW LET'S KEEP IT SIMPLE
- Just record states of own_ships and risk measured in the monitor


Ideal usage

env = NavigationEnvironment(*args, **kwargs)
sim = Simulator()
sim.run(tf=10, save_results=True)
sim.results.plot()

"""

from nav_env.environment.environment import NavigationEnvironment
from nav_env.risk.monitor import RiskMonitor
import time
import pandas as pd

class SimulationRecord:
    def __init__(self, env:NavigationEnvironment=None, monitor:RiskMonitor=None, path_to_existing_data:str=None):
        """
        We want to plot those data, so we need a vector for each of them
        ship1:
            states:
                x: [x0, x1, ..., xM],
                y: [y0, y1, ..., yM],
                psi: [psi0, psi1, ..., psiM],
                ...

            risk:
                r1: [r10, r11, ..., r1M],
                r2: [r20, r21, ..., r2M],
                ...
                rk: [rk0, rk1, ..., rkM]
        ship2:
            states:
                ...
            risk:
                ...
        ...
        shipN:
            states:
                ...
            risk:
                ...

        """

        self._monitor = monitor or RiskMonitor()
        self._env = env or NavigationEnvironment()
        self._own_ships_data = {}
        self._times = []

        if path_to_existing_data is not None:
            self.load(path_to_existing_data)
        else:
            self._init_record_from_env_and_monitor()
        
    def _init_record_from_env_and_monitor(self) -> None:
        risk_to_monitor:list[str] = self._monitor.legend()
        prev_ship_names = []
        for i, own_ship_i in enumerate(self._env.own_ships):            
            
            ### Ensure ships have different names in the record
            j = 1
            while own_ship_i.name in prev_ship_names:
                own_ship_i.name += str(i + j)
                j += 1

                if j>10:
                    raise NameError("Impossible to generate a ship name that does not already exist")
            prev_ship_names.append(own_ship_i.name)
            ### End name check

            # Build dict for OSi to store 'states' and 'risks'
            dict_for_own_ship_i = {own_ship_i.name:{'states':{}, 'risks':{}}}
            for risk in risk_to_monitor:
                dict_for_own_ship_i[own_ship_i.name]['risks'].update({risk:[]})
            for state in own_ship_i.states.keys:
                dict_for_own_ship_i[own_ship_i.name]['states'].update({state:[]})
            self._own_ships_data.update(dict_for_own_ship_i)            
        
        print(self._own_ships_data)

    def __call__(self) -> None:
        self._times.append(self._env.t)
        risks = self._monitor.monitor_now(self._env)
        for i, own_ship_i in enumerate(self._env.own_ships):
            for key in own_ship_i.states.keys:
                self._own_ships_data[own_ship_i.name]['states'][key].append(own_ship_i.states[key])
            for j, key in enumerate(self._monitor.legend()):
                self._own_ships_data[own_ship_i.name]['risks'][key].append(risks[i][j])


        ship_0_data = self._own_ships_data[self._env.own_ships[0].name]
        print(f"({self._times[-1]:.2f}) x: {ship_0_data['states']['x'][-1]:.2f} {list(ship_0_data['risks'].keys())[0]}: {list(ship_0_data['risks'].values())[0][-1]} ")

    def __getitem__(self, idx:str):
        return self._own_ships_data[idx]
    
    def load(self, path:str) -> None:
        # We don't need any a-priori information about the environment
        self._own_ships_data = {}

        # Read the CSV file into a DataFrame
        all_data_df = pd.read_csv(path, header=[0, 1, 2])

        # Extract times
        self._times = all_data_df[('times', 'times', 'times')].tolist()

        # Iterate over the top-level columns (ship names)
        for ship_name in all_data_df.columns.levels[0]:
            if ship_name == 'times':
                continue

            # Prepare dictionnary for a new ship
            self._own_ships_data.update({ship_name:{'states':None, 'risks':None}})

            ship_data = all_data_df[ship_name]

            # Extract states and risks
            states = ship_data['states']
            risks = ship_data['risks']

            # Populate the _own_ships_data attribute
            self._own_ships_data[ship_name]['states'] = states.to_dict(orient='list')
            self._own_ships_data[ship_name]['risks'] = risks.to_dict(orient='list')

       

    def save(self, path:str) -> None:
        all_data = []

        # Create a DataFrame for times
        times = pd.DataFrame({"times": self._times})
        times.columns = pd.MultiIndex.from_product([['times'], ['times'], times.columns])

        for ship_name, ship_data in self._own_ships_data.items():
            states = pd.DataFrame(ship_data['states'])
            risks = pd.DataFrame(ship_data['risks'])

            # Add a sub-category level to the columns
            states.columns = pd.MultiIndex.from_product([[ship_name], ['states'], states.columns])
            risks.columns = pd.MultiIndex.from_product([[ship_name], ['risks'], risks.columns])

            # Concatenate states and risks DataFrames
            data = pd.concat([states, risks], axis=1)

            all_data.append(data)

        # Concatenate all ship data into a single DataFrame
        all_data_df = pd.concat(all_data, axis=1)

        # Concatenate times DataFrame with all ship data
        final_df = pd.concat([times, all_data_df], axis=1)

        # Save to CSV
        final_df.to_csv(path, index=False)

    @property
    def times(self) -> list[float]:
        return self._times
    
