from nav_env.risk.risk import RiskMetric
from nav_env.risk.collection import RiskCollection
from nav_env.environment.environment import NavigationEnvironment
from typing import Type
import time


# TODO: Find a better structure for the user to specify what arguments should be used for the risk calculation.
class RiskMonitor:
    def __init__(self, risk_classes:list[Type[RiskMetric]]=None, dt:float=1.):
        self._risk_classes = risk_classes or []
        self._dt = dt

    def monitor(self, shared_env_dict, results_queue) -> None:
        """
        Monitor the environment.
        """
        print("Start monitoring the environment")
        env = NavigationEnvironment() # Initial environment is wrong but we don't care, we will update it in the loop
        while True:
            start = time.time()
            env.from_dict(shared_env_dict)
            risks = RiskCollection([risk(env) for risk in self._risk_classes])
            # results = risks.calculate_separately(env.own_ships[0])
            results = []

            t0 = time.time()
            for ship in env.own_ships:
                val = risks.calculate_separately(ship)
                results.append(val)

            # print(f"({1000*(time.time()-t0):.2f}) Risk: {results}, Ship0: {env.own_ships[0].states}")

            results.insert(0, env.t)
            # print(f"({start-t0:.2f}) Risk: {results}, Ship0: {env.own_ships[0].states}")
            results_queue.put(results)
            # results = [env.t, risks._risks[0].calculate(env.own_ships[0]), risks._risks[1].calculate(env.own_ships[0])]
            # print(f"({start-t0:.2f}) Risk: {results}, Ship0: {env.own_ships[0].states}")
            # results_queue.put(results)
            stop = time.time()
            time.sleep(max(1e-9, self._dt - (stop - start)))

    def monitor_now(self, env:NavigationEnvironment) -> list:
        """
        Monitor the environment when it currently is
        """
        risks = RiskCollection([risk(env) for risk in self._risk_classes])
        results = []
        for ship in env.own_ships:
            val = risks.calculate_separately(ship) # returns a list of risk values
            results.append(val)
        return results


    @property
    def dt(self) -> float:
        return self._dt

    def __iter__(self):
        return iter(self._risk_classes)
    
    def __getitem__(self, index: int) -> Type[RiskMetric]:
        return self._risk_classes[index]
    
    def __len__(self) -> int:
        return len(self._risk_classes)
    
    def __repr__(self):
        return f"RiskMonitor({len(self._risk_classes)} risks)"
    
    def legend(self):
        return [risk.__name__ for risk in self._risk_classes]
    
