from dataclasses import dataclass
import json

@dataclass
class ShipPhysicalParams:
    """
    Base class for ship parameters.
    """
    mass: float
    inertia: float
    length: float
    width: float
    xg: float
    yg: float
    xw: float
    yw: float
    xg_prime: float
    yg_prime: float
    xw_prime: float
    yw_prime: float

    @staticmethod
    def load_from_json(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
        return ShipPhysicalParams(**params)

def test():
    # Load the ship parameters from json file
    import os, pathlib
    path_to_json = os.path.join(pathlib.Path(__file__).parent, "ship_params.json")
    with open(path_to_json, "r") as f:
        ship_params = ShipPhysicalParams.load_from_json(path_to_json)
    print(ship_params)

if __name__ == "__main__":
    test()