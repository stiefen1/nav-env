from dataclasses import dataclass
import json, os, pathlib

PATH_TO_DEFAULT_JSON = os.path.join(pathlib.Path(__file__).parent, "default_ship_params.json")
PATH_TO_DEFAULT_NEW_JSON = os.path.join(pathlib.Path(__file__).parent, "new_ship_params.json")

@dataclass
class ShipPhysicalParams:
    """
    Base class for ship parameters.
    """
    help: dict
    inertia: dict
    mass_over_linear_friction_coefficient: dict
    nonlinear_friction_coefficient: dict
    added_mass_coefficient: dict
    dimensions: dict
    wind: dict
    water: dict

    @staticmethod
    def load_from_json(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
        return ShipPhysicalParams(**params)
    
    @staticmethod
    def default():
        return ShipPhysicalParams.load_from_json(PATH_TO_DEFAULT_NEW_JSON)


def test():
    print(ShipPhysicalParams.default())


if __name__ == "__main__":
    test()