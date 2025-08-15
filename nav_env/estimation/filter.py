from abc import ABC, abstractmethod
from scipy.signal import firwin

class FIR1D(ABC):
    def __init__(self, *args, order:int=1, init:list|float=None, coeffs:list=None, **kwargs):
        """
        Finite Impulse Response Filter base class (1D)

        coeffs:list[float] - [beta0, beta1, ..., betaN, alpha]

        The output is computed as y_k+1 = beta0*y_k + beta1*y_k-1 + ... + betaN*y_k-N+1 + alpha*u_k
        """
        if coeffs is None:
            self.init_coeffs(*args, order=order, **kwargs)
        if init is None:
            self.prev = len(self.coeffs) * [0.0]
        elif isinstance(init, float) or isinstance(init, int):
            self.prev = len(self.coeffs) * [init]
        elif isinstance(init, list):
            self.prev = init
        else:
            raise TypeError(f"init must be either None, float or list, not {type(init).__name__}")
        
        print("PREV: ", self.prev)

    def __call__(self, new:float, *args, **kwargs) -> float:
        self.prev.insert(0, new) # add u_k at the end
        out = sum([beta*prev for beta, prev in zip(self.coeffs, self.prev)])
        self.prev.pop(-1) # remove last value
        return out
    
    @abstractmethod
    def init_coeffs(self, *args, **kwargs) -> None:
        self.coeffs = []
        
class LowPass(FIR1D):
    def __init__(self, *args, cutoff:float=None, init:list|float=None, coeffs:list=None, sampling_frequency:float=None, order:int=1, **kwargs):
        """
        To be used for filtering raw sensor measurement before feeding it to a state estimator
        """
        super().__init__(*args, cutoff=cutoff, sampling_frequency=sampling_frequency, order=order, init=init, coeffs=coeffs, **kwargs)

    def init_coeffs(self, cutoff:float, sampling_frequency:float, *args, order:int=1, **kwargs) -> None:
        print(cutoff, sampling_frequency, order)
        self.coeffs = tuple(firwin(numtaps=order+1, cutoff=cutoff, fs=sampling_frequency).tolist())

class HighPass(FIR1D):
    def __init__(self, *args, **kwargs):
        pass

class BandPass(FIR1D):
    def __init__(self, *args, **kwargs):
        pass

class BandStop(FIR1D):
    def __init__(self, *args, **kwargs):
        pass

def test() -> None:
    import numpy as np, matplotlib.pyplot as plt
    lp = LowPass(cutoff=1e-1, sampling_frequency=1e0, order=20, init=1.0)
    print(lp.coeffs, lp.prev)
    
    x0 = np.random.random(size=1000).tolist()
    x, y = [], []
    for i, xi in enumerate(x0):
        if i > 300:
            xi += 3
        x.append(xi)
        y.append(lp(xi))

    plt.plot(x)
    plt.plot(y)
    plt.show()

if __name__=="__main__":
    test()