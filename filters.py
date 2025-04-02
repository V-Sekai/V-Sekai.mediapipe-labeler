import math

class LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha: float) -> None:
        if alpha <= 0 or alpha > 1.0:
            raise ValueError(f"alpha ({alpha}) should be in (0.0, 1.0]")
        self.__alpha = alpha

    def __call__(self, value: float, timestamp: float = None, alpha: float = None) -> float:
        if alpha: self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self) -> float: return self.__y
    def lastFilteredValue(self) -> float: return self.__s
    def reset(self) -> None: self.__y = None

class OneEuroFilter:
    def __init__(self, freq: float, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(mincutoff))
        self.__dx = LowPassFilter(self.__alpha(dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff: float) -> float:
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float = None) -> float:
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        prev_x = self.__x.lastFilteredValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))
