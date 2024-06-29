import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from io import StringIO
from scipy.optimize import linprog
from decimal import Decimal
from numpy import linalg as LA
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import cdd as pcdd
import time
from scipy.optimize import fsolve
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from scipy.integrate import solve_ivp
from itertools import cycle
import matplotlib.cm as cm
from ddeint import ddeint
from scipy.signal import find_peaks
import matplotlib.colors as colors
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib as mpl
from scipy.optimize import curve_fit
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence



####### 1 NALOGA #########


np.random.seed()


n = 0
N=0

for i in range(10**7):
    x = np.random.uniform()
    y = np.random.uniform()
    z = np.random.uniform()

    if (np.sqrt(x) +np.sqrt(y) + np.sqrt(z) <=1):
        n +=1
    N += 1

print(8*n/N)
