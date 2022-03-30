from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor
import scipy.linalg as spla
import numpy as np
import matplotlib.pyplot as plt

# s = ( z + p) / (zp + 1) Bark a,d = 1, b,c = p
# s = p(z - 1) / ( z + 1) Tustin
# s = (az + b) / (cz + d) Moebius


def moebius_transform(sys, a, b, c, d, sampling_time=0):
    assert a * d != b * c

    if isinstance(sys, LTIModel):
        A = sys.A.matrix
        B = sys.B.matrix
        C = sys.C.matrix
        D = sys.D.matrix
        n = A.shape[0]
        I = np.eye(n)

        Gamma = spla.inv(a * I - c * A)
        v = np.sqrt(np.abs(a * d - b * c))
        sig = np.sign(a * d - b * c)

        Ad = Gamma @ (d * A - b * I)
        Bd = sig * v * Gamma @ B
        Cd = v * C @ Gamma
        Dd = D + c * C @ Gamma @ B

        return LTIModel.from_matrices(Ad, Bd, Cd, D=Dd, E=None, sampling_time=sampling_time)
    else:
        return (a*sys+b)/(c*sys+d)


def c2d(sys, sampling_time):
    return moebius_transform(sys, 2, -2, sampling_time, sampling_time, 1/fs)


def d2c(sys):
    return moebius_transform(sys, sys.sampling_time / 2, 1, -sys.sampling_time / 2, 1)


def calc_p(fs, method="peak"):
    if method=="peak":
        return 1.0674 * np.sqrt(2 / np.pi * np.arctan(0.00006583 * fs)) - 0.1916
    elif method=="slope":
        return 1.0480 * np.sqrt(2 / np.pi * np.arctan(0.00007212 * fs)) - 0.1957


def h2b(sys):
    p = calc_p(1/sys.sampling_time)
    return moebius_transform(sys, 1, p, p, 1, 1/48)


def b2h(sys):
    p = calc_p(fs)
    return moebius_transform(sys, 1, -p, -p, 1, 1/fs)


def a(w, fs):
    p = calc_p(fs)
    return np.angle(moebius_transform(np.exp(1j*w), 1, -p, -p, 1))


freqs = np.array([
    50,
    150,
    250,
    350,
    450,
    570,
    700,
    840,
    1000,
    1170,
    1370,
    1600,
    1850,
    2150,
    2500,
    2900,
    3400,
    4000,
    4800,
    5800,
    7000,
    8500,
    10500,
    13500,
])

# load
iss = LTIModel.from_mat_file("iss1R.mat")
#A, B, C, D, _ = iss.to_matrices()
#iss = LTIModel.from_matrices(A, B[:, 0].reshape(-1, 1), C[0].reshape(1, -1), D[0, 0].reshape(1, 1))

# params
r = 8
fs = 27000
scale = 50
numfreqs = 1000

# MOR
issh = c2d(iss, 1 / scale)
bth = BTReductor(issh)
r1h = bth.reduce(r, projection="sr")
issb = h2b(issh)
btb = BTReductor(issb)
r2b = btb.reduce(r, projection="sr")
r2h = b2h(r2b)

# freq computations
bfh = freqs/fs*2*np.pi
bfb = a(bfh, fs)
fb = bfb*48
bark = fb/np.pi/2
fd = np.geomspace(1/800, 1, numfreqs) * np.pi
fc = 2 * fd * scale
flim = np.array([20, 10000])
blim = a(flim/fs*2*np.pi, fs)*48/np.pi/2

plt.close("all")

plot_opts = {"Hz": True, "dB": True}
fig, ax = plt.subplots()
plt.vlines(freqs, ymin=-100, ymax=0, linestyle=":")
issh.transfer_function.mag_plot(fd, linestyle="-.", color="k", **plot_opts)
r1h.transfer_function.mag_plot(fd, color="b", **plot_opts)
r2h.transfer_function.mag_plot(fd, color="r", linestyle=":", **plot_opts)
plt.xlim(flim)
plt.ylim([-95, -20])
plt.savefig("fig1.png")

plot_opts = {"Hz": True, "dB": True}
fig, ax = plt.subplots()
plt.vlines(bark, ymin=-100, ymax=0, linestyle=":")
issb.transfer_function.mag_plot(fd, linestyle="-.", color="k", **plot_opts)
r2b.transfer_function.mag_plot(fd, color="b", **plot_opts)
ax.set_xscale('linear')
plt.xlabel("Frequency (Bark)")
plt.xlim(blim)
plt.ylim([-95, -20])
plt.savefig("fig2.png")
