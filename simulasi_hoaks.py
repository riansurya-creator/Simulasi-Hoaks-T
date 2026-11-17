"""
Pemodelan dan Simulasi Penyebaran Hoaks di Media Sosial (Model S-I-R)
S = Susceptible (belum terpapar hoaks)
I = Infected/Spreader (aktif menyebarkan hoaks)
R = Recovered (sadar / berhenti menyebarkan)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ---------------- Parameter ----------------
N     = 10000      # total pengguna
beta  = 6e-5       # laju kontak-penyebaran (1/(pengguna*hari))
gamma = 0.2        # laju berhenti menyebar (1/hari) -> rata-rata 5 hari
u     = 0.0        # strategi kontrol (literasi digital), 0 <= u <= 1

S0, I0, R0_ = 9990, 10, 0
x0 = [S0, I0, R0_]

# ---------------- Model ----------------
def hoaks(x, t, u):
    S, I, R = x
    dS = -(1 - u) * beta * S * I
    dI =  (1 - u) * beta * S * I - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

t = np.linspace(0, 120, 1201)

# ---------------- Simulasi 1: tanpa kontrol ----------------
x = odeint(hoaks, x0, t, args=(0.0,))
S, I, R = x[:, 0], x[:, 1], x[:, 2]

plt.figure(figsize=(9, 5))
plt.plot(t, S, 'b-', label='S (belum terpapar)')
plt.plot(t, I, 'r-', label='I (penyebar aktif)')
plt.plot(t, R, 'g-', label='R (berhenti menyebar)')
plt.xlabel('Waktu (hari)'); plt.ylabel('Jumlah pengguna')
plt.title('Penyebaran Hoaks Tanpa Kontrol (u = 0)')
plt.grid(alpha=0.3); plt.legend()
plt.savefig('grafik1_tanpa_kontrol.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- Simulasi 2: variasi kontrol u ----------------
plt.figure(figsize=(9, 5))
for uu, st in [(0.0, 'r-'), (0.3, 'b--'), (0.5, 'g-.'), (0.7, 'm:')]:
    xx = odeint(hoaks, x0, t, args=(uu,))
    plt.plot(t, xx[:, 1], st, linewidth=2, label=f'u = {uu}')
plt.xlabel('Waktu (hari)'); plt.ylabel('Penyebar aktif I(t)')
plt.title('Pengaruh Literasi Digital (u) terhadap Kurva Penyebar')
plt.grid(alpha=0.3); plt.legend()
plt.savefig('grafik2_variasi_kontrol.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- Perhitungan manual: metode Euler ----------------
dt = 1.0
S_e, I_e, R_e = [S0], [I0], [R0_]
for k in range(5):
    dS, dI, dR = hoaks([S_e[-1], I_e[-1], R_e[-1]], 0, 0.0)
    S_e.append(S_e[-1] + dt * dS)
    I_e.append(I_e[-1] + dt * dI)
    R_e.append(R_e[-1] + dt * dR)

print("hari |   S      |   I      |   R")
for k in range(6):
    print(f"{k:4d} | {S_e[k]:8.2f} | {I_e[k]:8.2f} | {R_e[k]:8.2f}")

# Pembanding odeint pada hari ke-1..5
xo = odeint(hoaks, x0, np.arange(0, 6), args=(0.0,))
print("\nodeint:")
for k in range(6):
    print(f"{k:4d} | {xo[k,0]:8.2f} | {xo[k,1]:8.2f} | {xo[k,2]:8.2f}")

# ---------------- Grafik 3: Euler vs odeint ----------------
plt.figure(figsize=(9, 5))
plt.plot(np.arange(0, 6), I_e, 'ro--', label='I Euler (dt = 1 hari)')
plt.plot(np.arange(0, 6), xo[:, 1], 'b-', label='I odeint')
plt.xlabel('Waktu (hari)'); plt.ylabel('Penyebar aktif I(t)')
plt.title('Perbandingan Perhitungan Manual (Euler) dan odeint')
plt.grid(alpha=0.3); plt.legend()
plt.savefig('grafik3_euler_vs_odeint.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------- Ringkasan ----------------
R0_basic = beta * N / gamma
print(f"\nR0 = beta*N/gamma = {R0_basic:.2f}")
for uu in [0.0, 0.3, 0.5, 0.7]:
    xx = odeint(hoaks, x0, t, args=(uu,))
    ip = xx[:, 1].argmax()
    print(f"u={uu}: puncak I={xx[ip,1]:.0f} pada hari {t[ip]:.1f}; "
          f"total pernah menyebarkan (R akhir)={xx[-1,2]:.0f}; S akhir={xx[-1,0]:.0f}")
