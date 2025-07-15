import glob

import joblib
import matplotlib.pyplot as plt
from astropy import table
from astropy.io import fits
from laspec.ccf import RVM
from laspec.mrs import MrsSpec
from laspec.wavelength import vac2air
from tqdm import trange

# load catalog
l12_m31_g3_pm = table.Table.read("m31/lamost/l12_m31_gedr3_pm0.5.fits")


# load RVM for LRS
rvm_data = joblib.load("m31/RVMDATA_R1800_L7.joblib")
rvm_data["wave_mod"] = vac2air(rvm_data["wave_mod"])
rvm = RVM(**rvm_data)

# glob files
fl = glob.glob("m31/lamost_spec/*")
print(len(fl))

# plot spectra
for i in range(len(fl)):
    # f = fl[1]
    spec = MrsSpec.from_lrs(fl[i])
    spec.plot_norm(shift=i * 0.5)

plt.xlabel("Wavelength[A]")
plt.ylabel("Normalized Flux + Offset")

plt.close("all")
# measure RV
WAVE_BOUNDS = (4000, 7000)

rvm.make_cache(cache_name="M31", wave_range=(4000, 7000), rv_grid=(-1000, 500, 10))
res = []
for i in trange(len(fl)):
    spec = MrsSpec.from_lrs(fl[i])
    idx = (spec.wave > WAVE_BOUNDS[0]) & (spec.wave < WAVE_BOUNDS[1])
    this_res = rvm.measure(spec.wave[idx], spec.flux_norm[idx], rv_grid=(-1000, 500, 10),
                           flux_err=spec.flux_norm_err[idx], method="nelder-mead", nmc=100, flux_bounds=(0.3, 1.7),
                           cache_name="M31")
    this_res["obsid"] = fits.getval(fl[i], keyword="OBSID")
    res.append(this_res)

# save RV results
trv = table.Table(res)
trv.write("m31/lamost/rv.fits", overwrite=True)

# join catalog with RV
t = table.Table.read("m31/lamost/l12_m31_gedr3_pm0.5.fits")
t = table.join(t, trv, keys="obsid")

# compare with LAMOST redshift `z`
_lim = (-800, 200)
fig, ax = plt.subplots(1,1,figsize=(6, 5))
ax.plot(_lim, _lim, "k--")
s = ax.scatter(t["z"]*299792.458, t["rv_opt"], c=t["ccfmax"], s=40, edgecolor="k", marker="o", cmap="hot_r", vmin=0, vmax=0.8, alpha=.8)
ax.set_xlim(_lim)
ax.set_ylim(_lim)
ax.set_xlabel("LAMOST redshift z*c [km/s]")
ax.set_ylabel("RV in this work [km/s]")
fig.colorbar(s, ax=ax, label="CCF max")
fig.tight_layout()
fig.savefig("m31/figs/rv_z_comparison.pdf", dpi=300)
fig.savefig("m31/figs/rv_z_comparison.png", dpi=300)

# save table
t.write("m31/lamost/l12_m31_gedr3_pm0.5_rv_v2.fits", overwrite=True)
t.remove_columns(["pmod", "rv_pct"])
t.write("m31/lamost/l12_m31_gedr3_pm0.5_rv_v2.csv", overwrite=True)


# TOPCAT astronomy
plt.plot(t["ra"], t["rv_opt"], "o")
plt.scatter(t["ra"], t["rv_opt"], s=t["ccfmax"]*100, marker="o")
plt.scatter(t["ra"], t["z"]*299792.458, s=t["ccfmax"]*100, marker="o")
plt.close("all")