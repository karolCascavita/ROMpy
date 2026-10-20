# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#
import os
import json
import shutil
import sys

from dolfin import *
from rbnics import *
from testcases import *
from models import *

import numpy as np

from rbnics.backends import copy

def rom_summary(outputdir, desired_N, Nu_avl, Np_avl, Ns_avl, Nu, Np, Ns):

    output_file = outputdir +"/rom_summary.text"

    with open(output_file, "w") as f:

        f.write("\n")
        f.write(f"Requested nummber of modes: {desired_N}\n")

        f.write("\n")
        f.write("Available POD modes:\n")
        f.write(f"  u velocity modes     : {Nu_avl}\n")
        f.write(f"  p pressure modes     : {Np_avl}\n")
        f.write(f"  s supremizer modes   : {Ns_avl}\n")

        f.write("\n")
        f.write("Selected POD modes:\n")
        f.write(f"  u velocity modes     : {Nu}\n")
        f.write(f"  p pressure modes     : {Np}\n")
        f.write(f"  s supremizer modes   : {Ns}\n")


def make_navier_stokes_unsteady(V, params, testcase): 

    model = params["fom"]["model"]

    if model not in MODELS:
        raise ValueError(f"Unknown FOM model: {model}")

    ModelClass = MODELS[model]

    return ModelClass( V,
                       parameters=params,
                       testcase=testcase,
                       subdomains=testcase.subdomains,
                       boundaries=testcase.boundaries,
                       mesh=testcase.mesh
                    )

def fom_state_at(nsu_fom, t):
    """Full mixed-space FOM state stored at time t (None before pod.offline())."""
    series = nsu_fom._solution_over_time
    assert series is not None, "FOM time history is not available: call after pod.offline()"
    times = np.asarray(series.stored_times())
    idx = int(np.argmin(np.abs(times - t)))
    assert abs(times[idx] - t) < 1e-8, f"Requested t={t}, nearest stored time={times[idx]}"
    return copy(series[idx]), idx
 
 
def online_to_numpy(x):
    """Numpy view of an RBniCS online vector / function."""
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    try:
        return np.asarray(x.vector()).ravel()
    except Exception:
        return np.asarray(x.content).ravel()
 
 
def seed_rom_from_fom_state(nsu_fom, nsu_rom, snapshot, t_start):
    """
    Make the ROM start at t_start from the given FOM state.
 
    - The reduced IC is projected only once, inside pod.offline(), when the FOM IC is
      still zero. We therefore write the snapshot IN PLACE into the object held by the
      truth IC storage and re-project (rebinding nsu_fom.ic_field_u would not reach it).
    - The ROM clock and its monitor time series are shifted to t_start. The monitor
      dict is shared with nsu_fom, so it is copied before being modified.
    Side effects: overwrites the saved reduced operator 'initial_condition_u', and any
    later FOM solve on nsu_fom would start from this state.
    """
    stored = nsu_fom.initial_condition["u"][0]
    stored.assign(snapshot)
    assert stored.vector().norm("l2") > 0., "IC write failed: stored truth IC is still zero"
 
    nsu_rom._build_reduced_initial_condition("offline")
    ic_rhs = online_to_numpy(nsu_rom.initial_condition["u"][0])
    assert np.linalg.norm(ic_rhs) > 0., "Reduced IC is still zero after re-projection"
 
    nsu_rom._time_stepping_parameters["initial_time"] = t_start
    monitor = dict(nsu_rom._time_stepping_parameters["monitor"])
    monitor["initial_time"] = t_start
    nsu_rom._time_stepping_parameters["monitor"] = monitor
    nsu_rom._init_time_series("online")
 
    expected = list(nsu_rom._solution_over_time.expected_times())
    print(f"ROM seeded at t={t_start}: expected times {expected[0]} .. {expected[-1]} ({len(expected)} states)")
    print("FOM monitor t0 (must be unchanged):", nsu_fom._time_stepping_parameters["monitor"]["initial_time"])
 


@CustomizeReducedProblemFor(NavierStokesUnsteadyProblem)
def CustomizeReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
    class ReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNavierStokesUnsteady_Base.__init__(self, truth_problem, **kwargs)
            self._time_stepping_parameters.update({
                "report": True,
                "integrator_type": "bdf2", #"beuler",   # numpy online backend only supports
                "nonlinear_solver": {
                    "report": True,
                    "line_search": "wolfe"
                }
            })

        def custom_output_rom(self):
            """Write forces_rom.csv; returns the rows as (t, Cd, Cl, kt, Div)."""

            times = self._solution_over_time.stored_times()
            truth = self.truth_problem
            out_path = os.path.join(truth.name(), "forces_rom.csv")
            rows = []

            with open(out_path, "w") as f:
                f.write("time Cd Cl kt Div\n")

                for (t, solution) in zip(times, self._solution_over_time):
                    N = solution.N                                   # per export_solution's pattern
                    reconstructed = self.basis_functions[:N] * solution   # Function on full truth V
                    u_N, ubar_N, p_N = split(reconstructed)           # same 3-way split as truth problem

                    truth.t = t
                    Cd, Cl = truth.compute_force_coefficients(u_N, p_N)
                    k = truth.kinetic_energy(u_N)
                    Div = assemble(div(u_N)**2 * truth.dx)
                    
                    rows.append((t, Cd, Cl, k, Div))
                    f.write(f"{t} {Cd} {Cl} {k} {Div}\n")
                    f.flush()

            print("ROM OUTPUT FINISHED:", out_path)
            return rows 

    return ReducedNavierStokesUnsteady

# ----------------------------------------------------------------------------
# I. Parameters
# ----------------------------------------------------------------------------
# 0. Upload parameters file
params_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.json" 
with open(params_file, "r") as f:
    params = json.load(f)

# ----------------------------------------------------------------------------
# II. Test case and finite element space (Taylor-Hood P2-P1 + supremizer)
# ----------------------------------------------------------------------------
# 1. Create case
testcase = CylinderFlowCase(params["fom"]["meshname"])
print("1. MESH and BC's done")

# 2. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", testcase.mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", testcase.mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_u, element_p)
V = FunctionSpace(testcase.mesh, element, components=[["u", "s"], "u_bar", "p"])

# ----------------------------------------------------------------------------
# III. Full order model
# ----------------------------------------------------------------------------
print("====================== III. FOM - START ==============================")

mu_fixed = params["physics"]["viscosity"]
mu_range = [(mu_fixed, mu_fixed)] # dummy variable
nsu_fom = make_navier_stokes_unsteady(V, params, testcase)
nsu_fom.set_mu_range(mu_range)
nsu_fom.set_time_step_size(params["fom"]["dt"])
nsu_fom.set_final_time(params["fom"]["t_final"])

#  copy config file to results folder
output_dir = nsu_fom.name()
shutil.copy(params_file, os.path.join(output_dir, "parameters.json"))

print("====================== III. FOM - END  ==============================")

# ----------------------------------------------------------------------------
# IV. Offline phase (POD-Galerkin)
# ----------------------------------------------------------------------------
print("=================== III. OFFLINE - START ===========================")

pod = PODGalerkin(nsu_fom)
pod.set_Nmax(params["rom"]["Nmax"], nested_POD=params["rom"]["Nmax"] )
pod.initialize_training_set(1)
PETScOptions.set("ts_bdf_order", "2") # Not available option in _time_stepping_parameters
nsu_rom = pod.offline()
nsu_fom.offline = False

nsu_fom.custom_output_fom()

print("=================== III. OFFLINE - END  ===========================")
# ----------------------------------------------------------------------------
# V. Seed the ROM with the FOM state at pod_time_start
# ----------------------------------------------------------------------------
print("====================== V. SEED - START  =============================")

# Extract the true FOM state at the start of the ROM training window
t_start_rom = params["rom"]["pod_time_start"]  
ic_full_field, idx_start = fom_state_at(nsu_fom, t_start_rom)

ke_snapshot = nsu_fom.kinetic_energy(split(ic_full_field)[0])
print(f"FOM snapshot: t={t_start_rom}, index={idx_start}, KE={ke_snapshot}")

seed_rom_from_fom_state(nsu_fom, nsu_rom, ic_full_field, t_start_rom)

print("====================== V. SEED - END  =============================")
# ----------------------------------------------------------------------------
# VI. Online phase
# ----------------------------------------------------------------------------
print("=================== VI. ONLINE - START  ===========================")
# online_mu = (1e-2, )
# nsu_rom.set_mu(online_mu)
Nu_available = nsu_rom.N["u"]
Np_available = nsu_rom.N["p"]
Ns_available = nsu_rom.N["s"]

N = params["rom"]["N"] #to be generalized

Nu = min(N, Nu_available)
Np = min(N, Np_available)    
Ns = min(N, Ns_available)


rom_summary(output_dir, N, Nu_available, Np_available, Ns_available, Nu, Np, Ns)
nsu_rom.solve(u = Nu, s=Ns, p=Np)
nsu_rom.export_solution(filename="online_solution" + str(Nu))
print("=================== VI. ONLINE - END  ===========================")

# ----------------------------------------------------------------------------
# VII. Sanity checks and ROM output
# ----------------------------------------------------------------------------
print("=================== VII. CHECKS - START  ===========================")
x0 = online_to_numpy(nsu_rom._solution_over_time[0])
print("first stored state N:", nsu_rom._solution_over_time[0].N)
print("lifting coefficient x0 (expect ~1, exactly 1 only if the snapshot lies in the basis span):", x0[0])
print("first stored time:", nsu_rom._solution_over_time.stored_times()[0])
 
rows = nsu_rom.custom_output_rom()
print(f"KE check at t={rows[0][0]}: ROM {rows[0][3]} vs FOM snapshot {ke_snapshot}")
print("Note: Cd/Cl in the first row use p=0 (pressure IC is not seeded); compare from the second row on.")
print("POD online done")
print("=================== VII. CHECKS - END  ===========================")

# ----------------------------------------------------------------------------
# 7. Perform an error analysis
#pod.initialize_testing_set(1)
#pod.error_analysis()

# 8. Perform a speedup analysis
#pod.speedup_analysis()
 