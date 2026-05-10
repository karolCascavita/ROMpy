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

from dolfin import *
from rbnics import *
from testcases import *
from models import *


def rom_summary(outputdir, desired_N, Nu_avl, Np_avl, Ns_avl, Nu, Ns, Np):

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


@CustomizeReducedProblemFor(NavierStokesUnsteadyProblem)
def CustomizeReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
    class ReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNavierStokesUnsteady_Base.__init__(self, truth_problem, **kwargs)
            self._time_stepping_parameters.update({
                "report": True,
                "nonlinear_solver": {
                    "report": True,
                    "line_search": "wolfe"
                }
            })
            
    return ReducedNavierStokesUnsteady


# 0. Upload parameters file
with open("parameters.json", "r") as f:
    params = json.load(f)

# 1. Create case
testcase = CylinderFlowCase(params["fom"]["meshname"])
print("1. MESH and BC's done")

# 2. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", testcase.mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", testcase.mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_u, element_p)
V = FunctionSpace(testcase.mesh, element, components=[["u", "s"], "u_bar", "p"])

# 3. Allocate an object of the NavierStokesUnsteady class
print("2. START fluid-dynamics solver")
mu_range = []
nsu_fom = make_navier_stokes_unsteady(V, params, testcase)
nsu_fom.set_mu_range(mu_range)
nsu_fom.set_time_step_size(params["fom"]["dt"])
nsu_fom.set_final_time(params["fom"]["t_final"])
print("2. END fluid-dynamics solver ")

#  copy config file to results folder
output_dir = nsu_fom.name()
shutil.copy("parameters.json", os.path.join(output_dir, "parameters.json"))

# 4. Prepare reduction with a POD-Galerkin method
pod = PODGalerkin(nsu_fom)
pod.set_Nmax(params["rom"]["Nmax"])
print("3: POD prepared")

# 5. Perform the offline phase
#lifting_mu = (1e-1, )
#nsu_fom.set_mu(lifting_mu)
pod.initialize_training_set(1)
nsu_rom = pod.offline()
nsu_fom.offline = False
print("4: POD offline done")

# 6. Perform an online solve
# online_mu = (1e-2, )
# nsu_rom.set_mu(online_mu)
Nu_available = nsu_rom.N["u"]
Np_available = nsu_rom.N["p"]
Ns_available = nsu_rom.N["s"]

N = params["rom"]["N"] #to be generalized

for j in [N]:

    Nu = min(j, Nu_available)
    Np = min(j, Np_available)    
    Ns = min(j, Ns_available)

    rom_summary(output_dir, N, Nu_available, Np_available, Ns_available, Nu, Np, Ns)

    delta_str = ""
    nsu_rom.solve(u = Nu, s=Ns, p=Np)
    nsu_rom.export_solution(filename="online_solution" + str(Nu) + delta_str)

print("5: POD online done")

# 7. Perform an error analysis
#pod.initialize_testing_set(1)
#pod.error_analysis()

# 8. Perform a speedup analysis
#pod.speedup_analysis()
