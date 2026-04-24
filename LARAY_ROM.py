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

from dolfin import *
from rbnics import *
from testcases import *
import json
import shutil
import os

@ExactParametrizedFunctions()
class NavierStokesUnsteady(NavierStokesUnsteadyProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NavierStokesUnsteadyProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs

        self.mesh = kwargs["mesh"]         
        self.subdomains = kwargs["subdomains"]
        self.boundaries = kwargs["boundaries"]
        self.parameters = kwargs["parameters"]    
        self.testcase   = kwargs["testcase"]

        self._solution.assign(self.testcase.InitialCondition(V))
        
        self.dup = TrialFunction(V)
        (self.du, self.dubar, self.dp) = split(self.dup)
        (self.u , self.ubar , self.p)  = split(self._solution)
        
        vq = TestFunction(V)
        (self.v, self.vbar, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.f = testcase.Forcing(V)
        self.g = testcase.g(V)
        
        self.hmin = self.mesh.hmin()
        self.delta = 2*self.hmin**2
        self.nu = self.parameters["physics"]["viscosity"]
        self.kolmogorov =  5.62e-4

        self.offline = True

        self._time_stepping_parameters.update({
            "monitor": {
                "initial_time":  self.parameters["fom"]["monitor"]["initial_time"],
                "time_step_size": self.parameters["fom"]["monitor"]["time_step_size"]
            }
        })

        self._time_stepping_parameters.update({
            "report": True,
            "snes_solver": {
                "linear_solver": "mumps",
                "maximum_iterations": 20,
                "report": True
            }
        })
             
    # Return custom problem name
    def name(self):
        testcase = getattr(self, "testcase", None)
        if testcase is None:
            dirname = "Laray_ROM"
        elif hasattr(self, "parameters") and self.parameters:
            dirname = "Laray_ROM_" + testcase.name()+"/" + self.parameters["output_dir"] 
        else:
            dirname = "Laray_ROM_" + testcase.name()

        os.makedirs(dirname, exist_ok=True)
        return dirname       


    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        if term == "a":
            theta_a0 = self.nu
            return (theta_a0,)
        elif term in ("b", "bt"):
            theta_b0 = 1.
            return (theta_b0,)
        elif term == "c":
            if self.offline:
                print("OFFLINE")
                theta_c0 = 1.
                theta_c1 = 1e-16
                theta_c2 = 1e-16
                theta_c3 = 1e-16
            else:
                print("ONLINE")
                theta_c0 = 1e-16
                theta_c1 = 1
                theta_c2 = self.delta
                theta_c3 = 1
            return (theta_c0,theta_c1, theta_c2, theta_c3)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0,)
        elif term == "m":
            theta_m0 = 1.
            return (theta_m0, )
        elif term == "dirichlet_bc_u":
            t = self.t
            theta_bc0 = 1. # 6/((0.41)*(0.41))
            return (theta_bc0,)
        elif term == "dirichlet_bc_ubar":
            theta_bc00 = 1.
            return (theta_bc00,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_derivatives
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        print(term)
        if term == "a":
            u = self.du
            v = self.v
            a0 = inner(grad(u), grad(v))*dx
            return (a0,)
        elif term == "b":
            u = self.du
            q = self.q
            b0 = - q*div(u)*dx
            return (b0,)
        elif term == "bt":
            p = self.dp
            v = self.v
            bt0 = - p*div(v)*dx
            return (bt0,)
        elif term == "c":
            u = self.u
            v = self.v
            ubar = self.ubar
            vbar = self.vbar
            c0  = inner(grad(u)*u, v)*dx
            c0l = inner(grad(ubar)*u, v)*dx
            cl1 = inner(grad(ubar), grad(vbar))*dx # + inner(ubar,vbar)*dx - inner(u, vbar)*dx
            cl2 = inner(ubar,vbar)*dx - inner(u, vbar)*dx
            return (c0,c0l, cl1, cl2)
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v)*dx
            return (f0,)
        elif term == "g":
            q = self.q
            g0 = self.g*q*dx
            return (g0,)
        elif term == "m":
            u = self.du
            v = self.v
            m0 = inner(u, v)*dx
            return (m0,)
        elif term == "filter_lhs":
            u = self.du
            v = self.v
            filter0 = inner(u, v)*dx
            filter1 = inner(grad(u), grad(v))*dx
            return (filter0, filter1)
        elif term == "filter_rhs":
            u = self.du
            v = self.v
            filter0 = inner(u, v)*dx
            return (filter0,)
        elif term == "dirichlet_bc_u":
            bc0 = self.testcase.BoundaryConditions(self.V)
            return (bc0,)
        elif term == "dirichlet_bc_ubar":
            bc0 = self.testcase.BoundaryConditionsUbar(self.V)
            return (bc0,)
        elif term == "inner_product_u":
            u = self.du
            v = self.v
            x0 = inner(grad(u), grad(v))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.dp
            q = self.q
            x0 = inner(p, q)*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
           
# Customize the resulting reduced problem

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

# 1 Create case
testcase = CylinderFlowCase("Michele")

# 2. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", testcase.mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", testcase.mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_u, element_p)
V = FunctionSpace(testcase.mesh, element, components=[["u", "s"], "u_bar", "p"])

# 3. Allocate an object of the NavierStokesUnsteady class
print("START fluid-dynamics solver")
navier_stokes_unsteady_problem = NavierStokesUnsteady( V, parameters= params,
                            testcase=testcase, 
                            subdomains=testcase.subdomains, 
                            boundaries=testcase.boundaries, 
                            mesh=testcase.mesh)
mu_range = []
navier_stokes_unsteady_problem.set_mu_range(mu_range)
navier_stokes_unsteady_problem.set_time_step_size(params["fom"]["dt"])
navier_stokes_unsteady_problem.set_final_time(params["fom"]["t_final"])
print("END fluid-dynamics solver ")
#  copy config file to results folder
output_dir = navier_stokes_unsteady_problem.name()
shutil.copy("parameters.json", os.path.join(output_dir, "parameters.json"))

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(navier_stokes_unsteady_problem)
pod_galerkin_method.set_Nmax(params["rom"]["Nmax"])
print(params["rom"]["Nmax"])


# 5. Perform the offline phase
#lifting_mu = (1e-1, )
#navier_stokes_unsteady_problem.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(1)
reduced_navier_stokes_unsteady_problem = pod_galerkin_method.offline()
navier_stokes_unsteady_problem.offline = False

# 6. Perform an online solve
# online_mu = (1e-2, )
# reduced_navier_stokes_unsteady_problem.set_mu(online_mu)
for j in [20]:
    k = j
    k_p = 20
    delta_str = ""
    reduced_navier_stokes_unsteady_problem.solve(u = k, s=k_p, p=k_p)
    reduced_navier_stokes_unsteady_problem.export_solution(filename="online_solution" + str(k) + delta_str)

