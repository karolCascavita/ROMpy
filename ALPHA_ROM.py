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
        self.testcase   = kwargs["testcase"]

        self._solution.assign(self.testcase.InitialCondition(V))

        self.dup = TrialFunction(V)
        (self.du, self.dubar, self.dp) = split(self.dup)
        (self.u, self.ubar, self.p) = split(self._solution)
        
        vq = TestFunction(V)
        (self.v, self.vbar, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.f = testcase.Forcing(V)
        self.g = testcase.g(V)

        self.hmin = self.mesh.hmin()
        self.delta = 2*self.hmin**2
        self.nu = 1e-3
        self.kolmogorov =  5.62e-4

        self.offline = True

        self._time_stepping_parameters.update({
            "monitor": {
                "initial_time":  0,
                "time_step_size": 4e-3
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
            return "ALPHAROM"
        elif hasattr(self, "output_dir") and self.output_dir:
            return "ALPHAROM_" + testcase.name()+"/" + self.output_dir
        else:
            return "ALPHAROM_" + testcase.name()        
        
        
    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        if term == "a":
            theta_a0 = self.nu
            return (theta_a0,)
        elif term == "b":
            theta_b0 = 1.
            return (theta_b0,)
        elif term == "bt":
            if self.offline:
                theta_bt0 = 1.
            else:
                theta_bt0 = 0.    
            return (theta_bt0,)
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
            p = self.p
            alpha_term1 = inner(-ubar[1]*(u[1].dx(0) - u[0].dx(1)),v[0])*dx
            alpha_term2 = inner(ubar[0]*(u[1].dx(0) - u[0].dx(1)),v[1])*dx
            alpha_term = alpha_term1 + alpha_term2
            c0 = inner(grad(u)*u, v)*dx
            c0l = - inner(p + 0.5*inner(u,u), div(v))*dx + alpha_term
            cl1 =  inner(grad(ubar), grad(vbar))*dx # + inner(ubar,vbar)*dx - inner(u, vbar)*dx
            cl2 =  inner(ubar,vbar)*dx - inner(u, vbar)*dx
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
            bc0 = [DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 1), DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 4), DirichletBC(self.V.sub(0), self.inlet, self.boundaries, 3)]
            return (bc0,)
        elif term == "dirichlet_bc_ubar":
            bc0 = [DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 3),
                   DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 1), DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 4)]
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



mypath = ""

# 1. Create case
testcase = CylinderFlowCase("Michele_fine120")

# 2. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "u_bar", "p"])

# 3. Allocate an object of the NavierStokesUnsteady class
navier_stokes_unsteady_problem = NavierStokesUnsteady(V, subdomains=subdomains, boundaries=boundaries, mesh=mesh)
mu_range = []
navier_stokes_unsteady_problem.set_mu_range(mu_range)
navier_stokes_unsteady_problem.set_time_step_size(4e-4)
navier_stokes_unsteady_problem.set_final_time(4.)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(navier_stokes_unsteady_problem)
pod_galerkin_method.set_Nmax(200)

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
AAAAAAAAAAA
# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(1)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
