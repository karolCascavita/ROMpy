import os
import json
import shutil

from dolfin import *
from rbnics import *
from testcases import *


@ExactParametrizedFunctions()
class NavierStokesUnsteadyLeray(NavierStokesUnsteadyProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):

        self.testcase   = kwargs["testcase"]
        self.parameters = kwargs["parameters"]   

        # Call the standard initialization
        NavierStokesUnsteadyProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs

        self.mesh = kwargs["mesh"]         
        self.subdomains = kwargs["subdomains"]
        self.boundaries = kwargs["boundaries"]

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
            dirname = "LERAY_ROM"
        elif hasattr(self, "parameters") and self.parameters:
            dirname = "LERAY_ROM_" + testcase.name()+"/" + self.parameters["output_dir"] 
        else:
            dirname = "LERAY_ROM_" + testcase.name()

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
           

@ExactParametrizedFunctions()
class NavierStokesUnsteadyAlpha(NavierStokesUnsteadyProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):

        self.testcase   = kwargs["testcase"]
        self.parameters = kwargs["parameters"]    

        # Call the standard initialization
        NavierStokesUnsteadyProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        assert "testcase" in kwargs
        assert "parameters" in kwargs
        assert "mesh" in kwargs
        
        self.mesh = kwargs["mesh"]         
        self.subdomains = kwargs["subdomains"]
        self.boundaries = kwargs["boundaries"]

        self._solution.assign(self.testcase.InitialCondition(V))

        self.dup = TrialFunction(V)
        (self.du, self.dubar, self.dp) = split(self.dup)
        (self.u, self.ubar, self.p) = split(self._solution)
        
        vq = TestFunction(V)
        (self.v, self.vbar, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.f = self.testcase.Forcing(V)
        self.g = self.testcase.g(V)

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
            dirname = "ALPHA_ROM"
        elif hasattr(self, "parameters") and self.parameters:
            dirname = "ALPHA_ROM_" + testcase.name()+"/" + self.parameters["output_dir"] 
        else:
            dirname = "ALPHA_ROM_" + testcase.name()        
        
        os.makedirs(dirname, exist_ok=True)
        return dirname          
        
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


@ExactParametrizedFunctions()
class NavierStokesUnsteadyOmega(NavierStokesUnsteadyProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):

        self.testcase   = kwargs["testcase"]
        self.parameters = kwargs["parameters"]    

        # Call the standard initialization
        NavierStokesUnsteadyProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs

        self.mesh = kwargs["mesh"]         
        self.subdomains = kwargs["subdomains"]
        self.boundaries = kwargs["boundaries"]

        self._solution.assign(self.testcase.InitialCondition(V))
        
        self.dup = TrialFunction(V)
        (self.du, self.dubar, self.dp) = split(self.dup)
        (self.u , self.ubar , self.p ) = split(self._solution)

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
            dirname = "OMEGA_ROM"
        elif hasattr(self, "parameters") and self.parameters:
            dirname = "OMEGA_ROM_" + testcase.name()+"/" + self.parameters["output_dir"] 
        else:
            dirname = "OMEGA_ROM_" + testcase.name()        
        
        os.makedirs(dirname, exist_ok=True)
        return dirname    


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
            omega_term1 = inner(-u[1]*(ubar[1].dx(0) - ubar[0].dx(1)),v[0])*dx
            omega_term2 = inner(u[0]*(ubar[1].dx(0) - ubar[0].dx(1)),v[1])*dx
            omega_term = omega_term1 + omega_term2
            c0 = inner(grad(u)*u, v)*dx
            c0l = - inner(p + 0.5*inner(u,u), div(v))*dx + omega_term
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


MODELS = {
    "LERAY": NavierStokesUnsteadyLeray,
    "ALPHA": NavierStokesUnsteadyAlpha,
    "OMEGA": NavierStokesUnsteadyOmega
}