import os
import json
import shutil

from dolfin import *
from rbnics import *
from testcases import *


@ExactParametrizedFunctions()
class NavierStokesModelBase(NavierStokesUnsteadyProblem):

    MODEL_NAME = "BASE_MODEL"

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
        self.f = self.testcase.Forcing(V)
        self.g = self.testcase.g(V)
        
        self.hmin = self.mesh.hmin()
        self.delta = self.hmin**2
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
            dirname = self.MODEL_NAME
        elif hasattr(self, "parameters") and self.parameters:
            dirname = self.MODEL_NAME + "_" + testcase.name()+"/" + self.parameters["output_dir"] 
        else:
            dirname = self.MODEL_NAME + "_ROM_" + testcase.name()

        os.makedirs(dirname, exist_ok=True)
        return dirname  

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):

        constant_terms = {
                "a": (self.nu,),
                "b": (1.,),
                "bt":(1.,),
                "f": (1.,),
                "g": (1.,),
                "m": (1.,),
                "dirichlet_bc_u": (1.,), #  t = self.t,  6/((0.41)*(0.41))
                "dirichlet_bc_ubar": (1.,),
            }

        if term in constant_terms:
            return constant_terms[term]
        elif term == "c":
            '''   
            if self.offline:
                print("OFFLINE")
                theta_c0  = 1.
                theta_cl0 = 0.
                theta_cl1 = 0.
                theta_cl2 = 0.
            else:
                print("ONLINE")
                theta_c0  = 0.
                theta_cl0 = 1.
                theta_cl1 = self.delta
                theta_cl2 = 1.
            '''
            theta_c0  = 0.
            theta_cl0 = 1.
            theta_cf1 = self.delta
            theta_cf2 = 1.
            
            return (theta_c0,theta_cl0, theta_cf1, theta_cf2)
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
            return self.assemble_b()
        elif term == "bt":
            p = self.dp
            v = self.v
            bt0 = - p*div(v)*dx
            return (bt0,)
        elif term == "c" : 
            return self.assemble_c()
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
        elif term == "dirichlet_bc_u":
            bc0 = self.testcase.BoundaryConditions(self.V)
            return (bc0,)
        elif term == "dirichlet_bc_ubar":
            bc0 = self.testcase.BoundaryConditions(self.V)
            return (bc0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

    '''
    def compute_kinetic_energy(self):

        u = self.u
        return assemble(0.5 * inner(u, u) * self.dx)
    '''    
    def vorticity(self, u):
        return u[1].dx(0) - u[0].dx(1)


class NavierStokesUnsteadyLeray(NavierStokesModelBase):
    
    MODEL_NAME = "LERAY"

    def assemble_b(self):
        u = self.du
        q = self.q
        b0 = - q*div(u)*dx
        return (b0,)

   
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_c(self):
        dx = self.dx
        u = self.u
        v = self.v
        ubar = self.ubar
        vbar = self.vbar

        c0  = inner(grad(u)*u, v)*dx
        cl0 = inner(grad(u)*ubar, v)*dx
        cf1 = inner(grad(ubar), grad(vbar))*dx # + inner(ubar,vbar)*dx - inner(u, vbar)*dx
        cf2 = inner(ubar,vbar)*dx - inner(u, vbar)*dx
        return (c0, cl0, cf1, cf2)

          
class NavierStokesUnsteadyAlpha(NavierStokesModelBase):
    
    MODEL_NAME = "ALPHA"

    def assemble_b(self):
        q = self.q
        ubar = self.ubar

        b0 = - q*div(ubar)*dx
        return (b0,)

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_c(self):
        dx = self.dx
        p = self.p
        u = self.u
        v = self.v
        ubar = self.ubar
        vbar = self.vbar
        
        omega_u  = self.vorticity(u)
        alpha_term1 = inner(-ubar[1]*omega_u,v[0])*dx
        alpha_term2 = inner( ubar[0]*omega_u,v[1])*dx
        alpha_term = alpha_term1 + alpha_term2
        
        c0  =  inner(grad(u)*u, v)*dx
        c0l = -inner(0.5*inner(u,u), div(v))*dx + alpha_term
        cf1 =  inner(grad(ubar), grad(vbar))*dx # + inner(ubar,vbar)*dx - inner(u, vbar)*dx
        cf2 =  inner(ubar,vbar)*dx - inner(u, vbar)*dx
        return (c0, c0l, cf1, cf2)


class NavierStokesUnsteadyOmega(NavierStokesModelBase):

    MODEL_NAME = "OMEGA" 

    def assemble_b(self):
        q = self.q
        u = self.u

        b0 = - q*div(u)*dx
        return (b0,)

         
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_c(self):
        dx = self.dx
        p = self.p
        u = self.u
        v = self.v
        ubar = self.ubar
        vbar = self.vbar

        omega_ubar  = self.vorticity(ubar)
        omega_term1 = inner(-u[1]*omega_ubar,v[0])*dx
        omega_term2 = inner( u[0]*omega_ubar,v[1])*dx
        omega_term  = omega_term1 + omega_term2

        c0  =  inner(grad(u)*u, v)*dx
        c0l = -inner(0.5*inner(u,u), div(v))*dx + omega_term
        cf1 =  inner(grad(ubar), grad(vbar))*dx # + inner(ubar,vbar)*dx - inner(u, vbar)*dx
        cf2 =  inner(ubar,vbar)*dx - inner(u, vbar)*dx
        return  (c0, c0l, cf1, cf2)


MODELS = {
    "LERAY": NavierStokesUnsteadyLeray,
    "ALPHA": NavierStokesUnsteadyAlpha,
    "OMEGA": NavierStokesUnsteadyOmega
}