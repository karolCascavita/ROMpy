#
#
#
#
#
from dolfin import *
import numpy as np

class TestCase:
    
    ic_field_u = 0
    ic_field_p = 0
    ic_field_s = 0

    def InitialCondition(self, V):
        raise NotImplementedError        
    
    def BoundaryConditions_U(self, V):
        raise NotImplementedError

    def BoundaryConditions_P(self,V):
        raise NotImplementedError

    def BoundaryConditions_Ubar(self, V):
        raise NotImplementedError

    def BoundaryConditionsTheta_U(self, t):
        raise NotImplementedError

    def BoundaryConditionsTheta_P(self, t):
        raise NotImplementedError

    def BoundaryConditionsTheta_Ubar(self, t):
        raise NotImplementedError

    def Forcing(self, V): 
        return Constant((0.0, 0.0))    

    def g(self,V):
        return Constant(0.0)    

    def has_exact_solution(self):
        return False
        
    def ExactVelocity(self, t): 
        return None
    
    def ExactPressure(self, t): 
        return None
    
    def name(self):
        raise NotImplementedError


class CylinderFlowCase(TestCase):
    def __init__(self, mesh_label):

        path = "meshes/"+mesh_label #
  
        mesh = Mesh()

        with XDMFFile(path+"/mesh.xdmf") as infile:
            infile.read(mesh)

        mvc_boundaries = MeshValueCollection("size_t", mesh, 1)
        mvc_subdomains = MeshValueCollection("size_t", mesh, 2)
        
        with XDMFFile(path+"/facet_region.xdmf") as infile:
            infile.read(mvc_boundaries, "name_to_read")
        with XDMFFile(path + "/physical_region.xdmf") as infile:
            infile.read(mvc_subdomains, "name_to_read")

        self.boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
        self.subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomains)       
        self.mesh = mesh

        for f in facets(mesh):
            if f.exterior():
                marker = self.boundaries[f]
                if marker > 1000:
                    print("Unmarked exterior facet:", f.index())
                    print("midpoint:", f.midpoint().x(), f.midpoint().y())


    def InitialCondition(self, V):
        solution  = Function(V)
        return solution

    def BoundaryConditions_U(self,V):

        zero =  Constant((0.0, 0.0))
        self.inlet = Expression(("(6.0/(0.41*0.41))*x[1]*(0.41 - x[1])", "0."), 
                                element=V.sub(0).ufl_element(), t = 0.0)

        bc0 = [DirichletBC(V.sub(0), zero      , self.boundaries, 1), # walls
               #DirichletBC(V.sub(0), self.inlet, self.boundaries, 2), # outlet
               DirichletBC(V.sub(0), self.inlet, self.boundaries, 3), # intlet
               DirichletBC(V.sub(0), zero      , self.boundaries, 4)] # cylinder
        return bc0

    def BoundaryConditions_P(self,V):
        bc0 = [DirichletBC(V.sub(2), Constant(0), "near(x[0],0.0) && near(x[1],0.0)", "pointwise")] # pressure when all dirichlet
        
        return bc0

    def BoundaryConditions_Ubar(self,V):

        self.inlet_bar = Expression(("(6.0/(0.41*0.41))*x[1]*(0.41 - x[1])", "0."), 
                                element=V.sub(1).ufl_element(), t = 0.0)

        bc0 = [ 
                DirichletBC(V.sub(1), Constant((0.0, 0.0)), self.boundaries, 1),
                #DirichletBC(V.sub(1), self.inlet_bar      , self.boundaries, 2),
                DirichletBC(V.sub(1), self.inlet_bar      , self.boundaries, 3),
                DirichletBC(V.sub(1), Constant((0.0, 0.0)), self.boundaries, 4)
               ]
               
        return bc0

    def BoundaryConditionsTheta_U(self, t):       
        return (1., )  

    def BoundaryConditionsTheta_P(self, t):       
        return (1., )  

    def BoundaryConditionsTheta_Ubar(self, t):       

        # return (1.0, )  

        # WARNING: Temporally set to make ubar zero to solve NSE without regularizations. 
        # this has to be fixed
        return (0.0, )  

    def name(self):
        return "Cylinder" 


class CylinderFlowCaseSinProfile(TestCase):
    def __init__(self, mesh_label):

        path = "meshes/"+mesh_label #
  
        mesh = Mesh()

        with XDMFFile(path+"/mesh.xdmf") as infile:
            infile.read(mesh)

        mvc_boundaries = MeshValueCollection("size_t", mesh, 1)
        mvc_subdomains = MeshValueCollection("size_t", mesh, 2)
        
        with XDMFFile(path+"/facet_region.xdmf") as infile:
            infile.read(mvc_boundaries, "name_to_read")
        with XDMFFile(path + "/physical_region.xdmf") as infile:
            infile.read(mvc_subdomains, "name_to_read")

        self.boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
        self.subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomains)       
        self.mesh = mesh

        for f in facets(mesh):
            if f.exterior():
                marker = self.boundaries[f]
                if marker > 1000:
                    print("Unmarked exterior facet:", f.index())
                    print("midpoint:", f.midpoint().x(), f.midpoint().y())

    def InitialCondition(self, V):
        solution  = Function(V)
        return solution

    def BoundaryConditions_U(self,V):

        zero =  Constant((0.0, 0.0))

        self.inlet_u = Expression(("(6.0/(0.41*0.41))*sin(pi*t/8.0)*x[1]*(0.41 - x[1])", "0."), 
                                element=V.sub(0).ufl_element(), t = 0.0)

        bc0 = [DirichletBC(V.sub(0), zero        , self.boundaries, 1), # walls
               #DirichletBC(V.sub(0), self.inlet_u, self.boundaries, 2), # outlet
               DirichletBC(V.sub(0), self.inlet_u, self.boundaries, 3), # intlet
               DirichletBC(V.sub(0), zero        , self.boundaries, 4)] # cylinder
        
        return bc0

    def BoundaryConditions_P(self,V):
        bc0 = [DirichletBC(V.sub(2), Constant(0), "near(x[0],0.0) && near(x[1],0.0)", "pointwise")] # pressure when all dirichlet
        
        return bc0

    def BoundaryConditions_Ubar(self,V):

        self.inlet_ubar = Expression(("(6.0/(0.41*0.41))*sin(pi*t/8.0)*x[1]*(0.41 - x[1])", "0."), 
                                element=V.sub(1).ufl_element(), t = 0.0)

        bc0 = [ 
                DirichletBC(V.sub(1), Constant((0.0, 0.0)), self.boundaries, 1), # walls
        #        DirichletBC(V.sub(1), self.inlet_ubar     , self.boundaries, 2), # outlet
                DirichletBC(V.sub(1), self.inlet_ubar     , self.boundaries, 3), # intlet
                DirichletBC(V.sub(1), Constant((0.0, 0.0)), self.boundaries, 4), # cylinder
               ]

        return bc0

    def BoundaryConditionsTheta_U(self, t):       
        return (np.sin(np.pi * t/8.0), )

    def BoundaryConditionsTheta_P(self, t): 
        return (1.,)

    def BoundaryConditionsTheta_Ubar(self, t): 
        
        #return (np.sin(np.pi * t/8.0),)
        
        # WARNING: Temporally set to make ubar zero to solve NSE without regularizations. 
        # this has to be fixed
        #       
        return (0.0, ) 

    def name(self):
        return "Cylinder_SIN"


class TaylorVortexCase(TestCase):
    
    def __init__(self, n, Re):
        self.nu = 1.0/Re
        self.u_exact = Expression(
            ("-cos(n*pi*x[0])*sin(n*pi*x[1])*exp(-2.0*n*n*pi*pi*t/Re)",
             " sin(n*pi*x[0])*cos(n*pi*x[1])*exp(-2.0*n*n*pi*pi*t/Re)"),
            degree=5, pi=np.pi, nu=self.nu,n=n, t=0.0
        )
        self.p_exact = Expression(
            "-0.25*(cos(2.0*n*pi*x[0]) + cos(2.0*n*pi*x[1]))*exp(-4.0*n*n*pi*pi*t/Re)",
            degree=5, pi=np.pi, nu=self.nu,n=n, t=0.0
        )
        self.mesh = UnitSquareMesh(N, N)
        self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        self.subdomains.set_all(0)

        self.boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        self.boundaries.set_all(0)

    def InitialConditions(self,V, t0=0):
        V0 = V.sub(0).collapse()
        V1 = V.sub(1).collapse()
        V2 = V.sub(2).collapse()

        self.u_exact.t = t0
        self.p_exact.t = t0
                
        u_init     = interpolate(u_exact, V0)
        u_bar_init = interpolate(u_exact, V1)
        p_init     = interpolate(p_exact, V2)     
        
        solution  = Function(V)
        assigner  = FunctionAssigner(V, [V0, V1,V2])
        assigner.assign(solution, [u_init, u_bar_init_p_init])
        return solution

    def BoundaryConditions_U(self, V):
        bc0 = [DirichletBC(V.sub(0), self.u_exact, self.boundaries, "on_boundary")]
        return bc0

    def BoundaryConditions_P(self,V):
        bc0 = [DirichletBC(V.sub(2), Constant(0), "near(x[0],0.0) && near(x[1],0.0)", "pointwise")] # pressure when all dirichlet        
        return bc0

    def BoundaryConditions_Ubar(self, V):
        bc0 = [DirichletBC(V.sub(1), self.u_exact, self.boundaries, "on_boundary")]
        return bc0


    def BoundaryConditionsTheta_U(self, t):
        return (1.0,)

    def BoundaryConditionsTheta_P(self, t):
        return (1.0,)

    def BoundaryConditionsTheta_Ubar(self, t):
        return (1.0,)

    def ExactSolution(self, t):
        self.u_exact.t = t
        self.p_exact.t = t
        return self.u_exact, self.p_exact
                
    def name(self):
        return "GTVortex" 
      
