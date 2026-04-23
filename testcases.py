#
#
#
#
#
from dolfin import *


class TestCase:
    
    def InitialCondition(self, V):
        return NotImplementedError        
            
    def BoundaryConditions(self, V, boundaries):
        return NotImplementedError
    
    def BoundaryConditionsUbar(self, V, boundaries):
        return NotImplementedError

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
        return NotImplementedError

class CylinderFlowCase(TestCase):
    def __init__(self, mesh_specific_name):
        # Read the mesh for this problem
        path = "data_cylinder"
  
        self.mesh = Mesh("/beegfs/users/kcascavita/CODES/ROMpy/data_cylinder/Cylinder_refined_Michele.xml")
        self.subdomains = MeshFunction("size_t", self.mesh, "/beegfs/users/kcascavita/CODES/ROMpy/data_cylinder/Cylinder_refined_physical_region_Michele.xml")
        self.boundaries = MeshFunction("size_t", self.mesh, path+"/Cylinder_refined_facet_region_"+mesh_specific_name + ".xml")
        print(self.mesh.hmax(), self.mesh.hmin())


    def InitialCondition(self, V):
        solution  = Function(V)
        return solution

    def BoundaryConditions(self, V, boundaries):
        self.inlet = Expression(("6.0/((0.41)*(0.41))*x[1]*(0.41 - x[1])", "0."), 
                                element=V.sub(0).ufl_element())

        bc0 = [DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, 1),
               DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, 4), 
               DirichletBC(V.sub(0), self.inlet          , boundaries, 3)]
        
        return bc0

    def BoundaryConditionsUbar(self, V, boundaries):
        bc0 = [DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 3),
               DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 1), 
               DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 4)]
        return bc0

    def name(self):
        return "Cylinder" 


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

    def BoundaryConditions(self, V, boundaries):
        return [DirichletBC(V.sub(0), self.u_exact, boundaries, "on_boundary")]

    def BoundaryConditionsUbar(self, V, boundaries):
        bc0 = [DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 3),
               DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 1), 
               DirichletBC(self.V.sub(1), Constant((0.0, 0.0)), self.boundaries, 4)]
        return [DirichletBC(V.sub(0), self.u_exact, boundaries, "on_boundary")]
         #return bc0

    def ExactSolution(self, t):
        self.u_exact.t = t
        self.p_exact.t = t
        return self.u_exact, self.p_exact
                
    def name(self):
        return "GTVortex" 
      
