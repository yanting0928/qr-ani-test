import scitbx.lbfgs
import sys
from cctbx.array_family import flex
import iotbx.pdb
import mmtbx.model
from libtbx.utils import null_out
from mmtbx.ncs import tncs
from libtbx import adopt_init_args
from ase.optimize.lbfgs import LBFGS
import numpy
from aniserver import ANIRPCCalculator

class cctbx_tg_calculator(object):

  def __init__(self, model):
    self.model = model
    self.x = flex.double(self.model.size()*3, 0)
    self.sites_cart = self.model.get_sites_cart()
    self.ase_atoms = ase_atoms_from_model(model=self.model)

  def get_shift(self):
    return self.x

  def target_and_gradients(self, x):
    self.x = x
    es = self.model.get_restraints_manager().energies_sites(
      sites_cart        = self.sites_cart+flex.vec3_double(self.x),
      compute_gradients = True)
    print(list(es.gradients.as_double()))
    return es.target, es.gradients.as_double()

  def apply_shift(self):
    self.model.set_sites_cart(
      sites_cart = self.model.get_sites_cart()+flex.vec3_double(self.x))

class aniserver_tg_calculator(object):

  def __init__(self, model):
    self.model = model
#    self.x = flex.double(self.model.size()*3, 0)
    self.sites_cart = self.model.get_sites_cart()
    self.ase_atoms = ase_atoms_from_model(model=self.model)
    self.x = self.sites_cart.as_double()

  def get_shift(self):
    return self.x

  def target_and_gradients(self, x):
    self.x = x
    c = self.model
    calc = ANIRPCCalculator()
    c.set_sites_cart(sites_cart =c.get_sites_cart()+flex.vec3_double(self.x))
    self.ase_atoms = ase_atoms_from_model(model=c)
    self.ase_atoms.set_calculator(calc) 
    target = self.ase_atoms.get_potential_energy()
    gradients = self.ase_atoms.get_forces().tolist()
    g = flex.double([g for gradient in gradients for g in gradient])
    print list(g)
#    target = target * (-1)
    g = g * (-1)
#    print target, list(g)
    return target, g.as_double()

  def apply_shift(self):
    self.model.set_sites_cart(
      sites_cart = self.model.get_sites_cart()+flex.vec3_double(self.x))

class minimizer_bound(object):
  def __init__(self,
               calculator,
               stpmax,
               use_bounds,
               max_iterations):
    adopt_init_args(self, locals())
    self.x = self.calculator.x
    self.n = self.x.size()
    self.lower_bound = flex.double([-1*self.stpmax]*self.n)
    self.upper_bound = flex.double([   self.stpmax]*self.n)
    self.n_func_evaluations = 0
    self.max_iterations = max_iterations

  def run(self):
    self.minimizer = tncs.lbfgs_run(
      target_evaluator = self,
      use_bounds       = self.use_bounds,
      lower_bound      = self.lower_bound,
      upper_bound      = self.upper_bound,
      max_iterations   = self.max_iterations)
    self()
    self.calculator.apply_shift()
    return self

  def __call__(self):
    self.n_func_evaluations += 1
    f, g = self.calculator.target_and_gradients(x = self.x)
    self.f = f
#    self.g = g.as_double()
    self.g = g
    print self.n_func_evaluations, self.f
    return self.x, self.f, self.g

class minimizer_unbound(object):
  def __init__(self, calculator, stpmax, max_iterations, gradient_only):
    self.calculator = calculator
    self.x = self.calculator.get_shift()
    self.n_func_evaluations = 0
    import scitbx.lbfgs
    core_params = scitbx.lbfgs.core_parameters(
      stpmin = 1.e-9,
      stpmax = stpmax)
    termination_params = scitbx.lbfgs.termination_parameters(
      max_iterations = max_iterations,
      min_iterations = None)
    exception_handling_params = scitbx.lbfgs.exception_handling_parameters(
      ignore_line_search_failed_step_at_lower_bound = True)
    minimizer = scitbx.lbfgs.run(
      core_params               = core_params,
      termination_params        = termination_params,
      exception_handling_params = exception_handling_params,
      target_evaluator          = self,
      gradient_only             = gradient_only,
      line_search               = True,
      log                       = None)
    calculator.apply_shift()

  def compute_functional_and_gradients(self):
    self.n_func_evaluations += 1
    f, g = self.calculator.target_and_gradients(x = self.x)
    self.f = f
    self.g = g
    print self.n_func_evaluations, self.f
    return self.f, self.g

class minimizer_ase(object):
  def __init__(self, calculator, max_iterations): 
    self.calculator = calculator  
    self.max_iterations = max_iterations
    self.ase_atoms = calculator.ase_atoms
    self.ase_atoms.set_positions(flex.vec3_double(self.calculator.x))
    self.minimizer =  LBFGS(atoms = self.ase_atoms)    
    self.n_func_evaluations = 0
    self.run(nstep = max_iterations)
    self.calculator.apply_shift()
  
  def step(self):
    sites_cart = flex.vec3_double(self.minimizer.atoms.get_positions()) 
    print ("moving sites_cart", list(sites_cart))
    f,g = self.calculator.target_and_gradients(x = sites_cart)
    forces = numpy.array(g) * (-1)
    self.minimizer.step(forces)
    self.n_func_evaluations += 1
    print self.n_func_evaluations, f

  def run(self, nstep):
    for i in range(nstep):
      v = self.step()

from ase import Atoms 
def ase_atoms_from_model(model):
  positions = []
  symbols = []
  unit_cell = model.crystal_symmetry().unit_cell().parameters()
  for chain in model._pdb_hierarchy.chains():
    for residue_group in chain.residue_groups():
      for atom in residue_group.atoms():
        element = atom.element.strip()
        if (len(element) == 2):
          element = element[0] + element[1].lower()
        symbols.append(element)
        positions.append(list(atom.xyz))
  return Atoms(symbols=symbols, positions=positions, pbc=True, cell=unit_cell)

def run(args):
#  assert len(args)==1, args
  pdb_inp = iotbx.pdb.input(file_name = args[0])
  model = mmtbx.model.manager(model_input = pdb_inp, log = null_out())
  model.process(make_restraints=True)
  engine = args[1].strip()
  if (engine in ["cctbx"]):
    calculator = cctbx_tg_calculator(model = model)
  else:
    calculator = aniserver_tg_calculator(model = model)
  lbfgs_c = int(args[2])    #  0 - ase_lbfgs, 1 - cctbx_unbond, 2 - cctbx_bound
  if (lbfgs_c==0):
    minimized = minimizer_ase(
      calculator            = calculator,
      max_iterations        = 2)
  elif (lbfgs_c==1):
    minimized = minimizer_unbound(
      calculator     = calculator, 
      max_iterations = 50,
      gradient_only  = False,
      stpmax         = 1.e+9)
  else:
    minimized = minimizer_bound(
      calculator     = calculator, 
      max_iterations = 50,
      stpmax         = 1.e+9, 
      use_bounds     = 2).run()
  #
  with open("minimized_cctbx.pdb","w") as fo:
    fo.write(model.model_as_pdb())

if (__name__ == "__main__"):
  run(args = sys.argv[1:])
