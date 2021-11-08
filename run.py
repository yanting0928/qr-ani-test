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
#from aniserver import ANIRPCCalculator

master_params_str ="""
  restraints = ani cctbx
    .type = choice(multi=False)
  minimizer = lbfgs lbfgs_b lbfgs_ase
    .type = choice(multi=False)
  stpmax = None
    .type = float
  gradients_only = None
    .type = bool
  max_itarations = None
    .type = int
  macro_cycles = None
    .type = int
  gradient_only = None
    .type = bool
"""

def get_master_phil():
  return mmtbx.command_line.generate_master_phil_with_inputs(
    phil_string = master_params_str)

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
    return es.target, es.gradients.as_double()

  def apply_shift(self):
    self.model.set_sites_cart(
      sites_cart = self.model.get_sites_cart()+flex.vec3_double(self.x))

class aniserver_tg_calculator(object):

  def __init__(self, model):
    self.model = model
    self.x = flex.double(self.model.size()*3, 0)
    self.sites_cart = self.model.get_sites_cart()
    self.ase_atoms = ase_atoms_from_model(model=self.model)

  def get_shift(self):
    return self.x

  def target_and_gradients(self, x):
    self.x = x
    calc = ANIRPCCalculator()
    self.model.set_sites_cart(sites_cart = self.sites_cart+flex.vec3_double(self.x))
    self.ase_atoms = ase_atoms_from_model(model=self.model)
    self.ase_atoms.set_calculator(calc)
    target = self.ase_atoms.get_potential_energy()
    gradients = self.ase_atoms.get_forces().tolist()
    g = flex.double([g for gradient in gradients for g in gradient])
    g = g * (-1)
    return target, g.as_double()

  def apply_shift(self ):
    self.model.set_sites_cart(
      sites_cart = self.sites_cart + flex.vec3_double(self.x))

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
    self.g = g
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
    return self.f, self.g

class minimizer_ase(object):
  def __init__(self, engine, calculator, max_iterations, stpmax=0.04):
    # stpmax=0.04 is the default in ASE
    self.calculator = calculator
    self.max_iterations = max_iterations
    self.ase_atoms = calculator.ase_atoms
    self.x = self.calculator.get_shift()
    self.f = None
    self.engine = engine
    self.ase_atoms.set_positions(flex.vec3_double(self.calculator.x))
    self.minimizer = LBFGS(atoms = self.ase_atoms, maxstep=stpmax)
    self.n_func_evaluations = 0
    self.run(nstep = max_iterations)
    self.calculator.apply_shift()

  def step(self):
    x = flex.vec3_double(self.minimizer.atoms.get_positions())
    f,g = self.calculator.target_and_gradients(x = x)
    self.f = f
    forces = numpy.array(g) * (-1)
    self.minimizer.step(forces)
    self.n_func_evaluations += 1

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

def run(params, file_name):
  # Check inputs
  assert params.minimizer in ["lbfgs", "lbfgs_b", "lbfgs_ase"]
  assert params.restraints in ["cctbx", "ani"]
  assert params.stpmax is not None
  assert params.max_itarations is not None and type(params.max_itarations)==int
  assert params.macro_cycles  is not None and type(params.macro_cycles)==int
  assert params.max_itarations > 0
  assert params.macro_cycles > 0
  #
  pdb_inp = iotbx.pdb.input(file_name = file_name)
  model = mmtbx.model.manager(model_input = pdb_inp, log = null_out())
  model.process(make_restraints=True)
  for macro_cycle in range(1, params.macro_cycles):
    if(params.restraints == "cctbx"):
      calculator = cctbx_tg_calculator(model = model)
    elif(params.restraints == "ani"):
      calculator = aniserver_tg_calculator(model = model)
    else: assert 0 # safeguard
    if(params.minimizer == "lbfgs_ase"):
      minimized = minimizer_ase(
        engine         = params.restraints,
        calculator     = calculator,
        max_iterations = params.max_itarations,
        stpmax         = params.stpmax)
    elif(params.minimizer == "lbfgs"):
      assert params.gradient_only in [True, False]
      minimized = minimizer_unbound(
        calculator     = calculator,
        max_iterations = params.max_itarations,
        gradient_only  = params.gradient_only,
        stpmax         = params.stpmax)
    elif(params.minimizer == "lbfgs_b"):
      minimized = minimizer_bound(
        calculator     = calculator,
        max_iterations = params.max_itarations,
        stpmax         = params.stpmax,
        use_bounds     = 2).run()
    else: assert 0 # safeguard
    print "macro_cycle: %3d target value: %15.6f" % (macro_cycle, minimized.f)
  #
  prefix = "%s_%s_stpmax%s_maxiter%s_gronly%s"%(
    params.restraints,
    params.minimizer,
    str("%7.2f"%params.stpmax).strip(),
    str("%7.2f"%params.max_itarations).strip(),
    str(params.gradient_only))
  with open("%s.pdb"%prefix,"w") as fo:
    fo.write(model.model_as_pdb())

if (__name__ == "__main__"):
  args = sys.argv[1:]
  cmdline = mmtbx.utils.process_command_line_args(
    args          = args,
    master_params = get_master_phil())
  run(params = cmdline.params.extract(), file_name = cmdline.pdb_file_names[0])
