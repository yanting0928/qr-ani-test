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
from scitbx import minimizers

master_params_str ="""
  restraints = ani cctbx
    .type = choice(multi=False)
  minimizer = lbfgs lbfgs_b lbfgs_ase
    .type = choice(multi=False)
  stpmax = None
    .type = float
  max_itarations = None
    .type = int
  gradient_only = None
    .type = bool
  prefix = None
    .type = str
"""

def get_master_phil():
  return mmtbx.command_line.generate_master_phil_with_inputs(
    phil_string = master_params_str)

class cctbx_tg_calculator(object):

  def __init__(self, model, max_shift):
    self.model = model
    self.x = flex.double(self.model.size()*3, 0)
    self.n = self.x.size()
    self.f = None
    self.g = None
    self.f_start = None
    self.sites_cart = self.model.get_sites_cart()
    self.bound_flags = flex.int(self.n, 2)
    self.lower_bound = flex.double([-1*max_shift]*self.n)
    self.upper_bound = flex.double([   max_shift]*self.n)

  def target_and_gradients(self):
    es = self.model.get_restraints_manager().energies_sites(
      sites_cart        = self.sites_cart+flex.vec3_double(self.x),
      compute_gradients = True)
    self.f, self.g = es.target, es.gradients.as_double()
    if(self.f_start is None):
      self.f_start = self.f
    return self.f, self.g

  def compute_functional_and_gradients(self):
    return self.target_and_gradients()

  def apply_x(self):
    self.model.set_sites_cart(
      sites_cart = self.sites_cart+flex.vec3_double(self.x))

  def __call__(self):
    f, g = self.target_and_gradients()
    return self.x, f, g

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

  def apply_shift(self):
    self.model.set_sites_cart(
      sites_cart = self.sites_cart + flex.vec3_double(self.x))

class minimizer_ase(object):
  def __init__(self, engine, calculator, max_iterations, stpmax=0.04):
    # stpmax=0.04 is the default in ASE
    self.calculator = calculator
    self.max_iterations = max_iterations
    self.ase_atoms = ase_atoms_from_model(model=calculator.model)
    self.x = self.calculator.x
    self.f = None
    self.engine = engine
    self.ase_atoms.set_positions(flex.vec3_double(self.calculator.x))
    self.minimizer = LBFGS(atoms = self.ase_atoms, maxstep=stpmax)
    self.n_func_evaluations = 0
    self.run(nstep = max_iterations)

  def step(self):
    x = flex.vec3_double(self.minimizer.atoms.get_positions()).as_double()
    self.calculator.x = x
    f,g = self.calculator.target_and_gradients()
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
  assert params.max_itarations > 0
  #
  pdb_inp = iotbx.pdb.input(file_name = file_name)

  from qrefine import qr
  model = qr.process_model_file(pdb_file_name = file_name)
  #
  if(params.restraints == "cctbx"):
    calculator = cctbx_tg_calculator(
      model       = model,
      max_shift   = params.stpmax)
  elif(params.restraints == "ani"):
    calculator = aniserver_tg_calculator(model = model)
  #
  if(params.minimizer == "lbfgs_b"):
    minimized = minimizers.lbfgsb(
      calculator     = calculator,
      max_iterations = params.max_itarations)
    minimized.show()
  elif(params.minimizer == "lbfgs"):
    assert params.gradient_only in [True, False]
    minimized = minimizers.lbfgs(
      calculator     = calculator,
      max_iterations = params.max_itarations,
      gradient_only  = params.gradient_only,
      stpmax         = params.stpmax)
    minimized.show()
  elif(params.minimizer == "lbfgs_ase"):
    minimized = minimizer_ase(
      engine         = params.restraints,
      calculator     = calculator,
      max_iterations = params.max_itarations,
      stpmax         = params.stpmax)
  #
  calculator.apply_x()
  #
  if(params.prefix is None):
    params.prefix = "%s_%s_stpmax%s_maxiter%s_gronly%s"%(
      params.restraints,
      params.minimizer,
      str("%7.2f"%params.stpmax).strip(),
      str("%7.2f"%params.max_itarations).strip(),
      str(params.gradient_only))
  with open("%s.pdb"%params.prefix,"w") as fo:
    fo.write(model.model_as_pdb())

if (__name__ == "__main__"):
  args = sys.argv[1:]
  cmdline = mmtbx.utils.process_command_line_args(
    args          = args,
    master_params = get_master_phil())
  run(params = cmdline.params.extract(), file_name = cmdline.pdb_file_names[0])
