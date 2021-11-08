import scitbx.lbfgs
import sys
from cctbx.array_family import flex
import iotbx.pdb
import mmtbx.model
from libtbx.utils import null_out
from mmtbx.ncs import tncs
from libtbx import adopt_init_args

class cctbx_tg_calculator(object):

  def __init__(self, model):
    self.model = model
    self.x = flex.double(self.model.size()*3, 0)
    self.sites_cart = self.model.get_sites_cart()

  def get_shift(self):
    return self.x

  def target_and_gradients(self, x):
    self.x = x
    es = self.model.get_restraints_manager().energies_sites(
      sites_cart        = self.sites_cart+flex.vec3_double(self.x),
      compute_gradients = True)
    return es.target, es.gradients

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
    self.g = g.as_double()
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
    self.g = g.as_double()
    print self.n_func_evaluations, self.f
    return self.f, self.g

def run(args):
  assert len(args)==1, args
  pdb_inp = iotbx.pdb.input(file_name = args[0])
  model = mmtbx.model.manager(model_input = pdb_inp, log = null_out())
  model.process(make_restraints=True)
  calculator = cctbx_tg_calculator(model = model)
  if True:
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
