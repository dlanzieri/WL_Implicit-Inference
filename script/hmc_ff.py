from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
from functools import partial
import jax
from numpyro.handlers import seed, condition
from sbi_lens.simulator import lensingLogNormal
from sbi_lens.simulator.utils import get_reference_sample_posterior_full_field
import numpyro
import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()
logger.addFilter(CheckTypesFilter())
'unset XLA_FLAGS'

import pickle


from numpyro.handlers import seed, condition, trace


model = partial(lensingLogNormal,
                model_type='lognormal',
                with_noise=True)

# condition the model on a given set of parameters
fiducial_model = condition(model, {'omega_c': 0.2664, 
                                   'omega_b': 0.0492,
                                   'sigma_8': 0.831,
                                   'h_0': 0.6727,
                                   'n_s': 0.9645,
                                   'w_0': -1.0})


#sample a mass map (using the same seed we used to create the data for the ps and SBI)
model_trace = trace(seed(fiducial_model, jax.random.PRNGKey(42))).get_trace() 
m_data = model_trace['y']['value']


init_values = {k: model_trace[k]['value'] for k in ['z', 'omega_c', 'sigma_8', 'omega_b', 'h_0', 'n_s', 'w_0']}



samples_ff = get_reference_sample_posterior_full_field(    
            run_mcmc=True,
            N=256,
            map_size=10,
            gals_per_arcmin2=27,
            sigma_e=0.26,
            model=model,
            m_data=m_data,
            num_results=1000,
            num_warmup=200,
            max_tree_depth=6,
            step_size=1e-2,
            num_chains=1,
            nb_loop=20, 
            init_strat=numpyro.infer.init_to_value(values=init_values),
            chain_method='vectorized',
            key=jax.random.PRNGKey(3))


with open("posterior_full_field__"
                    "{}N_{}ms_{}gpa_{}se.npy".format(
                       256, 10, 27, 0.26), "wb") as fp:
    pickle.dump(samples_ff, fp)