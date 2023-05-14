from functools import partial
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from haiku._src.nets.resnet import ResNet34, ResNet18
import optax
import haiku as hk
from numpyro.handlers import seed, condition, trace

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

from sbi_lens.gen_dataset.lensing_lognormal_dataset import LensingLogNormalDataset
from sbi_lens.gen_dataset.utils import augmentation_noise, augmentation_flip

from sbi_lens.normflow.models import AffineSigmoidCoupling
from sbi_lens.normflow.models import (
  ConditionalRealNVP,
  AffineCoupling
)
from sbi_lens.normflow.train_model import TrainModel
from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator import lensingLogNormal

###################
import logging
import tensorflow_probability as tfp; tfp = tfp.substrates.jax

tfp.distributions.TransformedDistribution(
    tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity()
)

logger = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
###################


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=9000)
parser.add_argument("--resnet", type=str, default='resnet18')
parser.add_argument("--loss", type=str, default='train_compressor_vmim')
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--filename", type=str, default='res' )

args = parser.parse_args()

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

print('######## Data direct ########', ROOT_DIR /'notebook/tensorflow_dataset' )
print('######## CONFIG LSST Y 10 ########')

dim = 6

N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0


truth = config_lsst_y_10.truth

params_name = config_lsst_y_10.params_name


######## Specified the configuration used to training the compressor########
if args.loss == 'train_compressor_vmim':
  l_name = 'vmim'
elif args.loss == 'train_compressor_mse':
  l_name = 'mse'

print('######## CREATE OBSERVATION ########')

# plot observed mass map
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


#sample a mass map
key_par=jax.random.PRNGKey(args.seed)
model_trace = trace(seed(fiducial_model, key_par)).get_trace()
m_data = model_trace['y']['value']


print('######## DATA AUGMENTATION ########')
tf.random.set_seed(1)

augmentation = lambda example: augmentation_flip(
  augmentation_noise(
    example=example,
    N=N,
    map_size=map_size,
    sigma_e=sigma_e,
    gal_per_arcmin2=gals_per_arcmin2,
    nbins=nbins,
    a=a,
    b=b,
    z0=z0
  )
)
print('######## CREATE COMPRESSOR ########')

bijector_layers_compressor = [128] * 2

bijector_compressor = partial(
  AffineCoupling,
  layers=bijector_layers_compressor,
  activation=jax.nn.silu
)

NF_compressor = partial(
  ConditionalRealNVP,
  n_layers=4,
  bijector_fn=bijector_compressor
)

class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(dim)(y)
        return nvp

nf = hk.without_apply_rng(
  hk.transform(
    lambda theta, y : Flow_nd_Compressor()(y).log_prob(theta).squeeze()
  )
)

# compressor
if args.resnet == 'resnet34':
  print('ResNet34')

  compressor = hk.transform_with_state(
    lambda y : ResNet34(dim)(y, is_training=False)
  )

elif args.resnet == 'resnet18':
  print('ResNet18')

  compressor = hk.transform_with_state(
    lambda y : ResNet18(dim)(y, is_training=False)
  )

print('######## LOAD PARAMETERS COMPRESSOR ########')

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/sbi_lens/sbi_lens/data/params_compressor/params_nd_compressor_{}.pkl".format(l_name), 'rb') as f:
    parameters_compressor = pickle.load(f)

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/sbi_lens/sbi_lens/data/params_compressor/opt_state_resnet_{}.pkl".format(l_name), 'rb') as g:
    opt_state_resnet = pickle.load(g)
    
print('######## CHECK WE ARE LOADING THE CORRECT PARAMETERS: ########')
print('######## parameters_compressor: ########', "/gpfsdswork/projects/rech/ykz/ulm75uc/sbi_lens/sbi_lens/data/params_compressor/opt_state_resnet_{}.pkl".format(l_name))

print('######## CREATE NF for SBI########')

prior = tfd.MultivariateNormalDiag(jnp.array([0.2664,0.0492,0.831,0.6727,0.9645,-1.0]), jnp.array([0.2,0.006,0.14,0.063,0.08,0.9])*jnp.ones(dim))
theta = prior.sample(10000,key_par)
scale_theta = (jnp.std(theta, axis = 0)/0.06)
shift_theta = jnp.mean(theta/scale_theta, axis = 0)-0.5

# nf

bijector_layers = [128] * 2

bijector_npe = partial(
  AffineSigmoidCoupling, 
  layers=bijector_layers, 
  n_components=16, 
  activation=jax.nn.silu
)

NF_npe = partial(
    
  ConditionalRealNVP, 
  n_layers=4, 
  bijector_fn=bijector_npe
)


class SmoothNPE(hk.Module):
    def __call__(self, y):
        net = y
        nvp = NF_npe(dim)(net)
        nf = tfd.TransformedDistribution(
            nvp,
            tfb.Chain([tfb.Scale(scale_theta),tfb.Shift(shift_theta)])
        )
        return nf

nvp_nd = hk.without_apply_rng(
  hk.transform(
    lambda theta, y : SmoothNPE()(y).log_prob(theta).squeeze()
  )
)

print('######## TRAIN ########')
# init nd
params_nd = nvp_nd.init(
  key_par,
  theta=0.5 * jnp.ones([1, dim]),
  y=0.5 * jnp.ones([1, dim])
)


#optimizer
total_steps = args.total_steps
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.001,
    boundaries_and_scales={int(total_steps*0.2):0.7,
                           int(total_steps*0.4):0.7,
                           int(total_steps*0.6):0.7,
                           int(total_steps*0.8):0.7}
)

optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state_nd = optimizer.init(params_nd)


model = TrainModel(
        compressor = compressor,
        nf = nvp_nd,
        optimizer = optimizer,
        loss_name ='loss_for_sbi', 
        nb_pixels = N, 
        nb_bins = nbins, 
        info_compressor=[parameters_compressor, opt_state_resnet]
)

ds = tfds.load(
  'LensingLogNormalDataset/year_10_without_noise_score_density',
  split='train',
  data_dir = ROOT_DIR / 'notebook/tensorflow_dataset'
)

ds = ds.repeat()
ds = ds.shuffle(1000)
ds = ds.map(augmentation)
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds))


update = jax.jit(model.update)



batch_loss=[]
for batch in tqdm(range(total_steps)):
    samples = next(ds_train)
    if not jnp.isnan(samples['simulation']).any():
        l, params_nd, opt_state_nd, opt_state_resnet=update(
            model_params = params_nd, 
            opt_state=opt_state_nd,
            theta = samples['theta'],
            x=samples['simulation']
        )

        batch_loss.append(l)
        if jnp.isnan(l):
          print('NaN Loss')
          break



print('######## Compute useful quantity to make the plot########')
y, _ = compressor.apply(
   parameters_compressor, opt_state_resnet, None, m_data.reshape([1,N,N,nbins])
)
nvp_sample_nd = hk.transform(
    lambda x : SmoothNPE()(x).sample(1000000, seed=hk.next_rng_key())
)
sample_nd = nvp_sample_nd.apply(
    params_nd, 
    rng = key_par, 
    x = y*jnp.ones([1000000,dim])
)

print('######## Save parameters ########')
with open(DATA_DIR / "params_nd_{}_{}.pkl".format(args.filename, l_name), "wb") as fp:
  pickle.dump(params_nd, fp)

with open(DATA_DIR / "opt_state_nd_{}_{}.pkl".format(args.filename, l_name), "wb") as fp:
  pickle.dump(opt_state_nd, fp)

with open(DATA_DIR / "sample_nd_{}_{}.pkl".format(args.filename, l_name), "wb") as fp:
  pickle.dump(sample_nd, fp)
