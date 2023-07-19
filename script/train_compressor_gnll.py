from tqdm import tqdm
import pickle
import tensorboard
import jax
import jax.numpy as jnp

from flax.metrics import tensorboard

from haiku._src.nets.resnet import ResNet18
import optax
import haiku as hk

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfd = tfp.distributions

from sbi_lens.gen_dataset.lensing_lognormal_dataset import LensingLogNormalDataset
from sbi_lens.gen_dataset.utils import augmentation_noise, augmentation_flip
from sbi_lens.config import config_lsst_y_10
import pickle


'unset XLA_FLAGS'

import logging


import tensorflow_probability as tfp; tfp = tfp.substrates.jax

# this prints a WARNING

tfp.distributions.TransformedDistribution(
    tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity()
)

logger = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

# this does not print a WARNING

tfp.distributions.TransformedDistribution(tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity())

####################################################################
# Define configuration
####################################################################
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

tf.random.set_seed(1)

augmentation = lambda example: augmentation_flip(
    augmentation_noise(example=example,
                       N=N,
                       map_size=map_size,
                       sigma_e=sigma_e,
                       gal_per_arcmin2=gals_per_arcmin2,
                       nbins=nbins,
                       a=a,
                       b=b,
                       z0=z0))
####################################################################
# Build compressor
####################################################################
dim = 6
aug_dim = int((dim * (dim + 1) / 2)+dim)

compressor = hk.transform_with_state(
    lambda y : ResNet18(dim)(y, is_training=True)
  )

parameters_resnet, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, N, N, nbins]))

####################################################################
# Create Density Estimetor for the compressor
####################################################################
class ConditionalMultivariateGaussian(hk.Module):

  def __call__(self, x, dim=6):
    net = jax.nn.leaky_relu(hk.Linear(256)(x))
    net = jax.nn.tanh(hk.Linear(128)(net))
    gaussian_mu = hk.Linear(dim)(net)
    gaussian_tril = hk.Linear(dim * (dim + 1) // 2)(net)
    
    dist = tfd.MultivariateNormalTriL(loc=gaussian_mu, 
                  scale_tril=tfp.bijectors.FillScaleTriL(
                              diag_bijector=tfp.bijectors.Softplus(low=1e-3)
                            )(gaussian_tril))                                  
    
    return dist

model= hk.without_apply_rng(hk.transform(lambda theta, x : ConditionalMultivariateGaussian()(x).log_prob(theta).squeeze()))
params_mdn =model.init(jax.random.PRNGKey(0),  theta=0.5*jnp.ones([1,6]), x=0.5*jnp.ones([1, 6]))
####################################################################
# Marge parameters
####################################################################
parameters_compressor= hk.data_structures.merge(
    parameters_resnet,
    params_mdn
  )

####################################################################
# Dataset
####################################################################
ds = tfds.load('LensingLogNormalDataset/year_10_without_noise_score_density', 
               split='train[:80000]', 
               data_dir = 'tensorflow_dataset')

ds = ds.repeat()
ds = ds.shuffle(1000)
ds = ds.map(augmentation)
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds))

####################################################################
# Define configuration for the training
####################################################################
total_steps=150_000
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.0001,
    boundaries_and_scales={
        int(20000): 0.1,
        int(40000): 0.1,
        int(60000): 0.1,
        int(80000): 0.7,
        int(100000): 0.7,
        int(120000): 0.7,
    })

optimizer = optax.chain(
  optax.clip(1.0),
  optax.adam(learning_rate=lr_scheduler)
)

opt_state_c = optimizer.init(parameters_compressor)
####################################################################
# Define loss and update function 
####################################################################
def loss_gnll(params, theta, x, state_resnet):
    y, opt_state_resnet = compressor.apply(params, state_resnet, None, x)
    log_prob =  model.apply(
          params, 
          theta, 
          y)
    return -jnp.mean(log_prob), opt_state_resnet

@jax.jit    
def update(
    params,
    opt_state,
    theta,
    x,
    state_resnet=None
  ):

    (loss, opt_state_resnet), grads = jax.value_and_grad(
      loss_gnll,
      has_aux=True
    )(params, theta, x, state_resnet)

    updates, new_opt_state = optimizer.update(
      grads,
      opt_state
    )

    new_params = optax.apply_updates(
      params,
      updates
    )

    return loss, new_params, new_opt_state, opt_state_resnet
####################################################################
# TRAINING
####################################################################
store_loss = []
summary_writer = tensorboard.SummaryWriter('logs/')

for batch in tqdm(range(total_steps + 1)):
  ex = next(ds_train)
  if not jnp.isnan(ex['simulation']).any():
        l, parameters_compressor, opt_state_c, opt_state_resnet = update(
            params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex['theta'],
            x=ex['simulation'],
            state_resnet=opt_state_resnet)
        summary_writer.scalar('train_loss_var_gnll_smalllearning_150iter', l, batch)
        summary_writer.scalar('learning_rate_var_gnll_smalllearning_150iter',lr_scheduler(batch),  batch)
        store_loss.append(l)
        if jnp.isnan(l):
          print('NaN Loss')
          break
####################################################################
# SAVE FILE
####################################################################
with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/params_nd_compressor_gnll.pkl", "wb") as fp:
  pickle.dump(parameters_compressor, fp)

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/opt_state_resnet_gnll.pkl", "wb") as fp:
  pickle.dump(opt_state_resnet, fp)

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/loss_compressor_gnll.pkl", "wb") as fp:
  pickle.dump(jnp.asarray(store_loss), fp)
