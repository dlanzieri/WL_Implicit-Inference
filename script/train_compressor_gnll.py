
from functools import partial
from tqdm import tqdm
import pickle
from pathlib import Path
import tensorboard
import numpy as np
import jax
import jax.numpy as jnp

import flax.linen as nn
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


dim = 6
aug_dim = int((dim * (dim + 1) / 2)+dim)

# def aug_compressor(y):
#     net_0=ResNet18(dim)(y,is_training=True)
#     net=jax.nn.leaky_relu(net_0)
#     net=hk.Linear(128)(net)
#     net=jnp.tanh(net)
#     net=hk.Linear(aug_dim)(net)
#     return net  


def aug_compressor(y):
    net_0=ResNet18(128)(y,is_training=True)
    net=jax.nn.leaky_relu(net_0)
    net=hk.Linear(128)(net)
    net=jnp.tanh(net)
    net=hk.Linear(aug_dim)(net)
    return net

compressor =hk.transform_with_state(aug_compressor)


parameters_compressor, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, N, N, nbins]))

ds = tfds.load('LensingLogNormalDataset/year_10_without_noise_score_density', 
               split='train[:80000]', 
               data_dir = 'tensorflow_dataset')




ds = ds.repeat()
ds = ds.shuffle(1000)
ds = ds.map(augmentation)
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds))



total_steps=20_000
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.001,
    boundaries_and_scales={
        int(5000): 0.1,
        int(10000): 0.1,
        int(15000): 0.1,
        int(20000): 0.7,
        int(30000): 0.7,
    })

optimizer_c = optax.adam(learning_rate=lr_scheduler)
opt_state_c = optimizer_c.init(parameters_compressor)




def loss_gnll(params, theta, x, state_resnet):
    y, opt_state_resnet = compressor.apply(params, state_resnet, None, x)
    gmu = y[..., :6]
    gtril = y[..., 6:]
    dist = tfd.MultivariateNormalTriL(
        loc=gmu,
        scale_tril=tfp.bijectors.FillScaleTriL(
            diag_bijector=tfp.bijectors.Softplus(),diag_shift=1e-03)(gtril)
            )
    return -jnp.mean(dist.log_prob(theta)), opt_state_resnet



def update(
    model_params,
    opt_state,
    theta,
    x,
    state_resnet=None
  ):

    (loss, opt_state_resnet), grads = jax.value_and_grad(
      loss_gnll,
      has_aux=True
    )(model_params, theta, x, state_resnet)

    updates, new_opt_state = optimizer_c.update(
      grads,
      opt_state
    )

    new_params = optax.apply_updates(
      model_params,
      updates
    )

    return loss, new_params, new_opt_state, opt_state_resnet


store_loss = []


summary_writer = tensorboard.SummaryWriter('logs/')

for batch in tqdm(range(total_steps + 1)):
  ex = next(ds_train)
  if not jnp.isnan(ex['simulation']).any():
        l, parameters_compressor, opt_state_c, opt_state_resnet = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex['theta'],
            x=ex['simulation'],
            state_resnet=opt_state_resnet)
        summary_writer.scalar('train_loss_net27_tscript_nobott_res', l, batch)
        summary_writer.scalar('learning_rate_net27_tscript_nobott_res',lr_scheduler(batch),  batch)
        store_loss.append(l)
        if jnp.isnan(l):
          print('NaN Loss')
          break
            


import pickle

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/net_27/params_nd_compressor_gnll_net27_tscript_nobott_res.pkl", "wb") as fp:
  pickle.dump(parameters_compressor, fp)

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/net_27/opt_state_resnet_gnll_net27_tscript_nobott_res.pkl", "wb") as fp:
  pickle.dump(opt_state_resnet, fp)

with open("/gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/data/params_compressor/net_27/loss_compressor_gnll_net27_tscript_nobott_res.pkl", "wb") as fp:
  pickle.dump(jnp.asarray(store_loss), fp)
