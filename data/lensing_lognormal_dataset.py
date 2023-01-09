import tensorflow as tf
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd= tfp.distributions
tfb = tfp.bijectors
from tensorflow_datasets.core.utils import gcs_utils
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path
import pickle
from sbids.tasks.lensinglognormal import lensingLogNormal
from sbids.tasks import get_samples_and_scores
from numpyro.handlers import seed, trace, condition

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""


class AffineCoupling(hk.Module):
  """This is the coupling layer used in the Flow."""
  def __call__(self, x, output_units, **condition_kwargs):
   
    net = hk.Linear(128)(x)   
    net = jax.nn.leaky_relu(net)
    net = hk.Linear(128)(net)
    net = jax.nn.leaky_relu(net)    
    net = hk.Linear(128)(net)
    net = jax.nn.leaky_relu(net) 
    shifter = tfb.Shift(hk.Linear(output_units)(net))
    scaler = tfb.Scale(jnp.clip(jnp.exp(hk.Linear(output_units)(net)), 1e-2, 1e2))
    
    return tfb.Chain([shifter, scaler])     


class AffineFlow(hk.Module):
    """This is a normalizing flow using the coupling layers defined
    above."""
    def __call__(self):
        
        chain = tfb.Chain([
            tfb.RealNVP(1, bijector_fn=AffineCoupling(name='aff1')),
            tfb.Permute([1,0]),
            tfb.RealNVP(1, bijector_fn=AffineCoupling(name='aff2')),
            tfb.Permute([1,0]),
            tfb.RealNVP(1, bijector_fn=AffineCoupling(name='aff3')),
            tfb.Permute([1,0]),
            tfb.RealNVP(1, bijector_fn=AffineCoupling(name='aff4')),
            tfb.Permute([1,0]),
        ])
        
        nvp = tfd.TransformedDistribution(tfd.Independent(tfd.TruncatedNormal(0.5*jnp.ones(2), 
                                                                              0.1*jnp.ones(2), 
                                                                              0.01,0.99),
                                                          reinterpreted_batch_ndims=1),
                                          bijector=chain)
        return nvp


model_sample = hk.transform(lambda n : AffineFlow()().sample(n, seed=hk.next_rng_key()))

class LensingLogNormalDatasetConfig(tfds.core.BuilderConfig):

  def __init__(self, *, N, map_size, gal_per_arcmin2, sigma_e, model_type, proposal, score_type, with_noise, **kwargs):
    v1 = tfds.core.Version("0.0.1")
    super(LensingLogNormalDatasetConfig, self).__init__(
        description=("Lensing simulations."),
        version=v1,
        **kwargs)
    self.N = N
    self.map_size = map_size
    self.gal_per_arcmin2 = gal_per_arcmin2
    self.sigma_e = sigma_e
    self.model_type = model_type
    self.proposal = proposal
    self.score_type = score_type
    self.with_noise = with_noise

class LensingLogNormalDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }
  BUILDER_CONFIGS = [LensingLogNormalDatasetConfig(name="toy_model_with_proposal_without_noise", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=30, 
                                            sigma_e=0.2,
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density', 
                                            with_noise = False),
                    LensingLogNormalDatasetConfig(name="toy_model_with_proposal_with_noise", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=30, 
                                            sigma_e=0.2,
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density', 
                                            with_noise = True),
                    LensingLogNormalDatasetConfig(name="toy_model_without_noise", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=30, 
                                            sigma_e=0.2,
                                            model_type='lognormal', 
                                            proposal = False, 
                                            score_type = 'density', 
                                            with_noise = False),
                    LensingLogNormalDatasetConfig(name="year_1_score_density", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=10, 
                                            sigma_e=0.26,
                                            model_type='lognormal', 
                                            proposal = False, 
                                            score_type = 'density',
                                            with_noise = True),
                     LensingLogNormalDatasetConfig(name="year_10_score_density", 
                                            N=128, 
                                            map_size=5, 
                                            gal_per_arcmin2=27, 
                                            sigma_e=0.26, 
                                            model_type='lognormal', 
                                            proposal = True, 
                                            score_type = 'density',
                                            with_noise = True),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
          builder=self,
          description=_DESCRIPTION,
          features=tfds.features.FeaturesDict({
              'simulation': tfds.features.Tensor(shape=[self.builder_config.N, self.builder_config.N], dtype=tf.float32),
              'theta': tfds.features.Tensor(shape=[2], dtype=tf.float32),
              'score': tfds.features.Tensor(shape=[2], dtype=tf.float32),
          }),
          supervised_keys=None,  
          homepage='https://dataset-homepage/',
          citation=_CITATION,
      )
    

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return [
        tfds.core.SplitGenerator(name=tfds.Split.TRAIN, 
                                 gen_kwargs={'size': 200000}),

    ]

  def _generate_examples(self, size):
    """Yields examples."""

    SOURCE_FILE = Path(__file__)
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"

    if self.builder_config.name == 'toy_model_with_proposal_without_noise' or self.builder_config.name == 'toy_model_with_proposal_with_noise':
      FILE = "params_proposal_lensing_simulator_toy_model.pkl"
    elif self.builder_config.name == "year_1_score_density":
      FILE = " "
    elif self.builder_config.name == "year_10_score_density":
      FILE = " "

  
    bs = 100
    if self.builder_config.proposal == True:
        a_file = open(DATA_DIR / FILE, "rb")
        parameters = pickle.load(a_file)
        thetas = model_sample.apply(parameters, rng=jax.random.PRNGKey(6543), n=size)
        thetas = thetas.reshape([-1,bs,2])
    else: 
        thetas = np.array([None]).repeat(size // bs)

    model = partial(lensingLogNormal, 
                    self.builder_config.N, 
                    self.builder_config.map_size,
                    self.builder_config.gal_per_arcmin2,
                    self.builder_config.sigma_e,
                    self.builder_config.model_type, 
                    self.builder_config.with_noise)

    @jax.jit 
    def get_batch(key, thetas):
        (_, samples), scores = get_samples_and_scores(model = model, 
                                                key = key, 
                                                batch_size = bs, 
                                                score_type = self.builder_config.score_type,
                                                thetas = thetas, 
                                                with_noise = self.builder_config.with_noise) 

        return samples['y'], samples['theta'], scores


    master_key = jax.random.PRNGKey(2948570986789)


    for i in range(size // bs):    
      key, master_key = jax.random.split(master_key)
      simu, theta, score = get_batch(key, thetas[i])  

      for j in range(bs):                                  
        yield '{}-{}'.format(i,j), {
              'simulation': simu[j],
              'theta': theta[j],
              'score': score[j]
          }
