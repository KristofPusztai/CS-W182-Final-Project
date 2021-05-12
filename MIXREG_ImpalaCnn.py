from network import build_impala_cnn
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.policies import ActorCriticPolicy
import numpy as np
import tensorflow as tf

# ImpalaCNN Policy network
class ImpalaCnn(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, mix_alpha=0.2, **kwargs):
        super(ImpalaCnn, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        ###----MIXREG IMPLEMENTATION---###
        extra_tensors = {}
        COEFF = tf.placeholder(tf.float32, [None])
        INDICES = tf.placeholder(tf.int32, [None])
        OTHER_INDICES = tf.placeholder(tf.int32, [None])
        coeff = tf.reshape(COEFF, (-1, 1, 1, 1))
        extra_tensors['coeff'] = COEFF
        extra_tensors['indices'] = INDICES
        extra_tensors['other_indices'] = OTHER_INDICES
        self.__dict__.update(extra_tensors)
        ###-----------------------------###
        with tf.variable_scope("model", reuse=reuse):
            
            extracted_features = build_impala_cnn(self.processed_obs, **kwargs)

            pi_latent = vf_latent = extracted_features

            self._value_fn = linear(vf_latent, 'vf', 1)
            
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
