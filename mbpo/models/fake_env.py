import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:

    def __init__(self, model, config):
        self.model = model
        self.config = config

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds
    
    # TODO: 加参数变成有梯度
    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        # TODO 加参数True时需要有梯度
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        
        # 还是delta S
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info
    
    # def compute_EMD(self, u1, sigma1, u2, sigma2):
    #     ''' compute the EMD of two Gaussian Distributions u-log sigma-scale'''
    #     part1 = tf.reduce_sum(tf.reduce_sum(tf.square(u1 - u2), axis=-1), axis=-1)
    #     part2 = tf.trace(tf.square(sigma1) + tf.square(sigma2) - 2 * tf.sqrt(sigma1*tf.square(sigma2)*sigma1))
    #     return tf.sqrt(part1 + part2)
    
    # def union_step(self, obs_ph, act_ph, if_c=False):

    #     inputs = tf.concat([obs_ph, act_ph], axis=1)
        
    #     def create_forward(inputs, obs_ph):
            
    #         # TODO: 此处可以直接算出b 和 a，用不同的self.model即可，这一个step就可以是完整的计算图
    #         ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
                
    #         ensemble_model_stds = tf.sqrt(ensemble_model_vars)
            
    #         num_models, batch_size, _ = ensemble_model_means.shape
    #         batch_size = int(batch_size)
    #         ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
    #         ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

    #         batch_inds = np.arange(0, batch_size).reshape((batch_size, 1))
    #         model_inds = self.model.random_inds(batch_size).reshape((batch_size, 1))
            
    #         idx = np.hstack((model_inds,batch_inds))
            
    #         model_means = tf.gather_nd(ensemble_samples,idx)
    #         model_stds = tf.gather_nd(ensemble_model_stds,idx)

            
    #         return model_means, model_stds, ensemble_samples, ensemble_model_stds, batch_size
        
    #     model_means, model_stds, ensemble_means, ensemble_model_stds, batch_size = create_forward(inputs, obs_ph)

    #     uncertainty = tf.reduce_sum(self.compute_EMD(model_means, model_stds, ensemble_means, ensemble_model_stds)) / batch_size / (self.model.num_nets - 1)

    #     # p_theta = tf.distributions.Normal(model_means, model_stds)
    #     # p_other_theta = tf.distributions.Normal(ensemble_means, ensemble_model_stds)
    #     # uncertainty = (tf.reduce_sum(tf.distributions.kl_divergence(p_theta, p_other_theta)) / batch_size / (self.model.num_nets - 1) + \
    #     #     tf.reduce_sum(tf.distributions.kl_divergence(p_other_theta, p_theta)) / batch_size / (self.model.num_nets - 1)) / 2
    #     return uncertainty, model_means, model_stds
        
        
    # ## for debugging computation graph
    # def step_ph(self, obs_ph, act_ph, deterministic=False):
    #     assert len(obs_ph.shape) == len(act_ph.shape)

    #     inputs = tf.concat([obs_ph, act_ph], axis=1)
    #     # inputs = np.concatenate((obs, act), axis=-1)
    #     ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
    #     # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
    #     ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
    #     # ensemble_model_means[:,:,1:] += obs_ph
    #     ensemble_model_stds = tf.sqrt(ensemble_model_vars)
    #     # ensemble_model_stds = np.sqrt(ensemble_model_vars)

    #     if deterministic:
    #         ensemble_samples = ensemble_model_means
    #     else:
    #         # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
    #         ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

    #     samples = ensemble_samples[0]

    #     rewards, next_obs = samples[:,:1], samples[:,1:]
    #     terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
    #     info = {}

    #     return next_obs, rewards, terminals, info

    def close(self):
        pass



