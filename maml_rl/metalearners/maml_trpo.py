import torch
import time
from torch.nn.utils.clip_grad import clip_grad_value_

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.samplers import MultiTaskSampler
from maml_rl.metalearners.base import GradientBasedMetaLearner
import numpy as np
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.utils.global_tensor_val import get_value


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].
    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).
    fast_lr : float
        Step-size for the inner loop update/fast adaptation.
    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.
    first_order : bool
        If `True`, then the first order approximation of MAML is applied.
    device : str ("cpu" or "cuda")
        Name of the device for the optimization.
    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cuda'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order
        self.global_episode = -1
        self.global_episode_val = -1

    async def adapt(self, train_futures, first_order=None, Adv=1, writer=None):
        if first_order is None:
            first_order = self.first_order
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            ##print(get_value('Adv'))
            inner_loss = reinforce_loss(self.policy,
                                        await futures,
                                        params=params,
                                        writer=writer,
                                        global_episode = self.global_episode)
            self.global_episode+=1
            inner_loss= Adv * inner_loss
            ##print(Adv)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None, Adv=1, Outer_loop=False, writer = None):
        first_order = (old_pi is not None) or self.first_order
        params = await self.adapt(train_futures,
                                  first_order=first_order, Adv=Adv)
        params_pure = await self.adapt(train_futures,
                                  first_order=first_order, Adv=1)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            
            self.global_episode_val+=1
            if writer != None and self.global_episode_val >= 0:
               writer.add_scalar("reward/ep_valid", sum(valid_episodes.rewards).mean().item(), self.global_episode_val)
              
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
#----
            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
# ------------------------------------------------------------------------------------------------
            pi_new = self.policy(valid_episodes.observations.view((-1, *valid_episodes.observation_shape)),
                params=params)
            log_probs = pi_new.log_prob(valid_episodes.actions.view((-1, *valid_episodes.action_shape)))
            log_probs = log_probs.view(len(valid_episodes), valid_episodes.batch_size)

            Reinforce_loss = -weighted_mean(log_probs * valid_episodes.advantages,
                            lengths=valid_episodes.lengths)
# ------------------------------------------------------------------------------------------------
            pi_pure = self.policy(valid_episodes.observations.view((-1, *valid_episodes.observation_shape)),
                params=params_pure)
            log_probs_pure = pi_pure.log_prob(valid_episodes.actions.view((-1, *valid_episodes.action_shape)))
            log_probs_pure = log_probs_pure.view(len(valid_episodes), valid_episodes.batch_size)

            Reinforce_loss_pure = -weighted_mean(log_probs_pure * valid_episodes.advantages,
                            lengths=valid_episodes.lengths)
# ------------------------------------------------------------------------------------------------
            if Outer_loop!=False:
                losses*=Adv
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi, Reinforce_loss.mean(), Reinforce_loss_pure.mean()


    def step(self,
             train_futures,
             valid_futures,
             Base = False,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5,
             Adv=1,
             Epoch=1,
             Outer_loop=False,
             Epoch_big=0,
             writer = None,
             global_episode = 1,
             global_episode_val = 0):
        num_tasks = len(train_futures[0])
        logs = {}
        self.global_episode = global_episode
        self.global_episode_val = global_episode_val
        
        if Epoch_big!=0:
            logs['epoch_big'] = Epoch_big
        self.Epoch_big = Epoch_big
        logs['epoch'] = Epoch
        logs['time'] = time.asctime( time.localtime(time.time()) )

        # Compute the surrogate loss
        old_losses, old_kls, old_pis, _, _ = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None, Adv=Adv, Outer_loop=Outer_loop)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        # logs['loss_before'] = to_numpy(old_losses)
        # logs['kl_before'] = to_numpy(old_kls)
        global_episode = self.global_episode
        global_episode_val = self.global_episode_val

        old_loss = sum(old_losses) / num_tasks
        logs['loss_before'] = to_numpy(old_loss).tolist()
        Loss = to_numpy(old_loss).tolist()

        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())
        

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            losses, kls, _, new_losses, pure_losses = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi, Adv=Adv, Outer_loop=Outer_loop)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(sum(losses) / num_tasks).tolist()
                logs['improvement'] = to_numpy((sum(losses) / num_tasks) - old_loss).tolist()
                logs['reinforcementloss_after'] = to_numpy(sum(new_losses) / num_tasks).tolist()
                if Outer_loop!=False:
                   logs['reinforcementloss_after_outer'] = to_numpy(sum(new_losses) / num_tasks*Adv).tolist()
                logs['pure_loss'] = to_numpy(sum(pure_losses) / num_tasks).tolist()
                if writer != None and Epoch_big == 0:
                   Num=Epoch
                else:
                   Num=Epoch+Base*(Epoch_big-1)
                   
                writer.add_scalar("learner/kl", kl.item(), Num)
                writer.add_scalar("learner/surrogate_loss", to_numpy(sum(losses) / num_tasks).tolist(), Num)
                writer.add_scalar("learner/reinforcement_loss", to_numpy(sum(new_losses) / num_tasks).tolist(), Num)
                writer.add_scalar("learner/pure_loss", to_numpy(sum(pure_losses) / num_tasks).tolist(), Num)
                if Outer_loop!=False:
                    writer.add_scalar("learner/reinforcementloss_after_outer", to_numpy(sum(new_losses) / num_tasks*Adv).tolist(), Num)
                    
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
            
        return logs, global_episode, global_episode_val, sum(losses) / num_tasks


    def step_attacker_with_opt(self,
             train_futures,
             valid_futures,
             Base = False,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5,
             Adv=1,
             Epoch=0,
             optimizer=None,
             clip_value=False,
             Outer_loop=False,
             Epoch_big=0,
             writer=None):
        num_tasks = len(train_futures[0])
        logs = {}

        if Epoch_big!=0:
              logs['epoch_big'] = Epoch_big
        logs['epoch'] = Epoch
        logs['time'] = time.asctime( time.localtime(time.time()) )
       
        # Compute the surrogate loss
        old_losses, old_kls, old_pis, new_losses, pure_losses = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None, Adv=Adv, Outer_loop=Outer_loop)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        old_loss = sum(old_losses) / num_tasks
        #  + 0.1*torch.norm(Adv-1,p=2)
        logs['attacker_loss_surrogate'] = to_numpy(old_loss).tolist()

        new_losses = sum(new_losses) / num_tasks
        if Outer_loop!=False:
           new_losses*=Adv
        logs['attacker_loss'] = to_numpy(new_losses).tolist()
        logs['attacker_pure_loss'] = to_numpy(sum(pure_losses) / num_tasks).tolist()

        #Loss = to_numpy(old_loss).tolist()
        logs['Adv_before'] = to_numpy(Adv).tolist()
        
        Adv_loss = -new_losses
        optimizer.zero_grad()
        Adv_loss.backward()
        if clip_value!=False:
           #torch.nn.utils.clip_grad_norm_(Adv, max_norm=clip_value)
           torch.nn.utils.clip_grad_value_(Adv, clip_value=clip_value)
        optimizer.step()

        logs['Adv_after'] = to_numpy(Adv).tolist()
        
        grads = Adv.grad
        ##print("Adv is ", Adv, "lr_Adv is ", lr_Adv, "grads is ", grads)
        logs['grads'] = grads.item()
        if writer != None and Epoch_big == 0:
            Num=Epoch
        else:
            Num=Epoch+Base*(Epoch_big-1)
            
        writer.add_scalar("attacker/Adv", to_numpy(Adv).tolist(), Num)
        writer.add_scalar("attacker/attacker_loss", to_numpy(new_losses).tolist(), Num)
        writer.add_scalar("attacker/attacker_pure_loss", to_numpy(sum(pure_losses) / num_tasks).tolist(), Num)
        

        return logs, Adv, optimizer, grads
