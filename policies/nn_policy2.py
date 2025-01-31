import tensorflow as tf

from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
#from sac.policies.base import Policy2
from policies.base import Policy2



class NNPolicy2(Policy2, Serializable):
    def __init__(self, env_spec, observation_ph, actions,
                 scope_name=None):
        Serializable.quick_init(self, locals())

        self._observations_ph = observation_ph
        self._actions = actions
        self._scope_name = (
            tf.get_variable_scope().name if not scope_name else scope_name
        )
        super(NNPolicy2, self).__init__(env_spec)

    @overrides
    def get_action(self, observation,sub_level_actions):
        """Sample single action based on the observations."""
        return self.get_actions(observation[None],sub_level_actions)[0], {}

    @overrides
    def get_actions(self, observations,sub_level_actions):
        """Sample actions based on the observations."""
        feed_dict = {self._observations_ph: observations,self.sub_level_actions:sub_level_actions}
        actions = tf.get_default_session().run(self._actions, feed_dict)
        return actions

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        if tags:
            raise NotImplementedError
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


