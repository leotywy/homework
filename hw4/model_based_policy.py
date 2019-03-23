import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph = tf.placeholder(tf.float32, shape = (None, self._state_dim), name = 'state')
        action_ph = tf.placeholder(tf.float32, shape = (None, self._action_dim), name = 'action')
        next_state_ph = tf.placeholder(tf.float32, shape = (None, self._state_dim), name = 'next_state')

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        self.norm_state = utils.normalize(state, self._init_dataset.state_mean, self._init_dataset.state_std)
        self.norm_action = utils.normalize(action, self._init_dataset.action_mean, self._init_dataset.action_std)

        self.conc_input = tf.concat([self.norm_state, self.norm_action], 1)

        self.norm_delta_pre = utils.build_mlp(self.conc_input, self._state_dim,'dynamics_model', self._nn_layers, reuse = reuse)
        self.delta_pre = utils.unnormalize(self.norm_delta_pre, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)

        next_state_pred = state + self.delta_pre

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        self.act_dif = next_state_ph - state_ph
        self.pre_dif = next_state_pred - state_ph

        self.norm_act_dif = utils.normalize(self.act_dif, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)
        self.norm_pre_dif = utils.normalize(self.pre_dif, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)

        loss = tf.losses.mean_squared_error(self.norm_act_dif, self.norm_pre_dif)
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        action_shape = [self._num_random_action_selection, self._horizon, self._action_dim]
        sample_action = tf.random_uniform(action_shape, minval = self._action_space_low, maxval = self._action_space_high)
        cost = tf.zeros([self._num_random_action_selection])
        states = tf.stack([state_ph[0]] * self._num_random_action_selection)
        for t in range(self._horizon):
            #print(sample_action[:, t].shape)
            new_states = self._dynamics_func(states, sample_action[:, t], reuse = True)
            #print(new_states.shape)
            cost += self._cost_fn(states, sample_action[:, t], new_states)
            states = new_states

        mini_ind = tf.argmin(cost)
        best_action = sample_action[mini_ind][0]

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)
        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        feed_dict = {self._state_ph: states, self._action_ph: actions, self._next_state_ph: next_states}
        loss, _ = self._sess.run([self._loss, self._optimizer], feed_dict = feed_dict)

        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        feed_dict = {self._state_ph: np.expand_dims(state, axis = 0), self._action_ph: np.expand_dims(action, axis = 0)}

        next_state_pred = self._sess.run(self._next_state_pred, feed_dict = feed_dict)
        next_state_pred = np.squeeze(next_state_pred)

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._sess.run(self._best_action, feed_dict = {self._state_ph: state[None]})

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
