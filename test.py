import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers, activations
from keras.engine import InputSpec


def tfPrint(d, T): return tf.print(input_=T, data=[T, tf.shape(T)], message=d)


def to_list(x):
    '''This normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]

def _time_distributed_dense(x, w, b=None, dropout=None,
                        input_dim=None, output_dim=None,
                        timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

class Layer(object):
    '''Abstract base layer class.
    # Properties
        name: string, must be unique within a model.
        input_spec: list of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        trainable: boolean, whether the layer weights
            will be updated during training.
        uses_learning_phase: whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        input_shape: shape tuple. Provided for convenience,
            but note that there may be cases in which this
            attribute is ill-defined (e.g. a shared layer
            with multiple input shapes), in which case
            requesting `input_shape` will raise an Exception.
            Prefer using `layer.get_input_shape_for(input_shape)`,
            or `layer.get_input_shape_at(node_index)`.
        output_shape: shape tuple. See above.
        inbound_nodes: list of nodes.
        outbound_nodes: list of nodes.
        supports_masking: boolean
        input, output: input/output tensor(s). Note that if the layer is used
            more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: same as above, for masks.
        trainable_weights: list of variables.
        non_trainable_weights: list of variables.
        regularizers: list of regularizers.
        constraints: dict mapping weights to constraints.
    # Methods
        call(x, mask=None): where the layer's logic lives.
        __call__(x, mask=None): wrapper around the layer logic (`call`).
            if x is a Keras tensor:
                - connect current layer with last layer from tensor:
                    `self.add_inbound_node(last_layer)`
                - add layer to tensor history
            if layer is not built:
                - build from x._keras_shape
        get_weights()
        set_weights(weights)
        get_config()
        count_params()
        get_output_shape_for(input_shape)
        compute_mask(x, mask)
        get_input_at(node_index)
        get_output_at(node_index)
        get_input_shape_at(node_index)
        get_output_shape_at(node_index)
        get_input_mask_at(node_index)
        get_output_mask_at(node_index)
    # Class Methods
        from_config(config)
    # Internal methods:
        build(input_shape)
        add_inbound_node(layer, index=0)
        create_input_layer()
        assert_input_compatibility()
    '''
    def __init__(self, **kwargs):
        # these properties should have been set
        # by the child class, as appropriate.
        if not hasattr(self, 'input_spec'):
            self.input_spec = None
        if not hasattr(self, 'supports_masking'):
            self.supports_masking = False
        if not hasattr(self, 'uses_learning_phase'):
            self.uses_learning_phase = False

        # these lists will be filled via successive calls
        # to self.add_inbound_node()
        self.inbound_nodes = []
        self.outbound_nodes = []

        # these properties will be set upon call of self.build(),
        # which itself will be calld upon self.add_inbound_node if necessary.
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.regularizers = []
        self.constraints = {}  # dict {tensor: constraint instance}
        self.built = False

        # these properties should be set by the user via keyword arguments.
        # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'input_dtype',
                          'name',
                          'trainable',
                          'create_input_layer'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
            # in this case we will create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                batch_input_shape = (None,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape
            input_dtype = kwargs.get('input_dtype', K.floatx())
            self.input_dtype = input_dtype
            if 'create_input_layer' in kwargs:
                self.create_input_layer(batch_input_shape, input_dtype)

    def create_input_layer(self, batch_input_shape,
                           input_dtype=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower() + '_input_'
            name = prefix + str(K.get_uid(prefix))
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        # instantiate the input layer
        x = Input(batch_shape=batch_input_shape,
                  dtype=input_dtype, name=name)
        # this will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created.
        self(x)

    def assert_input_compatibility(self, input):
        '''This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.
        '''
        if not self.input_spec:
            return True
        assert type(self.input_spec) is list, ('input_spec must be a list of ' +
                                               'InputSpec instances. Found: ' +
                                               str(self.input_spec))
        inputs = to_list(input)
        if len(self.input_spec) > 1:
            if len(inputs) != len(self.input_spec):
                raise Exception('Layer ' + self.name + ' expects ' +
                                str(len(self.input_spec)) + ' inputs, '
                                'but it received ' + str(len(inputs)) +
                                ' input tensors. Input received: ' +
                                str(input))
        for input_index, (x, spec) in enumerate(zip(inputs, self.input_spec)):
            if spec is None:
                continue

            # check ndim
            if spec.ndim is not None:
                if type(spec.ndim) is str:
                    int_ndim = spec.ndim[:spec.ndim.find('+')]
                    ndim = int(int_ndim)
                    if K.ndim(x) < ndim:
                        raise Exception('Input ' + str(input_index) +
                                        ' is incompatible with layer ' +
                                        self.name + ': expected ndim >= ' +
                                        str(ndim) + ', found ndim=' +
                                        str(K.ndim(x)))
                else:
                    if K.ndim(x) != spec.ndim:
                        raise Exception('Input ' + str(input_index) +
                                        ' is incompatible with layer ' +
                                        self.name + ': expected ndim=' +
                                        str(spec.ndim) + ', found ndim=' +
                                        str(K.ndim(x)))
            if spec.dtype is not None:
                if K.dtype(x) != spec.dtype:
                    raise Exception('Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected dtype=' +
                                    str(spec.dtype) + ', found dtype=' +
                                    str(K.dtype(x)))
            if spec.shape is not None:
                if hasattr(x, '_keras_shape'):
                    x_shape = x._keras_shape
                elif hasattr(K, 'int_shape'):
                    # tensorflow shape inference
                    x_shape = K.int_shape(x)
                else:
                    continue
                for spec_dim, dim in zip(spec.shape, x_shape):
                    if spec_dim is not None:
                        if spec_dim != dim:
                            raise Exception('Input ' + str(input_index) +
                                            ' is incompatible with layer ' +
                                            self.name + ': expected shape=' +
                                            str(spec.shape) + ', found shape=' +
                                            str(x_shape))

    def call(self, x, mask=None):
        '''This is where the layer's logic lives.
        # Arguments
            x: input tensor, or list/tuple of input tensors.
            mask: a masking tensor (or list of tensors). Used mainly in RNNs.
        # Returns:
            A tensor or list/tuple of tensors.
        '''
        return x

    def __call__(self, x, mask=None):
        '''Wrapper around self.call(), for handling
        internal Keras references.
        If a Keras tensor is passed:
            - we call self.add_inbound_node()
            - if necessary, we `build` the layer to match
                the _keras_shape of the input(s)
            - we update the _keras_shape of every input tensor with
                its new shape (obtained via self.get_output_shape_for).
                This is done as part of add_inbound_node().
            - we update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of add_inbound_node().
        # Arguments
            x: can be a tensor or list/tuple of tensors.
            mask: tensor or list/tuple of tensors.
        '''
        if not self.built:
            # raise exceptions in case the input is not compatible
            # with the input_spec specified in the layer constructor
            self.assert_input_compatibility(x)

            # collect input shapes to build layer
            input_shapes = []
            for x_elem in to_list(x):
                if hasattr(x_elem, '_keras_shape'):
                    input_shapes.append(x_elem._keras_shape)
                elif hasattr(K, 'int_shape'):
                    input_shapes.append(K.int_shape(x_elem))
                else:
                    raise Exception('You tried to call layer "' + self.name +
                                    '". This layer has no information'
                                    ' about its expected input shape, '
                                    'and thus cannot be built. '
                                    'You can build it manually via: '
                                    '`layer.build(batch_input_shape)`')
            if len(input_shapes) == 1:
                self.build(input_shapes[0])
            else:
                self.build(input_shapes)
            self.built = True

        # raise exceptions in case the input is not compatible
        # with the input_spec set at build time
        self.assert_input_compatibility(x)
        # build and connect layer
        input_added = False
        input_tensors = to_list(x)

        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for input_tensor in input_tensors:
            if hasattr(input_tensor, '_keras_history') and input_tensor._keras_history:
                # this is a Keras tensor
                previous_layer, node_index, tensor_index = input_tensor._keras_history
                inbound_layers.append(previous_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers = None
                break
        if inbound_layers:
            # this will call layer.build() if necessary
            self.add_inbound_node(inbound_layers, node_indices, tensor_indices)
            input_added = True

        # get the output tensor to be returned
        if input_added:
            # output was already computed when calling self.add_inbound_node
            outputs = self.inbound_nodes[-1].output_tensors
            # if single output tensor: return it,
            # else return a list (at least 2 elements)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        else:
            # this case appears if the input was not a Keras tensor
            return self.call(x, mask)

    def add_inbound_node(self, inbound_layers,
                         node_indices=None, tensor_indices=None):
        '''
        # Arguments:
            inbound_layers: can be a layer instance
                or a list/tuple of layer instances.
            node_indices: integer (or list of integers).
                The input layer might have a number of
                parallel output streams;
                this is the index of the stream (in the input layer)
                where to connect the current layer.
            tensor_indices: integer or list of integers.
                The output of the inbound node might be a list/tuple
                of tensor, and we might only be interested in one sepcific entry.
                This index allows you to specify the index of the entry in the output list
                (if applicable). "None" means that we take all outputs (as a list).
        '''
        inbound_layers = to_list(inbound_layers)
        if not node_indices:
            node_indices = [0 for _ in range(len(inbound_layers))]
        else:
            node_indices = to_list(node_indices)
            assert len(node_indices) == len(inbound_layers)
        if not tensor_indices:
            tensor_indices = [0 for _ in range(len(inbound_layers))]
        else:
            tensor_indices = to_list(tensor_indices)

        if not self.built:
            # collect input_shapes for call to build()
            input_shapes = []
            for layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
                input_shapes.append(layer.inbound_nodes[node_index].output_shapes[tensor_index])
            # call build()
            if len(input_shapes) == 1:
                self.build(input_shape=input_shapes[0])
            else:
                self.build(input_shape=input_shapes)
            self.built = True
        # creating the node automatically updates self.inbound_nodes
        # as well as outbound_nodes on inbound layers.
        Node.create_node(self, inbound_layers, node_indices, tensor_indices)

    def get_output_shape_for(self, input_shape):
        '''Computes the output shape of the layer given
        an input shape (assumes that the layer will be built
        to match that input shape).
        # Arguments
            input_shape: shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        '''
        return input_shape

    def compute_mask(self, input, input_mask=None):
        '''Computes an output masking tensor, given an input tensor
        (or list thereof) and an input mask (or list thereof).
        # Arguments
            input: tensor or list of tensors.
            input_mask: tensor or list of tensors.
        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        '''
        if not hasattr(self, 'supports_masking') or not self.supports_masking:
            if input_mask is not None:
                if type(input_mask) is list:
                    if any(input_mask):
                        raise Exception('Layer ' + self.name + ' does not support masking, ' +
                                        'but was passed an input_mask: ' + str(input_mask))
                else:
                    raise Exception('Layer ' + self.name + ' does not support masking, ' +
                                    'but was passed an input_mask: ' + str(input_mask))
            # masking not explicitly supported: return None as mask
            return None
        # if masking is explictly supported, by default
        # carry over the input mask
        return input_mask

    def build(self, input_shape):
        '''Creates the layer weights.
        Must be implemented on all layers that have weights.
        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        '''
        self.built = True

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        '''Retrieves an attribute (e.g. input_tensors) from a node.
        # Arguments
            node_index: integer index of the node from which
                to retrieve the attribute
            attr: exact node attribute name
            attr_name: human-readable attribute name, for error messages
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined ' + attr_name + '.')
        if not len(self.inbound_nodes) > node_index:
            raise Exception('Asked to get ' + attr_name +
                            ' at node ' + str(node_index) +
                            ', but the layer has only ' +
                            str(len(self.inbound_nodes)) + ' inbound nodes.')
        values = getattr(self.inbound_nodes[node_index], attr)
        if len(values) == 1:
            return values[0]
        else:
            return values

    def get_input_shape_at(self, node_index):
        '''Retrieves the input shape(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_shapes',
                                                 'input shape')

    def get_output_shape_at(self, node_index):
        '''Retrieves the output shape(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')

    def get_input_at(self, node_index):
        '''Retrieves the input tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_tensors',
                                                 'input')

    def get_output_at(self, node_index):
        '''Retrieves the output tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')

    def get_input_mask_at(self, node_index):
        '''Retrieves the input mask tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_masks',
                                                 'input mask')

    def get_output_mask_at(self, node_index):
        '''Retrieves the output mask tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_masks',
                                                 'output mask')

    @property
    def input(self):
        '''Retrieves the input tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) > 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer input" '
                            'is ill-defined. '
                            'Use `get_input_at(node_index)` instead.')
        elif not self.inbound_nodes:
            raise Exception('Layer ' + self.name +
                            ' is not connected, no input to return.')
        return self._get_node_attribute_at_index(0, 'input_tensors',
                                                 'input')

    def set_input(self, input_tensor, shape=None):
        if len(self.inbound_nodes) > 1:
            raise Exception('Cannot `set_input` for layer ' + self.name +
                            ' because it has more than one inbound connection.')
        if len(self.inbound_nodes) == 1:
            # check that the inbound node is an Input node
            if self.inbound_nodes[0].inbound_layers:
                warnings.warn('You are manually setting the input for layer ' +
                              self.name + ' but it is not an Input layer. '
                              'This will cause part of your model '
                              'to be disconnected.')
        if self.outbound_nodes:
            warnings.warn('You are manually setting the input for layer ' +
                          self.name + ' but it has ' +
                          str(len(self.outbound_nodes)) +
                          ' outbound layers. '
                          'This will cause part of your model '
                          'to be disconnected.')
        if not shape:
            if hasattr(K, 'int_shape'):
                shape = K.int_shape(input_tensor)
            else:
                raise Exception('`set_input` needs to know the shape '
                                'of the `input_tensor` it receives, but '
                                'Keras was not able to infer it automatically.'
                                ' Specify it via: '
                                '`model.set_input(input_tensor, shape)`')
        # reset layer connections
        self.inbound_nodes = []
        self.outbound_nodes = []
        input_shape = tuple(shape)
        self.build(input_shape=input_shape)

        # set Keras tensor metadata
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (None, 0, 0)
        input_tensor._keras_shape = input_shape

        output_tensors = to_list(self.call(input_tensor))
        output_shapes = to_list(self.get_output_shape_for(input_shape))
        output_masks = to_list(self.compute_mask(input_tensor, None))

        for i, output_tensor in enumerate(output_tensors):
            output_tensor._keras_history = (self, 0, i)
            output_tensor._keras_shape = output_shapes[i]
            output_tensor._uses_learning_phase = self.uses_learning_phase

        # create node
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=output_tensors,
             input_masks=[None],
             output_masks=output_masks,
             input_shapes=[input_shape],
             output_shapes=output_shapes)

    @property
    def output(self):
        '''Retrieves the output tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer output" '
                            'is ill-defined. '
                            'Use `get_output_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'output_tensors',
                                                 'output')

    @property
    def input_mask(self):
        '''Retrieves the input mask tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer input mask" '
                            'is ill-defined. '
                            'Use `get_input_mask_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'input_masks',
                                                 'input mask')

    @property
    def output_mask(self):
        '''Retrieves the output mask tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer output mask" '
                            'is ill-defined. '
                            'Use `get_output_mask_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'output_masks',
                                                 'output mask')

    @property
    def input_shape(self):
        '''Retrieves the input shape tuple(s) of a layer. Only applicable
        if the layer has one inbound node,
        or if all inbound nodes have the same input shape.
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined input shape.')
        all_input_shapes = set([str(node.input_shapes) for node in self.inbound_nodes])
        if len(all_input_shapes) == 1:
            input_shapes = self.inbound_nodes[0].input_shapes
            if len(input_shapes) == 1:
                return input_shapes[0]
            else:
                return input_shapes
        else:
            raise Exception('The layer "' + str(self.name) +
                            ' has multiple inbound nodes, ' +
                            'with different input shapes. Hence ' +
                            'the notion of "input shape" is ' +
                            'ill-defined for the layer. ' +
                            'Use `get_input_shape_at(node_index)` instead.')

    @property
    def output_shape(self):
        '''Retrieves the output shape tuple(s) of a layer. Only applicable
        if the layer has one inbound node,
        or if all inbound nodes have the same output shape.
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined output shape.')
        all_output_shapes = set([str(node.output_shapes) for node in self.inbound_nodes])
        if len(all_output_shapes) == 1:
            output_shapes = self.inbound_nodes[0].output_shapes
            if len(output_shapes) == 1:
                return output_shapes[0]
            else:
                return output_shapes
        else:
            raise Exception('The layer "' + str(self.name) +
                            ' has multiple inbound nodes, ' +
                            'with different output shapes. Hence ' +
                            'the notion of "output shape" is ' +
                            'ill-defined for the layer. ' +
                            'Use `get_output_shape_at(node_index)` instead.')

    def set_weights(self, weights):
        '''Sets the weights of the layer, from Numpy arrays.
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        params = self.trainable_weights + self.non_trainable_weights
        if len(params) != len(weights):
            raise Exception('You called `set_weights(weights)` on layer "' + self.name +
                            '" with a  weight list of length ' + str(len(weights)) +
                            ', but the layer was expecting ' + str(len(params)) +
                            ' weights. Provided weights: ' + str(weights))
        for p, w in zip(params, weights):
            if K.get_value(p).shape != w.shape:
                raise Exception('Layer weight shape ' +
                                str(K.get_value(p).shape) +
                                ' not compatible with '
                                'provided weight shape ' + str(w.shape))
            K.set_value(p, w)

    def get_weights(self):
        '''Returns the current weights of the layer,
        as a list of numpy arrays.
        '''
        params = self.trainable_weights + self.non_trainable_weights
        weights = []
        for p in params:
            weights.append(K.get_value(p))
        return weights

    def get_config(self):
        '''Returns a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by Container (one layer of abstraction above).
        '''
        config = {'name': self.name,
                  'trainable': self.trainable}
        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape
        if hasattr(self, 'input_dtype'):
            config['input_dtype'] = self.input_dtype
        return config

    @classmethod
    def from_config(cls, config):
        '''This method is the reverse of get_config,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Container), nor weights (handled by `set_weights`).
        # Arguments
            config: a Python dictionary, typically the
                output of get_config.
        '''
        return cls(**config)

    def count_params(self):
        '''Returns the total number of floats (or ints)
        composing the weights of the layer.
        '''
        if not self.built:
            if self.__class__.__name__ in {'Sequential', 'Graph'}:
                self.build()
            else:
                raise Exception('You tried to call `count_params` on ' +
                                self.name + ', but the layer isn\'t built. '
                                'You can build it manually via: `' +
                                self.name + '.build(batch_input_shape)`.')
        return sum([K.count_params(p) for p in self.trainable_weights])

class Recurrent(Layer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a valid layer!
    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.
    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    # Example
    ```python
        # as the first layer in a Sequential model
        model = Sequential()
        model.add(LSTM(32, input_shape=(10, 64)))
        # now model.output_shape == (None, 10, 32)
        # note: `None` is the batch dimension.
        # the following is identical:
        model = Sequential()
        model.add(LSTM(32, input_dim=64, input_length=10))
        # for subsequent layers, not need to specify the input size:
        model.add(LSTM(16))
    ```
    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled,
            else a symbolic loop will be used. When using TensorFlow, the network
            is always unrolled, so this argument does not do anything.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        consume_less: one of "cpu", "mem". If set to "cpu", the RNN will use
            an implementation that uses fewer, larger matrix products,
            thus running faster (at least on CPU) but consuming more memory.
            If set to "mem", the RNN will use more matrix products,
            but smaller ones, thus running slower (may actually be faster on GPU)
            while consuming less memory.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)
    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.
    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.
    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.
        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_shape=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    # Note on using dropout with TensorFlow
        When using the TensorFlow backend, specify a fixed batch size for your model
        following the notes on statefulness RNNs.
    '''
    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 unroll=False, consume_less='cpu',
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.consume_less = consume_less

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return []

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_sepc with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'consume_less': self.consume_less}
        if self.stateful:
            config['batch_input_shape'] = self.input_spec[0].shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''Gated Recurrent Unit - Cho et al. 2014.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(GRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W_z = self.init((input_dim, self.output_dim),
                             name='{}_W_z'.format(self.name))
        self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_z'.format(self.name))
        self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

        self.W_r = self.init((input_dim, self.output_dim),
                             name='{}_W_r'.format(self.name))
        self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_r'.format(self.name))
        self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

        self.W_h = self.init((input_dim, self.output_dim),
                             name='{}_W_h'.format(self.name))
        self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_h'.format(self.name))
        self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_z,
                                                        self.W_r,
                                                        self.W_h]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_z,
                                                        self.U_r,
                                                        self.U_h]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_z,
                                                        self.b_r,
                                                        self.b_h]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                  self.W_r, self.U_r, self.b_r,
                                  self.W_h, self.U_h, self.b_h]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'cpu':
            x_z = x[:, :self.output_dim]
            x_r = x[:, self.output_dim: 2 * self.output_dim]
            x_h = x[:, 2 * self.output_dim:]
        else:
            x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
            x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
            x_h = K.dot(x * B_W[2], self.W_h) + self.b_h

        z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.dropout(ones, self.dropout_W) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class AttentionDecoder(Recurrent):
    def __init__(self, units, output_dim,
                 activation='relu',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param _units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self._units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self._activation = activations.get(activation)
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._recurrent_initializer = initializers.get(recurrent_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._recurrent_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)

        self._kernel_constraint = constraints.get(kernel_constraint)
        self._recurrent_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self._name = name
        self._return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self._units,),
                                   name='V_a',
                                   initializer=self._kernel_initializer,
                                   regularizer=self._kernel_regularizer,
                                   constraint=self._kernel_constraint)
        self.W_a = self.add_weight(shape=(self._units, self._units),
                                   name='W_a',
                                   initializer=self._kernel_initializer,
                                   regularizer=self._kernel_regularizer,
                                   constraint=self._kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self._units),
                                   name='U_a',
                                   initializer=self._kernel_initializer,
                                   regularizer=self._kernel_regularizer,
                                   constraint=self._kernel_constraint)
        self.b_a = self.add_weight(shape=(self._units,),
                                   name='b_a',
                                   initializer=self._bias_initializer,
                                   regularizer=self._bias_regularizer,
                                   constraint=self._bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self._units),
                                   name='C_r',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.U_r = self.add_weight(shape=(self._units, self._units),
                                   name='U_r',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self._units),
                                   name='W_r',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.b_r = self.add_weight(shape=(self._units, ),
                                   name='b_r',
                                   initializer=self._bias_initializer,
                                   regularizer=self._bias_regularizer,
                                   constraint=self._bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self._units),
                                   name='C_z',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.U_z = self.add_weight(shape=(self._units, self._units),
                                   name='U_z',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self._units),
                                   name='W_z',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.b_z = self.add_weight(shape=(self._units, ),
                                   name='b_z',
                                   initializer=self._bias_initializer,
                                   regularizer=self._bias_regularizer,
                                   constraint=self._bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self._units),
                                   name='C_p',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.U_p = self.add_weight(shape=(self._units, self._units),
                                   name='U_p',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self._units),
                                   name='W_p',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.b_p = self.add_weight(shape=(self._units, ),
                                   name='b_p',
                                   initializer=self._bias_initializer,
                                   regularizer=self._bias_regularizer,
                                   constraint=self._bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.U_o = self.add_weight(shape=(self._units, self.output_dim),
                                   name='U_o',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self._bias_initializer,
                                   regularizer=self._bias_regularizer,
                                   constraint=self._bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self._units),
                                   name='W_s',
                                   initializer=self._recurrent_initializer,
                                   regularizer=self._recurrent_regularizer,
                                   constraint=self._recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self._units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            '_units': self._units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))