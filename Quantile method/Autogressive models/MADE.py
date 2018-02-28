# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:17:21 2018

@author: Futami
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import template as template_ops
from tensorflow.python.ops import variable_scope as variable_scope_lib
from tensorflow.python.ops.distributions import bijector as bijector_lib


MASK_INCLUSIVE = "inclusive"
MASK_EXCLUSIVE = "exclusive"


def _gen_slices(num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
  """Generate the slices for building an autoregressive mask."""
  # TODO(b/67594795): Better support of dynamic shape.
  slices = []
  col = 0
  d_in = n_in // num_blocks
  d_out = n_out // num_blocks
  row = d_out if mask_type == MASK_EXCLUSIVE else 0
  for _ in range(num_blocks):
    row_slice = slice(row, None)
    col_slice = slice(col, col + d_in)
    slices.append([row_slice, col_slice])
    col += d_in
    row += d_out
  return slices


def _gen_mask(num_blocks,
              n_in,
              n_out,
              mask_type=MASK_EXCLUSIVE,
              dtype=dtypes.float32):
  """Generate the mask for building an autoregressive dense layer."""
  # TODO(b/67594795): Better support of dynamic shape.
  mask = np.zeros([n_out, n_in], dtype=dtype.as_numpy_dtype())
  slices = _gen_slices(num_blocks, n_in, n_out, mask_type=mask_type)
  for [row_slice, col_slice] in slices:
    mask[row_slice, col_slice] = 1
  return mask


def masked_dense(inputs,
                 units,
                 num_blocks=None,
                 exclusive=False,
                 kernel_initializer=None,
                 reuse=None,
                 name=None,
                 *args,
                 **kwargs):
  """A autoregressively masked dense layer. Analogous to `tf.layers.dense`.
  See [1] for detailed explanation.
  [1]: "MADE: Masked Autoencoder for Distribution Estimation."
       Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. ICML. 2015.
       https://arxiv.org/abs/1502.03509
  Arguments:
    inputs: Tensor input.
    units: Python `int` scalar representing the dimensionality of the output
      space.
    num_blocks: Python `int` scalar representing the number of blocks for the
      MADE masks.
    exclusive: Python `bool` scalar representing whether to zero the diagonal of
      the mask, used for the first layer of a MADE.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the
      `tf.glorot_random_initializer`.
    reuse: Python `bool` scalar representing whether to reuse the weights of a
      previous layer by the same name.
    name: Python `str` used to describe ops managed by this function.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.
  Returns:
    Output tensor.
  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  """
  # TODO(b/67594795): Better support of dynamic shape.
  input_depth = inputs.shape.with_rank_at_least(1)[-1].value
  if input_depth is None:
    raise NotImplementedError(
        "Rightmost dimension must be known prior to graph execution.")

  mask = _gen_mask(num_blocks, input_depth, units,
                   MASK_EXCLUSIVE if exclusive else MASK_INCLUSIVE).T

  if kernel_initializer is None:
    kernel_initializer = init_ops.glorot_normal_initializer()

  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * kernel_initializer(shape, dtype, partition_info)

  with ops.name_scope(name, "masked_dense", [inputs, units, num_blocks]):
    layer = layers.Dense(
        units,
        kernel_initializer=masked_initializer,
        kernel_constraint=lambda x: mask * x,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse,
        *args,
        **kwargs)
    return layer.apply(inputs)


def masked_autoregressive_default_template(
    hidden_layers,
    shift_only=False,
    activation=nn_ops.relu,
    log_scale_min_clip=-5.,
    log_scale_max_clip=3.,
    log_scale_clip_gradient=False,
    name=None,
    *args,
    **kwargs):
  """Build the MADE Model [1].
  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the input and returns the `loc` ("mu" [1]) and
  `log_scale` ("alpha" [1]) from the MADE network.
  Warning: This function uses `masked_dense` to create randomly initialized
  `tf.Variables`. It is presumed that these will be fit, just as you would any
  other neural architecture which uses `tf.layers.dense`.
  #### About Hidden Layers:
  Each element of `hidden_layers` should be greater than the `input_depth`
  (i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
  neural network). This is necessary to ensure the autoregressivity property.
  #### About Clipping:
  This function also optionally clips the `log_scale` (but possibly not its
  gradient). This is useful because if `log_scale` is too small/large it might
  underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
  bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
  `bool` indicates whether the gradient should also be clipped. The default does
  not clip the gradient; this is useful because it still provides gradient
  information (for fitting) yet solves the numerical stability problem. I.e.,
  `log_scale_clip_gradient = False` means
  `grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
  `grad[clip(x)] exp(clip(x))`.
  [1]: "MADE: Masked Autoencoder for Distribution Estimation."
       Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. ICML. 2015.
       https://arxiv.org/abs/1502.03509
  Arguments:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed. Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The minimum value to clip by. Default: -5.
    log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The maximum value to clip by. Default: 3.
    log_scale_clip_gradient: Python `bool` indicating that the gradient of
      `tf.clip_by_value` should be preserved. Default: `False`.
    name: A name for ops managed by this function. Default:
      "masked_autoregressive_default_template".
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.
  Returns:
    shift: `Float`-like `Tensor` of shift terms (the "mu" in [2]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the "alpha" in [2]).
  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  """

  with ops.name_scope(name, "masked_autoregressive_default_template",
                      values=[log_scale_min_clip, log_scale_max_clip]):
    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = x.shape.with_rank_at_least(1)[-1].value
      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      input_shape = (np.int32(x.shape.as_list()) if x.shape.is_fully_defined()
                     else array_ops.shape(x))
      for i, units in enumerate(hidden_layers):
        x = masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,
            **kwargs)
      x = masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,
          **kwargs)
      if shift_only:
        x = array_ops.reshape(x, shape=input_shape)
        return x, None
      x = array_ops.reshape(
          x, shape=array_ops.concat([input_shape, [2]], axis=0))
      shift, log_scale = array_ops.unstack(x, num=2, axis=-1)
      which_clip = (math_ops.clip_by_value if log_scale_clip_gradient
                    else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale
    return template_ops.make_template(
        "masked_autoregressive_default_template", _fn)


def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
  """Clips input while leaving gradient unaltered."""
  with ops.name_scope(name, "clip_by_value_preserve_grad",
                      [x, clip_value_min, clip_value_max]):
    clip_x = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
  return x + array_ops.stop_gradient(clip_x - x)


class MaskedAutoregressiveFlow(bijector_lib.Bijector):
  def __init__(self,
               shift_and_log_scale_fn,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.
    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from both the forward domain (`x`) and the inverse domain
        (`y`). Calculation must respect the "autoregressive property" (see class
        docstring). Suggested default
        `masked_autoregressive_default_template(hidden_layers=...)`.
        Typically the function contains `tf.Variables` and is wrapped using
        `tf.make_template`. Returning `None` for either (both) `shift`,
        `log_scale` is equivalent to (but more efficient than) returning zero.
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    """
    name = name or "masked_autoregressive_flow"
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    super(MaskedAutoregressiveFlow, self).__init__(
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    event_size = array_ops.shape(x)[-1]
    y0 = array_ops.zeros_like(x, name="y0")
    # call the template once to ensure creation
    _ = self._shift_and_log_scale_fn(y0)
    def _loop_body(index, y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with variable_scope_lib.variable_scope(
          variable_scope_lib.get_variable_scope()) as vs:
        if vs.caching_device is None:
          vs.set_caching_device(lambda op: op.device)
        shift, log_scale = self._shift_and_log_scale_fn(y0)
      y = x
      if log_scale is not None:
        y *= math_ops.exp(log_scale)
      if shift is not None:
        y += shift
      return index + 1, y
  
    _, y = control_flow_ops.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=[0, y0])
    return y

  def _inverse(self, y):
    shift, log_scale = self._shift_and_log_scale_fn(y)
    x = y
    if shift is not None:
      x -= shift
    if log_scale is not None:
      x *= math_ops.exp(-log_scale)
    return x

  def _inverse_log_det_jacobian(self, y):
    _, log_scale = self._shift_and_log_scale_fn(y)
    if log_scale is None:
      return constant_op.constant(0., dtype=y.dtype, name="ildj")
    return -math_ops.reduce_sum(log_scale, axis=-1)

def masked_autoregressive_default_template2(
    hidden_layers,
    shift_only=False,
    activation=tf.nn.sigmoid,
    log_scale_min_clip=-5.,
    log_scale_max_clip=3.,
    log_scale_clip_gradient=False,
    name=None,
    *args,
    **kwargs):
  with ops.name_scope(name, "masked_autoregressive_default_template",
                      values=[log_scale_min_clip, log_scale_max_clip]):
    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = x.shape.with_rank_at_least(1)[-1].value
      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      input_shape = (np.int32(x.shape.as_list()) if x.shape.is_fully_defined()
                     else array_ops.shape(x))
      for i, units in enumerate(hidden_layers):
        x = masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,
            **kwargs)
      x = masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,
          **kwargs)
      if shift_only:
        x = array_ops.reshape(x, shape=input_shape)
        return x
      x = array_ops.reshape(
          x, shape=array_ops.concat([input_shape, [2]], axis=0))
      shift, log_scale = array_ops.unstack(x, num=2, axis=-1)
      which_clip = (math_ops.clip_by_value if log_scale_clip_gradient
                    else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale
    return template_ops.make_template(
        "masked_autoregressive_default_template", _fn)
    


def _mask_matrix_made(input_d,hidden_d):
    """A generator of masks for two-layered MADE model (see https://arxiv.org/pdf/1502.03509.pdf)"""
    mask_vector1 = np.random.randint(1, input_d, hidden_d)
    mask_matrix0 = np.fromfunction(lambda k, d: mask_vector1[k] >= d+1, (hidden_d, input_d), dtype=int).astype(np.int32).astype(np.float32)
    mask_matrix1 = np.fromfunction(lambda d, k: d+1 > mask_vector1[k], (input_d, hidden_d), dtype=int).astype(np.int32).astype(np.float32)
    return mask_matrix0, mask_matrix1

def MADE_layer(inputs,units,mask,name,activation):

  kernel_initializer = init_ops.glorot_normal_initializer()

  def masked_initializer(shape=(units,inputs), dtype=None, partition_info=None):
    return  mask.T*kernel_initializer(shape, dtype, partition_info)

  with ops.name_scope(name, "masked_dense", [inputs, units]):
   layer = layers.Dense(units,activation=activation,kernel_initializer=masked_initializer,kernel_constraint=lambda x: mask.T * x,name=name)#,_reuse=tf.AUTO_REUSE)
  return layer.apply(inputs)


def One_hidden_made(U,input_d,hidden_d,name):
    mask_matrix0, mask_matrix1=_mask_matrix_made(input_d,hidden_d)
    
    x = MADE_layer(inputs=U,units=hidden_d,mask=mask_matrix0,name=name+str('h'),activation=tf.nn.sigmoid)
    
    shift = MADE_layer(inputs=x,units=input_d,mask=mask_matrix1,name=name+str('shift'),activation=None)
    log_scale = MADE_layer(inputs=x,units=input_d,mask=mask_matrix1,name=name+str('log_scale'),activation=None)
#    x = MADE_layer(inputs=x,units=2*input_d,mask=mask_matrix1,name=name+str('shift'),activation=None)
    #x = array_ops.reshape(x,(input_d,2))
    #shift, log_scale = array_ops.unstack(x, num=2, axis=-1)
    return shift, log_scale


def MADE_NN(input_d,hidden_d):
    def _fn(x):
        shift, log_scale1=One_hidden_made(x,input_d,hidden_d,name='1')
        sigma1=tf.nn.sigmoid(log_scale1)
        x2=sigma1*x+(1.-sigma1)*shift
             
        shift2, log_scale2=One_hidden_made(x2,input_d,hidden_d,name='2')
        sigma2=tf.nn.sigmoid(log_scale2)
        x3=sigma2*x2+(1.-sigma2)*shift2        
        
        shift3, log_scale3=One_hidden_made(x3,input_d,hidden_d,name='3')
        sigma3=tf.nn.sigmoid(log_scale3)
        x4=sigma3*x3+(1.-sigma3)*shift3
        
        return x4,[sigma1,sigma2,sigma3]
    
    return _fn