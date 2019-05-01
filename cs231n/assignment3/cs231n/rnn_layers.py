import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################

  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
  cache = (x, prev_h, Wx, Wh, next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (X, prev_h, Wx, Wh, next_h) = cache
  dhraw = (1 - next_h * next_h) * dnext_h  # Gradient according to tanh nonlinearity
  dx = np.dot(dhraw, Wx.T)
  dprev_h = np.dot(dhraw, Wh.T)
  dWx = np.dot(X.T, dhraw)
  dWh = np.dot(prev_h.T, dhraw)
  db = np.sum(dhraw, axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.

  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  h_prev = h0
  H = h0.shape[1]
  h = np.zeros(shape=(N, T, H))
  caches = []
  # print(T)
  for t in range(T):
    h_prev, cache = rnn_step_forward(x[:, t, :], h_prev, Wx, Wh, b)
    h[:, t, :] = h_prev
    caches.append(cache)
    # print('t : ', t)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, caches


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################

  # This is return value of rnn_step_backward()
  # return dx, dprev_h, dWx, dWh, db

  N, T, H = dh.shape
  D = cache[0][0].shape[1]
  # print(cache)
  # print((cache[0]))
  dx = np.zeros(shape=(N, T, D))
  dh0 = np.zeros(shape=(N, H))
  dWx = np.zeros(shape=(D, H))
  dWh = np.zeros(shape=(H, H))
  db = np.zeros(shape=(H, ))

  dprev_h = np.zeros(shape=(N, H))

  for t in reversed(range(T)):
    # Burada dprev_h geriye yollamam gereken türevi tutuyor.
    # Backprop into h !!!
    # RNN'in unrolled seklini düsünürsek dh[:, t, :] yukarıdan gelen,
    # dprev_h sağdan gelen türevi tutuyor. En son döngü bittiğinde
    # bunu h0'a atıyoruz.
    dnext_h = dh[:, t, :] + dprev_h
    dx[:, t, :], dprev_h, temp_dWx, temp_dWh, temp_db = rnn_step_backward(dnext_h, cache[t])
    dWx += temp_dWx
    dWh += temp_dWh
    db += temp_db

  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################

  out = W[x, :]
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  dW = np.zeros_like(W)
  # https://stackoverflow.com/questions/45473896/np-add-at-indexing-with-array
  np.add.at(dW, x, dout)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  H = prev_h.shape[1]
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  activation_vector = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

  # Şu anki cell state'e ne kadarını girdi olarak vereceğim.
  i_gate = sigmoid(activation_vector[:, 0:H])

  # Önceki cell state'den ne kadar unutacağım.
  f_gate = sigmoid(activation_vector[:, H:2 * H])

  # Cell state'in ne kadarını geçirmek istiyorum.
  o_gate = sigmoid(activation_vector[:, 2 * H:3 * H])

  # Bu aslında rnn'deki hidden state'in hesaplanmasıyla aynı.
  g_gate = np.tanh(activation_vector[:, 3 * H:4 * H])

  # Bir sonraki cell state'i hesapla.
  next_c = f_gate * prev_c + i_gate * g_gate

  # Bir sonraki hidden state'i hesapla
  next_h = o_gate * np.tanh(next_c)

  cache = (x, prev_h, prev_c, Wx, Wh, next_c, next_h, activation_vector, i_gate, f_gate, o_gate, g_gate)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return next_h, next_c, cache

  # Flowgraph (1):
  #
  # x                                          ai         i
  # --------+                               +-----(sigm)--- ... (2)
  # dx       \                             /  dai        di
  #          (*)---+                      /
  # Wx       /   da \                    /     af         f
  # --------+        \                  +---------(sigm)--- ... (2)
  # dWx               \             a  /      daf        df
  #                   (+)------(+)----+
  # prev_h            /    da  /   da  \       ao         o
  # --------+        /        /         +---------(sigm)--- ... (2)
  # dprev_h  \      /   b    /           \    dao        do
  #          (*)---+    ----+             \
  # Wh       /   da     db                 \   ag         g
  # --------+                               +-----(tanh)--- ... (2)
  # dWh                                       dag        dg

  # Flowgraph (2):
  #
  #         f
  # (1) ... --------+
  #         df       \
  #                  (*)---+
  # prev_c           /      \
  # ----------------+        \
  # dprev_c                   \    next_c         tanh_c      next_h
  #                           (+)---------(tanh)---------(*)---------
  #         i                 /   dnext_c        dtanh_c /   dnext_h
  # (1) ... --------+        /                          /
  #         di       \      /                     o    /
  #                  (*)---+              (1) ... ----+
  #         g        /                            do
  # (1) ... --------+
  #         dg

  # Derivative of tanh(x): 1 - tanh(x)^2
  # Derivative of sigm(x): sigm(x) * (1 - sigm(x))


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, next_c, next_h, activation_vector, i_gate, f_gate, o_gate, g_gate = cache
  # x, prev_h, prev_c, Wx, Wh, next_c, next_h = cache
  N, H = dnext_h.shape
  dactivation_vector = np.zeros((N, 4 * H))
  dnext_c += dnext_h * o_gate * (1 - np.tanh(next_c) ** 2)
  dprev_c = dnext_c * f_gate

  di = dnext_c * g_gate * (1 - i_gate) * i_gate
  df = dnext_c * prev_c * (1 - f_gate) * f_gate
  do = dnext_h * np.tanh(next_c) * (1 - o_gate) * o_gate
  dg = dnext_c * i_gate * (1 - g_gate ** 2)

  da = np.hstack([di, df, do, dg])

  # print("x shape:", x.shape)
  # print("wx shape:", Wx.shape)
  # print("Wh shape:", Wh.shape)
  # # print("b shape:", b.shape)
  # print("prev_h shape:", prev_h.shape)

  dx = np.dot(da, Wx.T)
  # print("dx shape:", dx.shape)

  dWx = np.dot(x.T, da)
  # print("dwx shape:", dWx.shape)

  dWh = np.dot(prev_h.T, da)
  # print("dwh shape:", dWh.shape)

  dprev_h = np.dot(da, Wh.T)
  db = np.sum(da, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, caches = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  hiddens = np.zeros((N, T, h0.shape[1]))

  caches = []
  prev_h = h0
  prev_c = np.zeros_like(h0)
  for t in range(T):
    prev_h, prev_c, cache = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
    hiddens[:, t, :] = prev_h
    caches.append(cache)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return hiddens, caches


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None

  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N, T, H = dh.shape
  D = cache[-1][0].shape[-1]

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros((4 * H, ))

  dprev_h = np.zeros((N, H))
  dprev_c = np.zeros((N, H))

  for t in reversed(range(T)):
    temp_dx, dprev_h, dprev_c, temp_dWx, temp_dWh, temp_db = lstm_step_backward(dh[:, t, :] + dprev_h, dprev_c, cache[t])

    dx[:, t, :] += temp_dx
    dWx += temp_dWx
    dWh += temp_dWh
    db += temp_db

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)

  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose:
    print('dx_flat: ', dx_flat.shape)

  dx = dx_flat.reshape(N, T, V)

  return loss, dx
