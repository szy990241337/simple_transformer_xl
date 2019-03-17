import tensorflow as tf


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)  # 得到矩阵
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :] # 增加一个维度 # 类似于[[1. 1. 2.],[1. 2. 1.]]--》[[[1. 1. 2.]],[[1. 2. 1.]]]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
    output = inp
    with tf.variable_scope(scope):
        output = tf.layers.dense(inp, d_inner, activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output, d_model,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_2')
        output = tf.layers.dropout(output, dropout, training=is_training,
                                   name='drop_2')
        output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mem, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        if mem is not None and mem.shape.ndims > 1:
            cat = tf.concat([mem, w],0)
        else:
            cat = w
        w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                                  kernel_initializer=kernel_initializer, name='qkv') # (inputs,unit)
        r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='r') # ??

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias # r_w_bias的维度 [n_head, d_head]
        rr_head_q = w_head_q + r_r_bias # r_w_bias的维度 [n_head, d_head]

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]  ### [None, None]是什么意思 增加了两个维度
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=kernel_initializer, name='o')  #
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, initializer, proj_initializer, scope='adaptive_embed'):
    # d_proj 就是 d_model,
    emb_scale = d_proj ** 0.5
    with tf.variable_scope(scope):
        lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                       initializer=initializer)
        y = tf.nn.embedding_lookup(lookup_table, x)
        if d_proj != d_embed:  # 256 != 128
            proj_W = tf.get_variable('proj_W', [d_embed, d_proj], # 如果 d_embed与d_model不一样，就无法skip connect了，所以，一旦不一样，强行通过矩阵乘法把词向量的维度弄成d_embed
                                     initializer=proj_initializer)
            y = tf.einsum('ibe,ed->ibd', y, proj_W)
        else:
            proj_W = None
        ret_params = [lookup_table, proj_W]

    y *= emb_scale
    return y, ret_params


def mask_adaptive_logsoftmax(hidden, target, n_token, params, scope='adaptive_softmax',
                             return_mean=True):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        softmax_b = tf.get_variable('bias', [n_token],
                                    initializer=tf.zeros_initializer())
        output = _logit(hidden, params_W, softmax_b, params_projs)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                             logits=output)
    if return_mean:
        nll = tf.reduce_mean(nll)
    return nll


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

    return tf.stop_gradient(new_mem)


def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt, initializer, is_training, proj_initializer=None,
                mem_len=None, same_length=False, clamp_len=-1,
                untie_r=False, scope='transformer'):
    """
    :param dec_inp: input_data(序列长度，batch_size)
    :param target:  labes(序列长度，batch_size)
    :param mems: list类型，保存了每一层的mermory
    :param n_token: 词汇数量
    :param n_layer: transformer是多层的，指定transformer的层数
    :param d_model: 为了更好的skip connect（残差连接），将每个feed-ward网络的输出维度指定为d_model
    :param d_embed: 词向量的维度
    :param n_head:  指定多头注意力的头数，类似的理解为卷积核的个数，一头注意力相当于一个卷积核（一头注意力，翻译地真尴尬）
    :param d_head:  每头注意力的维度（可以理解为一个卷积核通道的个数）
    :param d_inner:
    :param dropout:
    :param dropatt:
    :param initializer: 非None
    :param is_training:
    :param proj_initializer: 非None
    :param mem_len: 记忆长度
    :param cutoffs:
    :param div_val:
    :param tie_projs: a list of python bools. Whether to tie the projections.
    :param same_length:
    :param clamp_len:
    :param use_tpu: False
    :param input_perms: 默认none
    :param target_perms: 默认none
    :param head_target: 默认none
    :param untie_r:  # 如果untie_r为True，每层的attention都有自己的r_w_bias，r_r_bias，如果为False，共用一套r_w_bias，r_r_bias
    :param proj_same_dim: 传入时为True
    :param scope:
    :return:
    """
    new_mems = []

    with tf.variable_scope(scope):
        if untie_r:  # 如果untie_r为True，每层的attention都有自己的r_w_bias，r_r_bias，如果为False，共用一套r_w_bias，r_r_bias
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
        else: # select
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        if proj_initializer is None:
            proj_initializer = initializer

        lookup_fn = mask_adaptive_embedding_lookup
        embeddings, shared_params = lookup_fn(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            initializer=initializer,
            proj_initializer=proj_initializer,
            )  # shared_params：list:[矩阵一，矩阵二] 矩阵一就是词嵌入矩阵，矩阵二的shape是[d_embed, d_model]

        attn_mask = _create_mask(qlen, mlen, same_length)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:  # pass
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model)) # 广播机制，
        pos_emb = positional_embedding(pos_seq, inv_freq) #  (pos_seq, 1, inv_freq)

        output = tf.layers.dropout(embeddings, dropout, training=is_training)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mem=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)

                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)

        output = tf.layers.dropout(output, dropout, training=is_training)

        logsoftmax_fn = mask_adaptive_logsoftmax
        loss = logsoftmax_fn(hidden=output, target=target, n_token=n_token, params=shared_params)
        return loss, new_mems
