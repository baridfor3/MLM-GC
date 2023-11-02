# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.MLM import MLM_base
from UNIVERSAL.basic_layer import embedding_layer
from UNIVERSAL.utils import padding_util
from UNIVERSAL.basic_metric import seq2seq_metric, mean_metric
import tensorflow as tf
import sys

# from UNIVERSAL.block import TransformerBlock
# import CLPM

class Glo_MLM(MLM_base.MLM_base):
    def __init__(self, param,  **kwargs):
        super().__init__(param, **kwargs)
        self.linear_mode = param["linear_mode"]
        # The constant scalling : 1/sqrt(d)
        # Note that, it is non-tranable.
        self.linear_scale = self.add_weight(
            shape=[1],
            dtype="float32",
            name="linear_scale",
            initializer= tf.keras.initializers.Constant(self.param["num_units"] ** -0.5),
            trainable=False
        )
        # Compute Linear loss
        self.seq2seq_loss_linear = seq2seq_metric.MeanSquaredError_layer(name="linear_loss")
        # MLM loss
        self.seq2seq_loss_crossentropy = seq2seq_metric.CrossEntropy_layer(
            param["vocabulary_size"], param["label_smoothing"], name="cross_entropy"
        )
        # b_t
        self.linear_central_bias = embedding_layer.EmbeddingSharedWeights(param["vocabulary_size"],
            1,
            name="linear_central_bias",
            pad_id=param["PAD_ID"],
            affine=False,
            scale_we=False,initializer=tf.keras.initializers.Constant(0.0))
        # b_tn
        self.linear_context_bias = embedding_layer.EmbeddingSharedWeights(param["vocabulary_size"],
            1,
            name="linear_context_bias",
            pad_id=param["PAD_ID"],
            affine=False,
            scale_we=False,initializer=tf.keras.initializers.Constant(0.0) )
        self.total_loss = mean_metric.Mean_MetricLayer("loss")

        def __normLogToSUMone_fn(t):
            t = t * (
                1
                - padding_util.get_decoder_self_attention_bias(
                    tf.shape(t)[1], lower=self.param["window"], upper=self.param["window"]
                )
            )
            re = tf.math.divide_no_nan(t, tf.linalg.norm(t, ord=1, keepdims=True, axis=-1))
            return re


        self.normLogToSUMone = tf.keras.layers.Lambda(lambda t: __normLogToSUMone_fn(t))

    def pre_training(self, data):
        ((input_src, output_tgt, span,tgt_label,lang_ids),) = data
        src_lang_ids =  tgt_lang_ids = lang_ids
        span = tf.reshape(span, [-1, 256, 256])
        metric = tf.where(tf.equal(input_src, self.param["MASK_ID"]), tgt_label, input_src)
        context_only = tf.cast(tf.not_equal(input_src, self.param["MASK_ID"]),tf.int32)*input_src * tf.cast(tf.not_equal(input_src, self.param["EOS_ID"]),tf.int32)
        _ = self.seq2seq_training(
            self.call,
            input_src,
            output_tgt,
            sos=self.param["EOS_ID"],
            src_id=src_lang_ids,
            tgt_id=tgt_lang_ids,
            tgt_label=tgt_label,
            tgt_metric=metric,
            context_only = context_only,
            span=span,
        )
    ###############################MLM-GC main #################
    def seq2seq_training(self, call_fn, x, y, sos=None, training=True, **kwargs):
        # X_wtwn counts
        span = kwargs["span"]
        if self.param["pre_log"]:
            y_linear = span
        else:
            # log counts
            y_linear = tf.math.log(span + 1)
        with tf.GradientTape() as model_tape:
            if sos is not None:
                sos_y = tf.pad(y, [[0, 0], [1, 0]], constant_values=sos)[:, :-1]
            else:
                sos_y = y
            x_logits, hidden_state = call_fn((x, sos_y), training=training, **kwargs)

            ####### Note that y_label is also the central token.###############
            ######### y_label = [0,0,0,t1,0,0,t2,0]. i.e., masked tokens
            if "tgt_label" in kwargs:
                y_label = kwargs["tgt_label"]
            else:
                y_label = y

            _,l = tf.unstack( tf.shape(y_label))
            window_masking = (
                1
                - padding_util.get_decoder_self_attention_bias(
                    l, lower=self.param["window"], upper=self.param["window"]
                )
            )
            central_masking = padding_util.get_decoder_self_attention_bias(l,lower=0, upper=0)
            t_masking = 1-tf.reshape(padding_util.get_padding(y_label),[-1,l,1])
            linear_masking = window_masking*central_masking*t_masking
            c_masking = tf.expand_dims(tf.not_equal(kwargs["context_only"],0),-1)
            c_masking = tf.cast(c_masking,tf.float32)

            # factorize H
            # import pdb;pdb.set_trace()
            h = hidden_state * t_masking
            # o = hidden_state * (1-t_masking)
            # factorize O
            # o = self.embedding_softmax_layer(kwargs["context_only"])
            o = hidden_state*c_masking

            # fisrt term of regression
            x_linear = tf.matmul(
                    h,
                    o,
                    transpose_b=True,
                )*self.param["num_units"] ** -0.5
            # + t_bias + n_bias
            x_linear *=  linear_masking
            kwargs["src_metric"] = x_logits
            weighting_factor = tf.minimum(1.0, tf.pow(tf.math.divide_no_nan(span, 100), 3 / 4))

            y_linear = y_linear * linear_masking
            return self.seq2seq_update(
                [tf.expand_dims(y_linear, -1), tf.expand_dims(x_linear, -1), weighting_factor], # GC
                [y_label, x_logits], ## MLM
                model_tape,
                **kwargs
            )

    def seq2seq_update(self, linear, cross, model_tape, **kwargs):
        # y_linear, x_linear = linear
        y_label, x_logits = cross
        loss_linear = self.seq2seq_loss_linear(linear, auto_loss=False, penalty=1/(2*self.param["window"]))
        loss_cross = self.seq2seq_loss_crossentropy(cross, auto_loss=False)
        if self.linear_mode:
            loss = loss_linear
        else:
            loss = loss_linear + loss_cross
        model_gradients = model_tape.gradient(loss, self.trainable_variables)

        if self.param["clip_norm"] > 0:
            model_gradients, grad_norm = tf.clip_by_global_norm(
                model_gradients, self.param["clip_norm"]
            )
        else:
            grad_norm = tf.linalg.global_norm(model_gradients)
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        self.grad_norm_ratio(grad_norm)
        self.total_loss(loss)
        self.perplexity(tf.math.exp(tf.cast(loss_cross, tf.float32)))
        if "tgt_label" in kwargs:
            y = kwargs["tgt_label"]
        if "tgt_metric" in kwargs:
            y_metric = kwargs["tgt_metric"]
        else:
            y_metric = y_label
        if "src_metric" in kwargs:
            src_metric = kwargs["src_metric"]
        else:
            src_metric = x_logits
        self.seq2seq_metric([y_metric, src_metric])
        batch_size = tf.shape(x_logits)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x_logits)[1])), tf.float32))
        return
