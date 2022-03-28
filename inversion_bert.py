# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import tensorflow as tf
import time
import tqdm
from absl import app
from absl import flags

import models.bert.modeling as modeling
import models.bert.tokenization as tokenization
from data.bookcorpus import load_author_data as load_bookcorpus_author
from data.common import MODEL_DIR
from data.wiki103 import load_wiki_cross_domain_data as load_cross_domain_data
from invert.bert_common import read_examples, convert_examples_to_features, \
  mean_pool
from invert.models import MultiLabelInversionModel, MultiSetInversionModel
from invert.utils import count_label_freq, tp_fp_fn_metrics_np, \
  tp_fp_fn_metrics
from train_feature_mapper import linear_mapping, mlp_mapping, gan_mapping
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, get_similarity_metric

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BERT_DIR = os.path.join(MODEL_DIR, 'bert', 'uncased_L-12_H-768_A-12')

flags.DEFINE_integer('high_layer_idx', -1, 'Output layer index')
flags.DEFINE_integer('low_layer_idx', -1, 'Optimize layer index')
flags.DEFINE_integer('seq_len', 16, 'Fixed recover sequence length')
flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")
flags.DEFINE_integer('train_size', 250, 'Number of authors data to use')
flags.DEFINE_integer('test_size', 125, 'Number of authors data to test')
flags.DEFINE_integer('max_iters', 1000, 'Max iterations for optimization')
flags.DEFINE_integer('print_every', 1, 'Print metrics every iteration')
flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_float('percent', 1.0, 'Percent of data to use for learning')
flags.DEFINE_integer("max_seq_length", 32, "The maximum total input sequence length after WordPiece tokenization. "
						"Sequences longer than this will be truncated, and sequences shorter "
						"than this will be padded.")
flags.DEFINE_integer('read_model_epoch', -1, 'If not -1, will read that model for the given epoch instead of re-training')
flags.DEFINE_string("init_checkpoint", os.path.join(BERT_DIR, 'bert_model.ckpt'), "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("bert_config_file", os.path.join(BERT_DIR, 'bert_config.json'), "The config json file corresponding to the pre-trained BERT model. "
					"This specifies the model architecture.")
flags.DEFINE_string("vocab_file", os.path.join(BERT_DIR, 'vocab.txt'), "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string('metric', 'l2', 'Metric to optimize')
flags.DEFINE_string('mapper', 'linear', 'Mapper to use')
flags.DEFINE_string('model', 'multiset', 'Model for learning based inversion')
flags.DEFINE_string('read_files', '', 'If supplied, will attempt to read train_x and test_x files, otherwise, generate.'
					'The string should be a base path to which \'train_x.bin\' and \'test_x.bin\' will be appended.')
flags.DEFINE_string('encrypted_tag', '', 'The prefix to be prepended to \'train_x.bin\' and \'test_x.bin\' file name.')
flags.DEFINE_boolean('encrypted_training', False, 'If true, will use encrypted embeddings for training as well')
flags.DEFINE_boolean('gen_npy', False, 'If true, will read {train,test}_y.txt and generate .npy')
flags.DEFINE_boolean('continue_training', False, 'If true, will simply continue training given model regardless of the --read_model_epoch')
flags.DEFINE_boolean('validation_only', False, 'If true, will only validate latest model')
flags.DEFINE_boolean('validation_all', False, 'If true, will validate all epoch model, but not train')
flags.DEFINE_boolean('learning', False, 'Learning based inversion or optimize based')
flags.DEFINE_boolean('cross_domain', False, 'Cross domain data for learning based inversion')
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased "
					"models and False for cased models.")
flags.DEFINE_bool("use_cls_token", False, "Whether to lower case the input text. Should be True for uncased "
					"models and False for cased models.")
flags.DEFINE_float('temp', 1e-2, 'Temperature for optimization')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('alpha', 0.0, 'Coefficient for regularization')
flags.DEFINE_float('C', 0.0, 'Label distribution aware margin')
flags.DEFINE_float('wd', 1e-4, 'Weight decay')

FLAGS = flags.FLAGS


def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings):
	"""Returns `model_fn` closure for TPUEstimator."""

	input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
	input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
	input_type_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
	model = modeling.BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, token_type_ids=input_type_ids, use_one_hot_embeddings=use_one_hot_embeddings)

	all_layer_outputs = [model.get_word_embedding_output()]
	all_layer_outputs += model.get_all_encoder_layers()

	if FLAGS.high_layer_idx == FLAGS.low_layer_idx:
		if FLAGS.use_cls_token:
			outputs = model.get_pooled_output()
		else:
			outputs = all_layer_outputs[FLAGS.low_layer_idx]
			outputs = mean_pool(outputs, input_mask)
	else:
		low_outputs = all_layer_outputs[FLAGS.low_layer_idx]
		low_outputs = mean_pool(low_outputs, input_mask)
		if FLAGS.use_cls_token:
			high_outputs = model.get_pooled_output()
		else:
			high_outputs = all_layer_outputs[FLAGS.high_layer_idx]
			high_outputs = mean_pool(high_outputs, input_mask)
		outputs = (low_outputs, high_outputs)

	tvars = tf.trainable_variables()
	(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

	tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
	return input_ids, input_mask, input_type_ids, outputs


def sents_to_examples(sents, tokenizer):
	examples = read_examples(sents, tokenization.convert_to_unicode)
	return convert_examples_to_features(examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)


def load_inversion_data():
	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
	tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	train_sents, _, test_sents, _, _, _ = load_bookcorpus_author(train_size=FLAGS.train_size, test_size=FLAGS.test_size, unlabeled_size=0, split_by_book=True, split_word=False, top_attr=800)
	# train_sents, test_sents = load_all_diagnosis(split_word=False)

	if FLAGS.cross_domain:
		train_sents = load_cross_domain_data(800000, split_word=False)

	input_ids, input_mask, input_type_ids, outputs = model_fn_builder(bert_config=bert_config, init_checkpoint=FLAGS.init_checkpoint, use_one_hot_embeddings=False)

	sess = tf.Session()
	sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

	learn_mapping = FLAGS.high_layer_idx != FLAGS.low_layer_idx

	def encode_example(features):
		n_data = len(features[0])
		embs_low, embs_high = [], []
		pbar = tqdm.tqdm(total=n_data)
		for b_idx in iterate_minibatches_indices(n_data, 128):
			emb = sess.run(outputs, feed_dict={input_ids: features[0][b_idx], input_mask: features[1][b_idx], input_type_ids: features[2][b_idx]})
			if learn_mapping:
				embs_low.append(emb[0])
				embs_high.append(emb[1])
				n_batch = len(emb[0])
			else:
				embs_low.append(emb)
				n_batch = len(emb)
			pbar.update(n_batch)
		pbar.close()

		if learn_mapping:
			return np.vstack(embs_low), np.vstack(embs_high)
		else:
			return np.vstack(embs_low)

	assert not (learn_mapping and FLAGS.read_files != "")

	train_features = sents_to_examples(train_sents, tokenizer)
	train_x = encode_example(train_features)

	test_features = sents_to_examples(test_sents, tokenizer)
	test_x = encode_example(test_features)
	tf.keras.backend.clear_session()

	if learn_mapping:
		log('Training high to low mapping...')
		if FLAGS.mapper == 'linear':
			mapping = linear_mapping(train_x[1], train_x[0])
		elif FLAGS.mapper == 'mlp':
			mapping = mlp_mapping(train_x[1], train_x[0], epochs=30, activation=tf.tanh)
		elif FLAGS.mapper == 'gan':
			mapping = gan_mapping(train_x[1], train_x[0], disc_iters=5, batch_size=64, gamma=1.0, epoch=100, activation=tf.tanh)
		else:
			raise ValueError(FLAGS.mapper)
		test_x = mapping(test_x[1])

	return train_x, train_features, test_x, test_features


def encode(embedding_output, input_ids, input_mask, token_type_ids, config):
	with tf.variable_scope("bert", reuse=True):
		with tf.variable_scope("embeddings", reuse=True):
			embedding_output = modeling.embedding_postprocessor(input_tensor=embedding_output, use_token_type=True, token_type_ids=token_type_ids, token_type_vocab_size=config.type_vocab_size, token_type_embedding_name="token_type_embeddings", use_position_embeddings=True, position_embedding_name="position_embeddings", initializer_range=config.initializer_range, max_position_embeddings=config.max_position_embeddings, dropout_prob=config.hidden_dropout_prob)

		with tf.variable_scope("encoder", reuse=True):
			attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, input_mask)

			all_encoder_layers, _ = modeling.transformer_model(input_tensor=embedding_output, attention_mask=attention_mask, hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads, intermediate_size=config.intermediate_size, intermediate_act_fn=modeling.get_activation(config.hidden_act), hidden_dropout_prob=config.hidden_dropout_prob, attention_probs_dropout_prob=config.attention_probs_dropout_prob, initializer_range=config.initializer_range, do_return_all_layers=True)

		all_encoder_layers = [embedding_output] + all_encoder_layers
		if FLAGS.use_cls_token:
			with tf.variable_scope("pooler", reuse=True):
				first_token_tensor = tf.squeeze(all_encoder_layers[-1][:, 0:1, :], 1)
				pooled_output = tf.layers.dense(first_token_tensor, config.hidden_size, activation=tf.tanh, kernel_initializer=modeling.create_initializer(config.initializer_range))
		else:
			sequence_output = all_encoder_layers[FLAGS.low_layer_idx]
			pooled_output = mean_pool(sequence_output, input_mask)
	return pooled_output


def filter_labels(labels, filters):
	new_labels = []
	for y in labels:
		new_y = np.setdiff1d(y, filters)
		new_labels.append(new_y)
	return np.asarray(new_labels)


def optimization_inversion():
	tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
	cls_id = tokenizer.vocab['[CLS]']
	sep_id = tokenizer.vocab['[SEP]']
	mask_id = tokenizer.vocab['[MASK]']

	_, _, x, y = load_inversion_data()
	filters = [cls_id, sep_id, mask_id]
	y = filter_labels(y[0], filters)

	batch_size = FLAGS.batch_size
	seq_len = FLAGS.seq_len
	max_iters = FLAGS.max_iters

	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
	input_ids = tf.ones((batch_size, seq_len + 2), tf.int32)
	input_mask = tf.ones_like(input_ids, tf.int32)
	input_type_ids = tf.zeros_like(input_ids, tf.int32)

	model = modeling.BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, token_type_ids=input_type_ids, use_one_hot_embeddings=False)

	bert_vars = tf.trainable_variables()

	(assignment_map, _) = modeling.get_assignment_map_from_checkpoint(bert_vars, FLAGS.init_checkpoint)
	tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
	word_emb = model.embedding_table

	batch_cls_ids = tf.ones((batch_size, 1), tf.int32) * cls_id
	batch_sep_ids = tf.ones((batch_size, 1), tf.int32) * sep_id
	cls_emb = tf.nn.embedding_lookup(word_emb, batch_cls_ids)
	sep_emb = tf.nn.embedding_lookup(word_emb, batch_sep_ids)

	prob_mask = np.zeros((bert_config.vocab_size, ), np.float32)
	prob_mask[filters] = -1e9
	prob_mask = tf.constant(prob_mask, dtype=np.float32)

	logit_inputs = tf.get_variable(name='inputs', shape=(batch_size, seq_len, bert_config.vocab_size), initializer=tf.random_uniform_initializer(-0.1, 0.1))
	t_vars = [logit_inputs]
	t_var_names = {logit_inputs.name}

	logit_inputs += prob_mask
	prob_inputs = tf.nn.softmax(logit_inputs / FLAGS.temp, axis=-1)
	emb_inputs = tf.matmul(prob_inputs, word_emb)

	emb_inputs = tf.concat([cls_emb, emb_inputs, sep_emb], axis=1)
	if FLAGS.low_layer_idx == 0:
		encoded = mean_pool(emb_inputs, input_mask)
	else:
		encoded = encode(emb_inputs, input_ids, input_mask, input_type_ids, bert_config)
	targets = tf.placeholder(tf.float32, shape=(batch_size, encoded.shape.as_list()[-1]))
	loss = get_similarity_metric(encoded, targets, FLAGS.metric, rtn_loss=True)
	loss = tf.reduce_sum(loss)

	if FLAGS.alpha > 0.:
		# encourage the words to be different
		diff = tf.expand_dims(prob_inputs, 2) - tf.expand_dims(prob_inputs, 1)
		reg = tf.reduce_sum(-tf.exp(tf.reduce_sum(diff**2, axis=-1)), [1, 2])
		loss += FLAGS.alpha * tf.reduce_sum(reg)

	optimizer = tf.train.AdamOptimizer(FLAGS.lr)

	start_vars = set(v.name for v in tf.global_variables() if v.name not in t_var_names)
	grads_and_vars = optimizer.compute_gradients(loss, t_vars)
	train_ops = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())

	end_vars = tf.global_variables()
	new_vars = [v for v in end_vars if v.name not in start_vars]

	preds = tf.argmax(prob_inputs, axis=-1)
	batch_init_ops = tf.variables_initializer(new_vars)

	total_it = len(x) // batch_size

	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

		def invert_one_batch(batch_targets):
			sess.run(batch_init_ops)
			feed_dict = {targets: batch_targets}
			prev = 1e6
			for i in range(max_iters):
				curr, _ = sess.run([loss, train_ops], feed_dict)
				# stop if no progress
				if (i + 1) % (max_iters // 10) == 0 and curr > prev:
					break
				prev = curr
			return sess.run([preds, loss], feed_dict)

		start_time = time.time()
		it = 0.0
		all_tp, all_fp, all_fn, all_err = 0.0, 0.0, 0.0, 0.0

		for batch_idx in iterate_minibatches_indices(len(x), batch_size, False, False):
			y_pred, err = invert_one_batch(x[batch_idx])
			tp, fp, fn = tp_fp_fn_metrics_np(y_pred, y[batch_idx])

			# for yp, yt in zip(y_pred, y[batch_idx]):
			#   print(','.join(set(tokenizer.convert_ids_to_tokens(yp))))
			#   print(','.join(set(tokenizer.convert_ids_to_tokens(yt))))

			it += 1.0
			all_err += err
			all_tp += tp
			all_fp += fp
			all_fn += fn

			all_pre = all_tp / (all_tp + all_fp + 1e-7)
			all_rec = all_tp / (all_tp + all_fn + 1e-7)
			all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)

			if it % FLAGS.print_every == 0:
				it_time = (time.time() - start_time) / it
				log("Iter {:.2f}%, err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%,"
					" {:.2f} sec/it".format(it / total_it * 100, all_err / it, all_pre * 100, all_rec * 100, all_f1 * 100, it_time))

		all_pre = all_tp / (all_tp + all_fp + 1e-7)
		all_rec = all_tp / (all_tp + all_fn + 1e-7)
		all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)
		log("Final err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%".format(all_err / it, all_pre * 100, all_rec * 100, all_f1 * 100))


def learning_inversion():
	assert FLAGS.low_layer_idx == FLAGS.high_layer_idx == -1

	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
	num_words = bert_config.vocab_size

	tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	cls_id = tokenizer.vocab['[CLS]']
	sep_id = tokenizer.vocab['[SEP]']
	mask_id = tokenizer.vocab['[MASK]']
	filters = [cls_id, sep_id, mask_id, 0]

	if FLAGS.read_files == "":
		log("Computing")

		train_x, train_y, test_x, test_y = load_inversion_data()
		train_y = filter_labels(train_y[0], filters)
		test_y = filter_labels(test_y[0], filters)

		log("Will put to files")
		train_x.tofile("train_x.bin")
		test_x.tofile("test_x.bin")
		np.save('train_y.npy', train_y, allow_pickle=True)
		np.save('test_y.npy', test_y, allow_pickle=True)
		log("Written")
	else:
		log(f"Will read files from {FLAGS.read_files} with tag {FLAGS.encrypted_tag}")
		train_x = np.fromfile(f"{FLAGS.read_files}/{FLAGS.encrypted_tag if FLAGS.encrypted_training else ''}train_x.bin", dtype=np.float32)
		train_x = train_x.reshape((200000, 768))
		test_x = np.fromfile(f"{FLAGS.read_files}/{FLAGS.encrypted_tag}test_x.bin", dtype=np.float32)
		test_x = test_x.reshape((100000, 768))

		if not FLAGS.gen_npy:
			train_y = np.load(f"{FLAGS.read_files}/train_y.npy", allow_pickle=True)
			test_y = np.load(f"{FLAGS.read_files}/test_y.npy", allow_pickle=True)
		else:

			def generate_tokenized_sents(tag):
				lines = []
				with open(f"{FLAGS.read_files}/{tag}_y.txt", "r") as sents_file:
					while True:
						line = sents_file.readline()
						if not line:
							break
						lines += [line]
				tokenized = sents_to_examples(lines, tokenizer)
				tokenized = filter_labels(tokenized[0], filters)
				np.save(f"{FLAGS.read_files}/{tag}_y.npy", tokenized, allow_pickle=True)
				return tokenized

			train_y = generate_tokenized_sents("train")
			test_y = generate_tokenized_sents("test")
			log("Generated .npy")

		log("Read")

	if FLAGS.percent < 1.0:
		train_x = train_x[:int(len(train_x) * FLAGS.percent)]
		test_x = test_x[:int(len(test_x) * FLAGS.percent)]
		train_y = train_y[:int(len(train_y) * FLAGS.percent)]
		test_y = test_y[:int(len(test_y) * FLAGS.percent)]

	label_freq = count_label_freq(train_y, num_words)
	log('Imbalace ratio: {}'.format(np.max(label_freq) / np.min(label_freq)))

	label_margin = tf.constant(np.reciprocal(label_freq**0.25), dtype=tf.float32)
	C = FLAGS.C

	log('Build attack model for {} words...'.format(num_words))

	encoder_dim = train_x.shape[1]
	inputs = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs")
	labels = tf.placeholder(tf.float32, (None, num_words), name="labels")
	training = tf.placeholder(tf.bool, name='training')

	if FLAGS.model == 'multiset':
		init_word_emb = None
		emb_dim = 512
		model = MultiSetInversionModel(emb_dim, num_words, FLAGS.seq_len, init_word_emb, C=C, label_margin=label_margin)
	elif FLAGS.model == 'multilabel':
		model = MultiLabelInversionModel(num_words, C=C, label_margin=label_margin)
	else:
		raise ValueError(FLAGS.model)

	preds, loss = model.forward(inputs, labels, training)
	true_pos, false_pos, false_neg, tp_indices, fp_indices, fn_indices = tp_fp_fn_metrics(labels, preds)
	eval_fetch = [loss, true_pos, false_pos, false_neg, tp_indices, fp_indices, fn_indices]

	t_vars = tf.trainable_variables()
	wd = FLAGS.wd
	post_ops = [tf.assign(v, v * (1 - wd)) for v in t_vars if 'kernel' in v.name]

	optimizer = tf.train.AdamOptimizer(FLAGS.lr)
	grads_and_vars = optimizer.compute_gradients(loss + tf.losses.get_regularization_loss(), t_vars)
	train_ops = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())

	with tf.control_dependencies([train_ops]):
		train_ops = tf.group(*post_ops)

	saver = tf.train.Saver()

	def get_model_name(epoch):
		return "{tmp}--inversion-model-v2--{dataset}--{training}--{tag}--{epoch}".format(
			tmp="temp" if FLAGS.percent < 1.0 else "perm",
			dataset="trec" if "trec" in FLAGS.read_files else "bookcorpus",
			training="enc" if FLAGS.encrypted_training else "normal",
			tag=FLAGS.encrypted_tag if FLAGS.encrypted_tag != "" and not FLAGS.validation_only else "plain",
			epoch=epoch,
		)

	def should_skip(epoch):
		if FLAGS.validation_all:
			return False
		if FLAGS.continue_training or FLAGS.validation_only:
			return \
				os.path.isdir(f"./model-files/{get_model_name(epoch )}") and \
				os.path.isdir(f"./model-files/{get_model_name(epoch + 1)}")
		return FLAGS.read_model_epoch != -1 and epoch < FLAGS.read_model_epoch

	def should_train(epoch):
		if FLAGS.validation_only or FLAGS.validation_all:
			return False
		if FLAGS.continue_training:
			return not os.path.isdir(f"./model-files/{get_model_name(epoch)}")
		return FLAGS.read_model_epoch == -1 or FLAGS.read_model_epoch < epoch

	log('Train attack model with {} data...'.format(len(train_x)))
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(FLAGS.epochs):
			model_name = get_model_name(epoch)

			if should_skip(epoch):
				log(f"Skipping epoch {epoch}")
				continue

			train_iterations = 0
			train_loss = 0

			if should_train(epoch):
				with tqdm.tqdm(total=len(train_y)) as pbar:
					pbar.set_description(f"Epoch {epoch}: training")
					for batch_idx in iterate_minibatches_indices(len(train_y), FLAGS.batch_size, True):
						one_hot_labels = np.zeros((len(batch_idx), num_words), dtype=np.float32)
						for i, idx in enumerate(batch_idx):
							one_hot_labels[i][train_y[idx]] = 1
						feed = {inputs: train_x[batch_idx], labels: one_hot_labels, training: True}
						err, _ = sess.run([loss, train_ops], feed_dict=feed)
						train_loss += err
						train_iterations += 1
						pbar.update(len(batch_idx))

				log(f"Will write model: {model_name}")
				os.makedirs(f"./model-files/{model_name}", exist_ok=True)
				saver.save(sess, f"./model-files/{model_name}/model")
			else:
				log(f"Will read model: {model_name}")
				saver = tf.train.import_meta_graph(f"./model-files/{model_name}/model.meta")
				saver.restore(sess, tf.train.latest_checkpoint(f"./model-files/{model_name}"))
				train_iterations = epoch + 1

			test_iterations = 0
			test_loss = 0
			test_tp, test_fp, test_fn = 0, 0, 0

			words = {
				"tp": [],
				"fp": [],
				"fn": [],
			}

			with tqdm.tqdm(total=len(test_y)) as pbar:
				pbar.set_description(f"Epoch {epoch}: validation")
				for batch_idx in iterate_minibatches_indices(len(test_y), batch_size=512, shuffle=False):
					one_hot_labels = np.zeros((len(batch_idx), num_words), dtype=np.float32)
					for i, idx in enumerate(batch_idx):
						one_hot_labels[i][test_y[idx]] = 1
					feed = {inputs: test_x[batch_idx], labels: one_hot_labels, training: False}

					fetch = sess.run(eval_fetch, feed_dict=feed)
					err, tp, fp, fn, tp_indices, fp_indices, fn_indices = fetch

					words["tp"] += [str(tokenizer.convert_ids_to_tokens([word])[0]) for sentence_id, word in tp_indices]
					words["fp"] += [str(tokenizer.convert_ids_to_tokens([word])[0]) for sentence_id, word in fp_indices]
					words["fn"] += [str(tokenizer.convert_ids_to_tokens([word])[0]) for sentence_id, word in fn_indices]

					test_iterations += 1
					test_loss += err
					test_tp += tp
					test_fp += fp
					test_fn += fn

					pbar.update(len(batch_idx))

			precision = test_tp / (test_tp + test_fp) * 100
			recall = test_tp / (test_tp + test_fn) * 100
			f1 = 2 * precision * recall / (precision + recall)

			log("Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, "
				"pre: {:.2f}%, rec: {:.2f}%, f1: {:.2f}%".format(epoch, train_loss / train_iterations, test_loss / test_iterations, precision, recall, f1))

			for metric in ["tp", "fp", "fn"]:
				with open(f"./word-dumps/{model_name}-against-{FLAGS.encrypted_tag}-{metric}.txt", "w", encoding="utf-8") as file:
					for word in words[metric]:
						file.write("%s\n" % word)
			log(f"Written to {model_name}-against-{FLAGS.encrypted_tag}-{{tp,fp,fn,tn}}.txt")


def main(_):
	if FLAGS.learning:
		assert FLAGS.low_layer_idx == FLAGS.high_layer_idx
		learning_inversion()
	else:
		optimization_inversion()


if __name__ == '__main__':
	app.run(main)
