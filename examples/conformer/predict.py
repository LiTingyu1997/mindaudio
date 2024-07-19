"""ASR inference process.

python predict.py --config_path <CONFIG_FILE>
"""

import os

import mindspore
import numpy as np
from asr_model import creadte_asr_model
from dataset import create_asr_predict_dataset, load_language_dict
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindaudio.metric.wer import wer
from mindaudio.models.decoders.decoder_factory import (
    Attention,
    AttentionRescoring,
    CTCGreedySearch,
    CTCPrefixBeamSearch,
    PredictNet,
)
from mindaudio.utils.config import get_config
from mindaudio.utils.log import get_logger
from mindaudio.utils.parallel_info import get_device_id
from mindaudio.utils.recognize import (
    attention_rescoring,
    ctc_greedy_search,
    ctc_prefix_beam_search,
    recognize,
)

logger = get_logger()
config = get_config("conformer")


def main():
    """main function for asr_predict."""
    exp_dir = config.exp_name
    decode_mode = config.decode_mode
    model_dir = os.path.join(exp_dir, "model")
    decode_ckpt = os.path.join(model_dir, config.decode_ckpt)
    decode_dir = os.path.join(exp_dir, "test_" + decode_mode)
    os.makedirs(decode_dir, exist_ok=True)
    result_file = open(os.path.join(decode_dir, "result.txt"), "w")

    mindspore.set_context(
        mode=0,
        device_target="Ascend",
        device_id=get_device_id(),
    )
    mindspore.set_context(jit_config={"jit_level": "O2"})

    # load test data
    test_dataset = create_asr_predict_dataset(
        config.test_data, config.dict, config.dataset_conf, config.collate_conf
    )
    # load dict
    sos, eos, vocab_size, char_dict = load_language_dict(config.dict)

    collate_conf = config.collate_conf
    input_dim = collate_conf.feature_extraction_conf.mel_bins

    # define network
    network = creadte_asr_model(config, input_dim, vocab_size)
    param_dict = load_checkpoint(decode_ckpt)
    load_param_into_net(network, param_dict)
    logger.info("Successfully loading the asr model: %s", decode_ckpt)
    network.set_train(False)

    if config.decode_mode == "ctc_greedy_search":
        model = Model(CTCGreedySearch(network))
    elif config.decode_mode == "attention" and config.full_graph:
        model = Model(Attention(network, config.beam_size, eos))
    elif config.decode_mode == "attention" and not config.full_graph:
        model = Model(PredictNet(network, config.beam_size, eos))
    elif config.decode_mode == "ctc_prefix_beam_search":
        model = Model(CTCPrefixBeamSearch(network, config.beam_size))
    elif config.decode_mode == "attention_rescoring":
        # do ctc prefix beamsearch first and then do rescoring by decoder
        model_ctc = Model(CTCPrefixBeamSearch(network, config.beam_size))
        model_rescore = Model(AttentionRescoring(network, config.beam_size))
    tot_sample = test_dataset.get_dataset_size()
    logger.info("Total predict samples size: %d", tot_sample)
    count = 0
    sum = 0
    for data in test_dataset:
        uttid, xs_pad, xs_masks, tokens, xs_lengths = data
        logger.info("Using decoding strategy: %s", config.decode_mode)
        if config.decode_mode == "attention":
            start_token = np.array([sos], np.int32)
            scores = np.array(
                [0.0] + [-float("inf")] * (config.beam_size - 1), np.float32
            )
            end_flag = np.array([0.0] * config.beam_size, np.float32)
            base_index = np.array(np.arange(xs_pad.shape[0]), np.int32).reshape(-1, 1)

            hyps, _ = recognize(
                model,
                xs_pad,
                xs_masks,
                start_token,
                base_index,
                scores,
                end_flag,
                config.beam_size,
                eos,
                xs_lengths,
                config.full_graph,
            )
            hyps = [hyp.tolist() for hyp in hyps]
        elif config.decode_mode == "ctc_greedy_search":
            hyps, _ = ctc_greedy_search(model, xs_pad, xs_masks, xs_lengths)
        # ctc_prefix_beam_search and attention_rescoring restrict the batch_size = 1
        # and return one result in List[int]. Here change it to List[List[int]] for
        # compatible with other batch decoding mode
        elif config.decode_mode == "ctc_prefix_beam_search":
            assert xs_pad.shape[0] == 1
            hyps, _, _ = ctc_prefix_beam_search(
                model, xs_pad, xs_masks, config.beam_size, xs_lengths
            )
            hyps = [hyps[0][0]]
        elif config.decode_mode == "attention_rescoring":
            assert xs_pad.shape[0] == 1
            hyps, _ = attention_rescoring(
                model_ctc,
                model_rescore,
                xs_pad,
                xs_masks,
                xs_lengths,
                sos,
                eos,
                config.beam_size,
                config.ctc_weight,
            )
            hyps = [hyps]
        else:
            raise NotImplementedError

        # batch size equals to 1
        content = ""
        content_list = []
        ground_truth = ""
        ground_truth_list = []
        count += 1
        for w in hyps[0]:
            w += 2
            if w == eos:
                break
            if w > len(char_dict):
                continue
            character = char_dict[w]
            content += character
            content_list.append(character)
        tokens_np = tokens.asnumpy()
        for w in tokens_np:
            w += 2
            character = char_dict[w]
            ground_truth += character
            ground_truth_list.append(character)
        logger.info("Labs (%d/%d): %s %s", count, tot_sample, uttid, ground_truth)
        logger.info("Hyps (%d/%d): %s %s", count, tot_sample, uttid, content)
        if not content_list:
            raise ValueError("The Hypothesis utterance should not be empty")
        cer = wer(ground_truth_list, content_list)
        logger.info("cer : %.3f", cer)
        result_file.write("{} {}\n".format(uttid, content))
        result_file.flush()
        sum += cer
    logger.info("cer_average : %f", sum / count)
    result_file.close()


if __name__ == "__main__":
    main()
