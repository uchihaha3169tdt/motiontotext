import warnings
from itertools import groupby

import torch

try:
    import ctcdecode
except Exception:
    ctcdecode = None


class Decode:
    def __init__(self, gloss_dict, num_classes, search_mode="beam", blank_id=0):
        self.i2g = {v: k for k, v in gloss_dict.items()}
        self.blank_id = blank_id
        self.search_mode = search_mode
        self.ctc_decoder = None

        if search_mode != "max":
            if ctcdecode is None:
                warnings.warn("ctcdecode not installed — using greedy max decoding.", RuntimeWarning)
                self.search_mode = "max"
            else:
                vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
                self.ctc_decoder = ctcdecode.CTCBeamDecoder(
                    vocab, beam_width=10, blank_id=blank_id, num_processes=4
                )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self._max_decode(nn_output, vid_lgt)
        return self._beam_decode(nn_output, vid_lgt, probs)

    def _beam_decode(self, nn_output, vid_lgt, probs=False):
        if self.ctc_decoder is None:
            return self._max_decode(nn_output, vid_lgt)
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, _, _, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret = []
        for b in range(len(nn_output)):
            seq = beam_result[b][0][:out_seq_len[b][0]]
            if len(seq):
                seq = torch.stack([x[0] for x in groupby(seq)])
            ret.append([(self.i2g[int(g)], i) for i, g in enumerate(seq)])
        return ret

    def _max_decode(self, nn_output, vid_lgt):
        index_list = nn_output.argmax(dim=2)
        ret = []
        for b in range(index_list.size(0)):
            groups = [x[0] for x in groupby(index_list[b][:vid_lgt[b]])]
            filtered = [x for x in groups if x != self.blank_id]
            if filtered:
                filtered = [x[0] for x in groupby(torch.stack(filtered))]
            ret.append([(self.i2g[int(g)], i) for i, g in enumerate(filtered)])
        return ret