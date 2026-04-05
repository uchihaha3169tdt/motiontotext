import re
import numpy as np

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def normalize_gloss_sequence(sequence):
    sequence = re.sub(r'\b(loc-|cl-|qu-|poss-|lh-)', '', sequence)
    replacements = {
        'S0NNE': 'SONNE', 'HABEN2': 'HABEN',
        '__EMOTION__': '', '__PU__': '', '__LEFTHAND__': '', '__EPENTHESIS__': '',
        '+': ' ',
    }
    for k, v in replacements.items():
        sequence = sequence.replace(k, v)
    sequence = re.sub(r'\bWIE AUSSEHEN\b', 'WIE-AUSSEHEN', sequence)
    sequence = re.sub(r'\bZEIGEN\b', 'ZEIGEN-BILDSCHIRM', sequence)
    sequence = re.sub(r'\b(\w+)\s+\1\b', r'\1', sequence)
    sequence = re.sub(r'-PLUSPLUS', '', sequence)
    return re.sub(r'\s+', ' ', sequence).strip()


def wer_list(references, hypotheses):
    total_err = total_del = total_ins = total_sub = total_ref = 0
    for r, h in zip(references, hypotheses):
        res = wer_single(r, h)
        total_err += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref += res["num_ref"]

    if total_ref == 0:
        return {"wer": 0.0, "del": 0.0, "ins": 0.0, "sub": 0.0}

    return {
        "wer": total_err / total_ref * 100,
        "del": total_del / total_ref * 100,
        "ins": total_ins / total_ref * 100,
        "sub": total_sub / total_ref * 100,
    }


def wer_single(r, h):
    r, h = r.strip().split(), h.strip().split()
    d = _edit_distance(r, h)
    alignment, alignment_out = _get_alignment(r, h, d)
    alignment_arr = np.array(alignment)
    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": int((alignment_arr == "C").sum()),
        "num_del": int((alignment_arr == "D").sum()),
        "num_ins": int((alignment_arr == "I").sum()),
        "num_sub": int((alignment_arr == "S").sum()),
        "num_err": int((alignment_arr != "C").sum()),
        "num_ref": len(r),
    }


def _edit_distance(r, h):
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint32)
    for i in range(len(r) + 1):
        d[i][0] = i * WER_COST_DEL
    for j in range(len(h) + 1):
        d[0][j] = j * WER_COST_INS
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j-1] + WER_COST_SUB,
                              d[i][j-1] + WER_COST_INS,
                              d[i-1][j] + WER_COST_DEL)
    return d


def _get_alignment(r, h, d):
    x, y = len(r), len(h)
    align_list, align_ref, align_hyp, alignment = [], "", "", ""
    while x > 0 or y > 0:
        if x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]:
            align_list.append("C")
            align_ref  = " " + r[x-1] + align_ref
            align_hyp  = " " + h[y-1] + align_hyp
            alignment  = " " * (len(r[x-1]) + 1) + alignment
            x -= 1; y -= 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] + WER_COST_SUB:
            ml = max(len(r[x-1]), len(h[y-1]))
            align_list.append("S")
            align_ref  = " " + r[x-1].ljust(ml) + align_ref
            align_hyp  = " " + h[y-1].ljust(ml) + align_hyp
            alignment  = " S" + " " * (ml - 1) + alignment
            x -= 1; y -= 1
        elif y >= 1 and d[x][y] == d[x][y-1] + WER_COST_INS:
            align_list.append("I")
            align_ref  = " " + "*" * len(h[y-1]) + align_ref
            align_hyp  = " " + h[y-1] + align_hyp
            alignment  = " I" + " " * (len(h[y-1]) - 1) + alignment
            y -= 1
        else:
            align_list.append("D")
            align_ref  = " " + r[x-1] + align_ref
            align_hyp  = " " + "*" * len(r[x-1]) + align_hyp
            alignment  = " D" + " " * (len(r[x-1]) - 1) + alignment
            x -= 1

    return (align_list[::-1],
            {"align_ref": align_ref.lstrip(), "align_hyp": align_hyp.lstrip(),
             "alignment": alignment.lstrip()})