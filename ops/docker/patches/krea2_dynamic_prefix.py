#!/usr/bin/env python3
"""Build-time fix for krea2 prompt encoding in the pinned ai-toolkit (99be3d96).

`encode_krea_prompt` builds its input as PREFIX + prompt (+ suffix appended
separately) and then removes the system prefix by slicing a FIXED number of
tokens, `PROMPT_TEMPLATE_ENCODE_START_IDX`. That constant is only correct for
one particular tokenizer serialization of the prefix. When the Qwen3-VL
snapshot in the training cache serializes the prefix to a different length,
the fixed slice is wrong in both directions:

  * prefix shorter than the constant -> the slice eats into the tokens that
    follow. For the empty prompt -- which the training template exercises on
    every step via `do_differential_guidance`'s unconditional pass -- there is
    little after the prefix to eat, and once the slice consumes everything the
    encoder returns zero tokens and the trainer dies in a reshape.
  * prefix longer than the constant -> stale prefix tokens leak into the
    conditioning ahead of the caption.

Either way the caption alignment silently depends on which tokenizer snapshot
happens to be on disk. The fix is to measure the prefix length under the
tokenizer actually loaded and slice exactly that many tokens.

Applied at image build; if the pinned source ever drifts so the anchor block
is missing, the build FAILS rather than shipping the unpatched encoder.
"""

from pathlib import Path

TARGET = Path("/app/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/text_encoder.py")

ANCHOR = """    # Drop the system-prefix tokens; what remains is prompt + suffix conditioning.
    hiddens = hiddens[:, prefix_idx:]
    return hiddens[0]"""

REPLACEMENT = """    # Drop the system-prefix tokens; what remains is prompt + suffix conditioning.
    # The default prefix_idx is a constant calibrated for one tokenizer
    # serialization of the prefix; measure the real length under the tokenizer
    # actually in use so the slice stays aligned whatever snapshot is cached.
    measured_prefix = int(
        tokenizer([PROMPT_TEMPLATE_ENCODE_PREFIX], return_tensors="pt")["input_ids"].shape[1]
    )
    hiddens = hiddens[:, measured_prefix:]
    if hiddens.shape[1] == 0:
        raise RuntimeError(
            "krea2 prompt encoding produced zero conditioning tokens "
            f"(measured prefix {measured_prefix} of {input_ids.shape[1]} total)"
        )
    return hiddens[0]"""


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if REPLACEMENT in source:
        print("krea2 prefix fix: already applied")
        return
    if ANCHOR not in source:
        raise SystemExit(
            f"krea2 prefix fix: anchor block not found in {TARGET}; "
            "the pinned source drifted -- refusing to build with the unpatched encoder"
        )
    TARGET.write_text(source.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    print("krea2 prefix fix: dynamic prefix slice installed")


if __name__ == "__main__":
    main()
