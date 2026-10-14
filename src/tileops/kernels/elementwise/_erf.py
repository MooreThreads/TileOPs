"""erf(x) at the accuracy the result's storage dtype can hold."""

import tilelang.language as T

__all__ = ["erf"]

# Abramowitz and Stegun 7.1.26, the float32 constants the CPU reference
# evaluates. The Horner step is an fma: a separate multiply and add rounds
# the small-|x| tail one representable step off the reference.
_P = 0.32759109139442444
_A5 = 1.0614054203033447
_A4 = -1.453152060508728
_A3 = 1.421413779258728
_A2 = -0.2844967246055603
_A1 = 0.2548295855522156


def erf(x, out_dtype):
    """Element-wise erf(x), evaluated and returned in float32.

    float32 calls the device erf. float16 and bfloat16 follow the reference
    polynomial: t = 1 / (1 + p|x|), then
    sign(x) * (1 - poly(t) * t * exp(-x^2)), with every multiply-add fused.

    NaN is not preserved. The sign test is a comparison, which is false for
    NaN, and the clamp then returns its non-NaN bound, so NaN reads back as
    -1. Restoring NaN costs another ``T.if_then_else``.

    Args:
        x: The argument. Any float dtype; promoted to float32 before evaluation.
        out_dtype: The dtype the caller stores the result in. It selects the
            evaluation, not the return dtype.

    Returns:
        erf(x) in float32.
    """
    wide = T.cast(x, "float32")
    if out_dtype == "float32":
        return T.erf(wide)
    zero = T.cast(0.0, "float32")
    one = T.cast(1.0, "float32")
    ax = T.abs(wide)
    den = T.ieee_fmaf(T.cast(_P, "float32"), ax, one)
    t = one / den
    z = T.cast(_A5, "float32")
    z = T.ieee_fmaf(z, t, T.cast(_A4, "float32"))
    z = T.ieee_fmaf(z, t, T.cast(_A3, "float32"))
    z = T.ieee_fmaf(z, t, T.cast(_A2, "float32"))
    z = T.ieee_fmaf(z, t, T.cast(_A1, "float32"))
    ex = T.exp(-(wide * wide))
    y = T.ieee_fmaf(z, t * (-ex), one)
    # wide < 0 is false for NaN. The clamp lowers to fmin/fmax, which return
    # their non-NaN operand, so a NaN argument comes back as -1.
    y = T.if_then_else(wide < zero, -y, y)
    return T.min(T.max(y, -one), one)
