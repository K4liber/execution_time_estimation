from pypapi import events, papi_high as high
import numpy


def test_flops():
    high.start_counters([events.PAPI_DP_OPS, events.PAPI_TOT_CYC])
    a = 10.056 * 11.001
    x = high.stop_counters()
    print(x)  # Or record results in another way


if __name__ == '__main__':
    test_flops()