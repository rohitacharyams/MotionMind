# Auto-generated orientation blacklist (regenerated 2026-06-26 by the
# authoritative VRM-FK orientation verifier, _verify_orientation.py).
# These 67 clips FAIL the upright test on the CURRENT corrected
# clip JSON: head drops below hips (inverted), the body folds /
# collapses, or it lies horizontal at some frame. They must never
# be played. Filtered out of every clip-selection path in tools.py
# AND excluded from the served catalogue in motion_index.list_motions.
#
# Verdict breakdown:
#   FAIL_COLLAPSE    48
#   FAIL_INVERTED    19
INVERTED_BLACKLIST = {
    "cmu_01_01_08",                      # FAIL_COLLAPSE cos_min=0.102 bodyv_min=0.033
    "cmu_01_01_10",                      # FAIL_INVERTED cos_min=-0.357 bodyv_min=-0.126
    "cmu_02_02_06",                      # FAIL_COLLAPSE cos_min=0.218 bodyv_min=0.072
    "cmu_02_02_10",                      # FAIL_COLLAPSE cos_min=-0.013 bodyv_min=-0.004
    "cmu_05_05_02",                      # FAIL_COLLAPSE cos_min=-0.074 bodyv_min=-0.026
    "cmu_05_05_04",                      # FAIL_COLLAPSE cos_min=0.299 bodyv_min=0.102
    "cmu_05_05_05",                      # FAIL_COLLAPSE cos_min=0.19 bodyv_min=0.067
    "cmu_05_05_06",                      # FAIL_INVERTED cos_min=-0.817 bodyv_min=-0.277
    "cmu_05_05_08",                      # FAIL_INVERTED cos_min=-0.114 bodyv_min=-0.04
    "cmu_05_05_11",                      # FAIL_COLLAPSE cos_min=0.104 bodyv_min=0.036
    "cmu_05_05_13",                      # FAIL_COLLAPSE cos_min=0.374 bodyv_min=0.129
    "cmu_05_05_18",                      # FAIL_COLLAPSE cos_min=0.28 bodyv_min=0.098
    "cmu_05_05_19",                      # FAIL_COLLAPSE cos_min=0.406 bodyv_min=0.142
    "cmu_06_06_11",                      # FAIL_INVERTED cos_min=-0.188 bodyv_min=-0.066
    "cmu_06_06_12",                      # FAIL_COLLAPSE cos_min=-0.048 bodyv_min=-0.017
    "cmu_105_105_01",                    # FAIL_INVERTED cos_min=-0.125 bodyv_min=-0.044
    "cmu_105_105_03",                    # FAIL_COLLAPSE cos_min=-0.006 bodyv_min=-0.002
    "cmu_105_105_04",                    # FAIL_COLLAPSE cos_min=-0.067 bodyv_min=-0.023
    "cmu_105_105_07",                    # FAIL_COLLAPSE cos_min=0.184 bodyv_min=0.065
    "cmu_105_105_16",                    # FAIL_COLLAPSE cos_min=0.349 bodyv_min=0.122
    "cmu_105_105_23",                    # FAIL_COLLAPSE cos_min=0.33 bodyv_min=0.116
    "cmu_105_105_40",                    # FAIL_COLLAPSE cos_min=0.275 bodyv_min=0.094
    "cmu_105_105_41",                    # FAIL_COLLAPSE cos_min=0.169 bodyv_min=0.058
    "cmu_105_105_42",                    # FAIL_COLLAPSE cos_min=0.227 bodyv_min=0.078
    "cmu_105_105_47",                    # FAIL_COLLAPSE cos_min=0.418 bodyv_min=0.145
    "cmu_105_105_50",                    # FAIL_COLLAPSE cos_min=-0.094 bodyv_min=-0.031
    "cmu_105_105_59",                    # FAIL_COLLAPSE cos_min=0.337 bodyv_min=0.12
    "cmu_105_105_60",                    # FAIL_COLLAPSE cos_min=0.046 bodyv_min=0.016
    "cmu_105_105_61",                    # FAIL_COLLAPSE cos_min=0.069 bodyv_min=0.024
    "cmu_105_105_62",                    # FAIL_COLLAPSE cos_min=0.16 bodyv_min=0.056
    "cmu_106_106_06",                    # FAIL_INVERTED cos_min=-0.39 bodyv_min=-0.138
    "cmu_106_106_15",                    # FAIL_COLLAPSE cos_min=0.443 bodyv_min=0.144
    "cmu_106_106_17",                    # FAIL_INVERTED cos_min=-0.228 bodyv_min=-0.079
    "cmu_106_106_19",                    # FAIL_INVERTED cos_min=-0.192 bodyv_min=-0.067
    "cmu_106_106_30",                    # FAIL_INVERTED cos_min=-0.243 bodyv_min=-0.085
    "cmu_107_107_08",                    # FAIL_COLLAPSE cos_min=-0.019 bodyv_min=-0.007
    "cmu_107_107_11",                    # FAIL_COLLAPSE cos_min=0.426 bodyv_min=0.14
    "cmu_108_108_21",                    # FAIL_COLLAPSE cos_min=0.373 bodyv_min=0.124
    "cmu_108_108_23",                    # FAIL_COLLAPSE cos_min=0.155 bodyv_min=0.052
    "cmu_108_108_24",                    # FAIL_COLLAPSE cos_min=0.194 bodyv_min=0.064
    "cmu_108_108_25",                    # FAIL_INVERTED cos_min=-0.193 bodyv_min=-0.067
    "cmu_108_108_26",                    # FAIL_COLLAPSE cos_min=-0.067 bodyv_min=-0.024
    "cmu_111_111_02",                    # FAIL_COLLAPSE cos_min=0.059 bodyv_min=0.02
    "cmu_111_111_03",                    # FAIL_COLLAPSE cos_min=0.002 bodyv_min=0.001
    "cmu_111_111_06",                    # FAIL_INVERTED cos_min=-0.973 bodyv_min=-0.344
    "cmu_111_111_07",                    # FAIL_INVERTED cos_min=-0.662 bodyv_min=-0.221
    "cmu_111_111_08",                    # FAIL_INVERTED cos_min=-0.43 bodyv_min=-0.15
    "cmu_111_111_09",                    # FAIL_COLLAPSE cos_min=0.32 bodyv_min=0.11
    "cmu_111_111_10",                    # FAIL_INVERTED cos_min=-0.137 bodyv_min=-0.049
    "cmu_49_49_08",                      # FAIL_INVERTED cos_min=-0.978 bodyv_min=-0.363
    "gBR_sBM_cAll_d04_mBR0_ch05",        # FAIL_INVERTED cos_min=-0.461 bodyv_min=-0.151
    "gBR_sBM_cAll_d04_mBR0_ch09",        # FAIL_COLLAPSE cos_min=0.367 bodyv_min=0.106
    "gBR_sBM_cAll_d05_mBR0_ch05",        # FAIL_COLLAPSE cos_min=0.258 bodyv_min=0.09
    "gBR_sBM_cAll_d05_mBR0_ch08",        # FAIL_INVERTED cos_min=-0.182 bodyv_min=-0.06
    "gBR_sBM_cAll_d05_mBR0_ch09",        # FAIL_COLLAPSE cos_min=0.402 bodyv_min=0.14
    "gBR_sBM_cAll_d05_mBR0_ch10",        # FAIL_INVERTED cos_min=-0.327 bodyv_min=-0.114
    "gBR_sFM_cAll_d04_mBR0_ch01",        # FAIL_INVERTED cos_min=-0.325 bodyv_min=-0.109
    "gJB_sFM_cAll_d07_mJB5_ch06",        # FAIL_COLLAPSE cos_min=0.305 bodyv_min=0.106
    "gJS_sFM_cAll_d01_mJS3_ch04",        # FAIL_COLLAPSE cos_min=0.12 bodyv_min=0.038
    "gJS_sFM_cAll_d02_mJS3_ch04",        # FAIL_COLLAPSE cos_min=-0.044 bodyv_min=-0.014
    "gKR_sBM_cAll_d29_mKR1_ch01",        # FAIL_COLLAPSE cos_min=0.561 bodyv_min=0.14
    "gKR_sFM_cAll_d28_mKR2_ch03",        # FAIL_COLLAPSE cos_min=0.335 bodyv_min=0.109
    "gKR_sFM_cAll_d30_mKR2_ch17",        # FAIL_COLLAPSE cos_min=0.335 bodyv_min=0.106
    "gWA_sFM_cAll_d26_mWA0_ch08",        # FAIL_COLLAPSE cos_min=0.486 bodyv_min=0.145
    "gWA_sFM_cAll_d27_mWA0_ch15",        # FAIL_COLLAPSE cos_min=0.198 bodyv_min=0.063
    "multistyle_reel03_s23",             # FAIL_COLLAPSE cos_min=-0.067 bodyv_min=-0.022
    "wave_hello_mx",                     # FAIL_COLLAPSE cos_min=0.0 bodyv_min=0.0
}
