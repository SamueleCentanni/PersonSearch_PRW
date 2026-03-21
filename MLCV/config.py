class Config:
    """
    Configuration for SeqNet Person Search on PRW.

    Base values come from SeqNet defaults.py.
    PRW-specific overrides (from prw.yaml): LUT_SIZE=482, CQ_SIZE=500, MAX_EPOCHS=18.
    """

    class MODEL:
        class RPN:
            PRE_NMS_TOPN_TRAIN = 12000
            PRE_NMS_TOPN_TEST = 6000
            POST_NMS_TOPN_TRAIN = 2000
            POST_NMS_TOPN_TEST = 300
            NMS_THRESH = 0.7

            POS_THRESH_TRAIN = 0.7
            NEG_THRESH_TRAIN = 0.3
            BATCH_SIZE_TRAIN = 256
            POS_FRAC_TRAIN = 0.5

        class ROI_HEAD:
            BN_NECK = True

            POS_THRESH_TRAIN = 0.5
            NEG_THRESH_TRAIN = 0.5
            BATCH_SIZE_TRAIN = 128
            POS_FRAC_TRAIN = 0.5

            SCORE_THRESH_TEST = 0.5
            NMS_THRESH_TEST = 0.4
            DETECTIONS_PER_IMAGE_TEST = 300

        class LOSS:
            LUT_SIZE = 482
            CQ_SIZE = 500
            OIM_MOMENTUM = 0.5
            OIM_SCALAR = 30.0

    class INPUT:
        DATASET = "PRW"
        MIN_SIZE = 900
        MAX_SIZE = 1500

        BATCH_SIZE_TRAIN = 5
        BATCH_SIZE_TEST = 1
        NUM_WORKERS_TRAIN = 5
        NUM_WORKERS_TEST = 1

    class SOLVER:
        MAX_EPOCHS = 18
        
        BASE_LR = 0.003
        CONVNEXT_BASE_LR = 5e-5 # ConvNeXt-specific optimizer settings, from the paper
        
        WARMUP_FACTOR = 1.0 / 1000
        LR_DECAY_MILESTONES = [16]
        GAMMA = 0.1

        CONVNEXT_WEIGHT_DECAY = 1e-8 # ConvNeXt-specific optimizer settings, from the paper
        WEIGHT_DECAY = 0.0005
        SGD_MOMENTUM = 0.9

        CLIP_GRADIENTS = 10.0

        LW_RPN_REG = 1.0
        LW_RPN_CLS = 1.0
        LW_PROPOSAL_REG = 10.0
        LW_PROPOSAL_CLS = 1.0
        LW_BOX_REG = 1.0
        LW_BOX_CLS = 1.0
        LW_BOX_REID = 1.0


cfg = Config()
