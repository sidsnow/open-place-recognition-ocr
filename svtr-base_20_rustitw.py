dictionary = dict(
    type='Dictionary',
    dict_file=
    'add dir of dict file',
    with_padding=True,
    with_unknown=True)
model = dict(
    type='SVTR',
    preprocessor=dict(
        type='STN',
        in_channels=3,
        resized_image_size=(
            32,
            64,
        ),
        output_image_size=(
            48,
            160,
        ),
        num_control_points=20,
        margins=[
            0.05,
            0.05,
        ]),
    encoder=dict(
        type='SVTREncoder',
        img_size=[
            48,
            160,
        ],
        in_channels=3,
        out_channels=256,
        embed_dims=[
            128,
            256,
            384,
        ],
        depth=[
            3,
            6,
            9,
        ],
        num_heads=[
            4,
            8,
            12,
        ],
        mixer_types=[
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
        ],
        window_size=[
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
        ],
        merging_types='Conv',
        prenorm=False,
        max_seq_len=40),
    decoder=dict(
        type='SVTRDecoder',
        in_channels=256,
        module_loss=dict(
            type='CTCModuleLoss', letter_case='lower', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dict(
            type='Dictionary',
            dict_file=
            'add dir of dict file',
            with_padding=True,
            with_unknown=True)),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[
            127.5,
        ], std=[
            127.5,
        ]))
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ]),
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ]),
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ]),
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            [
                dict(type='Resize', scale=(
                    256,
                    64,
                )),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    )),
            ],
        ]),
]
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
