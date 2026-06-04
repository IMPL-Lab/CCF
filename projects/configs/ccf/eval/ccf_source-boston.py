_base_ = [
    '../ccf_source.py'
]

data = dict(
    test=dict(
        ann_file='data/nuscenes/splits/nuscenes_infos_val_boston_target.pkl',
    )
)
