# nuScenes Data Split Tools

This directory contains the scripts used to reproduce the nuScenes source and
target splits required by the CCF configs.

The final artifacts are pkl files under `data/nuscenes/splits/`. The JSON files
under `tools/nuscenes_data_split/splits/` are intermediate tag annotations
derived from official nuScenes metadata.

## Split Definition

The current project only needs these domains:

- Source train: nuScenes train samples from Singapore, excluding rain and night.
- Source val: nuScenes val samples from Singapore, excluding rain and night.
- Target val Boston: nuScenes val samples from Boston.
- Target val night: nuScenes val samples whose scene description contains `night`.
- Target val rain: nuScenes val samples whose scene description contains `rain`.

These outputs match the `ann_file` entries referenced by the current configs:

- `data/nuscenes/splits/nuscenes_infos_train_singapore_norain_day_source.pkl`
- `data/nuscenes/splits/nuscenes_infos_val_singapore_norain_day_source.pkl`
- `data/nuscenes/splits/nuscenes_infos_val_boston_target.pkl`
- `data/nuscenes/splits/nuscenes_infos_val_night_target.pkl`
- `data/nuscenes/splits/nuscenes_infos_val_rain_target.pkl`

## Inputs

Expected nuScenes metadata files for tag generation:

- `data/nuscenes/v1.0-trainval/sample.json`
- `data/nuscenes/v1.0-trainval/scene.json`
- `data/nuscenes/v1.0-trainval/log.json`

Expected info pkl files for final filtering:

- `data/nuscenes/nuscenes_infos_train.pkl`
- `data/nuscenes/nuscenes_infos_val.pkl`

The info pkl may use either the `infos` key or the `data_list` key for sample
records. The filtering script preserves the original schema key.

## Step 1: Build Intermediate Tag Annotations

Skip this step if the trainval tag JSON files in
`tools/nuscenes_data_split/splits/` already exist and match the current official
nuScenes metadata.

```bash
python tools/nuscenes_data_split/build_city_annotations.py \
  --dataroot data/nuscenes \
  --output-json tools/nuscenes_data_split/splits/trainval_city.json

python tools/nuscenes_data_split/build_weather_annotations.py \
  --dataroot data/nuscenes \
  --output-json tools/nuscenes_data_split/splits/trainval_rain_night.json
```

These JSON files contain tags for all official `v1.0-trainval` samples. The
filtering step only looks up tokens present in the input pkl being filtered.

## Step 2: Build Source Splits

Source is Singapore day & no-rain. In practice this keeps Singapore samples and
excludes samples whose scene description contains `rain` or `night`.

Train source:

```bash
python tools/nuscenes_data_split/filter_nuscenes_infos.py \
  --info data/nuscenes/nuscenes_infos_train.pkl \
  --city-annotations tools/nuscenes_data_split/splits/trainval_city.json \
  --weather-annotations tools/nuscenes_data_split/splits/trainval_rain_night.json \
  --include-city Singapore \
  --exclude-weather rain night \
  --output data/nuscenes/splits/nuscenes_infos_train_singapore_norain_day_source.pkl
```

Val source:

```bash
python tools/nuscenes_data_split/filter_nuscenes_infos.py \
  --info data/nuscenes/nuscenes_infos_val.pkl \
  --city-annotations tools/nuscenes_data_split/splits/trainval_city.json \
  --weather-annotations tools/nuscenes_data_split/splits/trainval_rain_night.json \
  --include-city Singapore \
  --exclude-weather rain night \
  --output data/nuscenes/splits/nuscenes_infos_val_singapore_norain_day_source.pkl
```

## Step 3: Build Target Splits

All target splits are generated from the val info pkl.

Boston target:

```bash
python tools/nuscenes_data_split/filter_nuscenes_infos.py \
  --info data/nuscenes/nuscenes_infos_val.pkl \
  --city-annotations tools/nuscenes_data_split/splits/trainval_city.json \
  --exclude-city Singapore \
  --output data/nuscenes/splits/nuscenes_infos_val_boston_target.pkl
```

Night target:

```bash
python tools/nuscenes_data_split/filter_nuscenes_infos.py \
  --info data/nuscenes/nuscenes_infos_val.pkl \
  --weather-annotations tools/nuscenes_data_split/splits/trainval_rain_night.json \
  --include-weather night \
  --output data/nuscenes/splits/nuscenes_infos_val_night_target.pkl
```

Rain target:

```bash
python tools/nuscenes_data_split/filter_nuscenes_infos.py \
  --info data/nuscenes/nuscenes_infos_val.pkl \
  --weather-annotations tools/nuscenes_data_split/splits/trainval_rain_night.json \
  --include-weather rain \
  --output data/nuscenes/splits/nuscenes_infos_val_rain_target.pkl
```

## Notes

- Annotation scripts write intermediate tag JSON files with `--output-json`.
- `filter_nuscenes_infos.py --output` writes the final pkl files consumed by the
  training and evaluation configs.
- The Boston target command uses `--exclude-city Singapore` because nuScenes
  trainval locations in this setup are categorized as Singapore or Boston.
