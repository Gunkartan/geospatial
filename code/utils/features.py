WATER_FEATURES = [
    'ndwi_mean',
    'mndwi_mean',
    'bsi_variance',
    'ndti_mean',
    'ndbsi_variance',
    'bi',
    'bi_mean',
    'bi_variance',
    'rbc_mean',
    'rsr_mean',
    'rsr_variance',
    'rgri_mean',
    'snr_mean',
    'snr_variance'
]
BUILDING_FEATURES = [
    'ndvi',
    'ndwi_mean',
    'ndwi_variance',
    'evi_mean',
    'ndbi_mean',
    'mndwi_mean',
    'mndwi_variance',
    'ndti',
    'ndti_mean',
    'ndbsi_mean',
    'ndbsi_variance',
    'bi',
    'bi_mean',
    'bi_variance',
    'rbc_mean',
    'grri_mean',
    'grri_variance',
    'rsr_mean',
    'rsr_variance',
    'rgri_mean',
    'snr_variance'
]
UPSTREAM_FEATURES = list(dict.fromkeys(WATER_FEATURES + BUILDING_FEATURES))
CROP_MONTHS = {
    'oct': '10-31',
    'nov': '11-30',
    'dec': '12-31'
}
CROP_INDICES = [
    'ndvi',
    'evi',
    'ndwi',
    'mtci',
    'swir_long'
]
TEXTURE_WINDOW = 5
CROP_SPECTRAL_FEATURES = [f'{index}_{month}' for month in CROP_MONTHS for index in CROP_INDICES]
# 5x5 (50 m) neighbourhood mean and variance of each index: in a 5-fold spatial
# ablation this beat 3x3 and the no-texture baseline in every fold and year.
CROP_TEXTURE_FEATURES = [
    f'{index}_{month}_{stat}_{TEXTURE_WINDOW}'
    for month in CROP_MONTHS
    for index in CROP_INDICES
    for stat in ('mean', 'var')
]
# Upstream land-use group models: probability column name -> (model file, features).
# Each is a binary model for one LU_ID_L1 group (1 built-up, 3 forest, 4 water,
# 5 miscellaneous); forest and misc target the non-crop land the crop model most
# often calls crop.
#
# Every upstream model sees its features for October, November and December
# (e.g. 'ndwi_mean_nov'). October alone is missing for 44% of 2020 pixels; adding
# the later months raised 2020 water AP from 0.33 to 0.61 and building AP from 0.21 to 0.74.
def monthly(features: list[str]) -> list[str]:
    return [f'{feature}_{month}' for month in CROP_MONTHS for feature in features]

UPSTREAM_MONTHLY_FEATURES = monthly(UPSTREAM_FEATURES)
UPSTREAM_MODELS = {
    'water': ('water', monthly(WATER_FEATURES)),
    'building': ('buildings', monthly(BUILDING_FEATURES)),
    'forest': ('forest', UPSTREAM_MONTHLY_FEATURES),
    'misc': ('misc', UPSTREAM_MONTHLY_FEATURES)
}
PROBABILITY_COLUMNS = [
    f'{name}_probability{suffix}'
    for name in UPSTREAM_MODELS
    for suffix in ('', '_mean_5')
]# Labels left out of crop training and scoring because their ground truth is ambiguous.
# 32767: every mixed land-use polygon (e.g. 'A403/A419' Durian/Mangosteen). Their joined
# LU_ID_L3 codes (e.g. 240312419) overflow the int16 label raster and clip to 32767;
# nearly all are mixtures that include a target crop.
# The rest are crop-like 'other agriculture' classes that are neither a target crop nor
# clearly non-crop: mixed orchard, mixed perennial and abandoned agricultural land.
MIXED_LAND_USE_LABEL = 32767
EXCLUDED_LABELS = [
    MIXED_LAND_USE_LABEL,
    2401,
    2301,
    2100,
    2200,
    2300,
    2400
]
