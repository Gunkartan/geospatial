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
PROBABILITY_COLUMNS = [
    'water_probability',
    'water_probability_mean_5',
    'building_probability',
    'building_probability_mean_5'
]