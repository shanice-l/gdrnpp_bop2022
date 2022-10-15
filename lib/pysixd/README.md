Author: TOMÁŠ HODAŇ

Source: https://github.com/thodan/sixd_toolkit


from the paper AAE:
2. In params/dataset_params.py** set **common_base_path** to the path of the SIXD datasets. (e.g. the directory containing t-less/t-less_v2/test_primesense)
3. replace eval_loc.py and eval_calc_erros.py
4. Download the visibility statistics for the different datasets from the SIXD challenge
"""
for each object in the image:
    bbox_obj: [265, 200, 42, 54]
    bbox_visib: [265, 200, 41, 53]
    px_count_all: 1466
    px_count_valid: 1466
    px_count_visib: 1449
    visib_fract: 0.98840382
"""
   homepage:
   e.g. T-LESS: http://ptak.felk.cvut.cz/6DB/public/datasets/t-less/
   folder structure should be like t-less/t-less_v2/test_primesense_gt_stats/
5. use ae_eval to evaluate on any sixd dataset (T-Less, Linemod, etc.)
