# 5.0: 47b529db-0593-4908-b3e7-4b24a32a0f70
# 6.0: c354b519-ccf0-4c1c-b3cc-645ed5751bb5
# 6.0(modular testing): 2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921
# 6.0(no telephoto camera and clock sensor): 4622f73a-250e-4633-9a3d-901ede6b9551
# 6.0(no clock sensor): f68151d1-604c-438e-a1a5-aa96d5581f4b
# 6.0(with signal sensor): 9272dd1a-793a-45b2-bff4-3a160b506d75
# 6.0(modular testing, birdview): b20c0d8a-f310-46b2-a639-6ce6be4f2b14

vehicle_models = {
    'apollo_6_with_signal': {
        'id': '9272dd1a-793a-45b2-bff4-3a160b506d75',
        'modules': [
            'Localization',
            'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            'Camera',
            # 'Traffic Light',
            'Control'
        ]
    },
    'apollo_6_modular': {
        'id': '2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921',
        'modules': [
            'Localization',
            # 'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            # 'Camera',
            # 'Traffic Light',
            'Control'
        ]
    },
    'apollo_6_modular_2gt': {
        'id': 'f0daed3e-4b1e-46ce-91ec-21149fa31758',
        'modules': [
            'Localization',
            # 'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            # 'Camera',
            # 'Traffic Light',
            'Control'
        ]
    },
    'apollo_6': {
        'id': 'c354b519-ccf0-4c1c-b3cc-645ed5751bb5',
        'modules': [
            'Localization',
            'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            'Camera',
            'Traffic Light',
            'Control'
        ]
    }
}


def get_modules_for_id(_id: str):
    for k in vehicle_models:
        if vehicle_models[k]['id'] == _id:
            return vehicle_models[k]['modules']
    raise Exception('unknown model_id: '+ _id)
