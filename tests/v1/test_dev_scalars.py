from types import SimpleNamespace

from latent_working_memory.v1.dev_scalars import (
    configure_development_panels, development_panels, development_scalars,
)


def test_sparse_generation_points_and_eight_curve_limit():
    reports = {'semantic': {'groups': {'all/ae/memory': {'nll': 2.0}}},
               'random': {'groups': {'all/ae/memory': {'nll': 3.0, 'correct_prefix_ratio': 0.0}}}}
    points = development_scalars(reports)
    assert points == {'dev/overview/ae/nll/semantic/memory': 2.0,
                      'dev/overview/ae/nll/random/memory': 3.0,
                      'dev/overview/ae/correct_prefix_ratio/random/memory': 0.0}
    panels = development_panels(list(reports))
    assert len(panels) == 6
    assert len(panels['dev/overview/ae/correct_prefix_ratio']['config']['yAxis']) == 8
    assert all(len(p['config']['yAxis']) <= 8 for p in panels.values())
    keys = [y['key'] for p in panels.values() for y in p['config']['yAxis']]
    assert len(keys) == len(set(keys)) == 42
    assert set(points) <= set(keys)
    assert all(p['config']['xAxis']['key'] == 'step' for p in panels.values())


def test_native_panels_created_once_before_scalar_upload(monkeypatch):
    posts = []
    section = {'index': 'dev-section', 'name': 'dev', 'chartIndex': []}
    charts = {}

    class Api:
        def run(self, path):
            return SimpleNamespace(run_id='cloud-id')

        def _get(self, path, params=None):
            data = ([section] if posts else []) if path.endswith('/sections') else charts[path.split('/')[-2]]
            return SimpleNamespace(ok=True, data=data)

        def _post(self, path, data):
            assert path.endswith('/columns')
            posts.append(data)
            for column in data:
                index = column['chartIndex']
                if index not in charts:
                    charts[index] = {'title': column['chartName'], 'index': index, 'type': 'LINE',
                                     'config': {'yAxis': []}}
                    section['chartIndex'].append(index)
                charts[index]['config']['yAxis'].append(column['key'])
            return SimpleNamespace(ok=True, data=[])

        def _put(self, path, data):
            index = path.split('/')[-3]
            charts[index].update(data)
            return SimpleNamespace(ok=True, data=None)

    monkeypatch.setattr('latent_working_memory.v1.dev_scalars.swanlab.Api', Api)
    run = SimpleNamespace(id='slug', url='https://swanlab.cn/@user/project/runs/slug')
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 1 and len(posts[0]) == 42
    assert len(charts) == 6
    assert all('hidden' not in column and column['sectionName'] == 'dev' for column in posts[0])
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 1


def test_delete_only_redundant_panels_and_empty_sections(monkeypatch):
    from latent_working_memory.v1.dev_scalars import remove_individual_dev_panels
    key = 'dev/overview/ae/nll/semantic/memory'
    deleted = []
    sections = [{'name': 'dev', 'index': 'main', 'chartIndex': ['grouped']},
                {'name': 'dev/overview/ae/nll/semantic', 'index': 'auto', 'chartIndex': ['single']}]
    hidden = [{'name': 'Hidden', 'type': 'HIDDEN', 'index': 'hidden', 'chartIndex': ['old', 'other']}]
    charts = {
        'grouped': {'title': 'dev/overview/ae/nll', 'type': 'LINE', 'config': {'yAxis': [key]}},
        'single': {'title': key, 'type': 'LINE', 'config': {'yAxis': [key]}},
        'old': {'title': key, 'type': 'LINE', 'config': {'yAxis': [key]}},
        'other': {'title': 'train/loss', 'type': 'LINE', 'config': {'yAxis': ['train/loss']}},
    }

    class Api:
        def run(self, path):
            return SimpleNamespace(run_id='cloud')
        def _get(self, path, params=None):
            data = hidden if path.endswith('/protected') else sections if path.endswith('/sections') else charts[path.split('/')[-2]]
            return SimpleNamespace(ok=True, data=data)
        def _delete(self, path):
            deleted.append(path)
            if path.endswith('/hard'):
                index = path.split('/')[-2]
                for section in sections + hidden:
                    section['chartIndex'] = [i for i in section['chartIndex'] if i != index]
            return SimpleNamespace(ok=True, data=None)

    monkeypatch.setattr('latent_working_memory.v1.dev_scalars.swanlab.Api', Api)
    run = SimpleNamespace(id='slug', url='https://swanlab.cn/@user/project/runs/slug')
    assert remove_individual_dev_panels(run, ['semantic']) == ['single', 'old']
    assert deleted == ['/experiment/cloud/chart/single/hard', '/experiment/cloud/chart/old/hard',
                       '/experiment/cloud/section/auto']
