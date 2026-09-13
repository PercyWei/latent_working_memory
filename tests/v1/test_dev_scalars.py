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
    definitions, posts = [], []
    section = {'index': 'dev-section', 'name': 'dev', 'chartIndex': []}
    charts = {}

    class Api:
        def run(self, path):
            return SimpleNamespace(run_id='cloud-id')

        def _get(self, path, params=None):
            if path.endswith('/sections'):
                data = [section] if posts else []
            else:
                data = charts[path.split('/')[-2]]
            return SimpleNamespace(ok=True, data=data)

        def _post(self, path, data):
            posts.append((path, data))
            if path.endswith('/chart/line'):
                index = str(len(charts))
                charts[index] = dict(data, index=index, type='LINE')
                section['chartIndex'].append(index)
            return SimpleNamespace(ok=True, data=section)

    monkeypatch.setattr('latent_working_memory.v1.dev_scalars.swanlab.Api', Api)
    run = SimpleNamespace(id='slug', url='https://swanlab.cn/@user/project/runs/slug',
                          define_metric=lambda key, **kw: definitions.append((key, kw)))
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 7  # one section and six native panels
    assert all(kw == {'hidden': True} for _, kw in definitions)
    configure_development_panels(run, ['semantic', 'random'], 'online')
    assert len(posts) == 7
