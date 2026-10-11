import numpy as np
import pytest
from calibrate_table_local import manual_ball, order_ball_candidates, build_parser, fit_options


def test_physical_long_dimension_maps_to_selected_image_axis():
    argv = ['--table-length-cm', '223.52', '--table-width-cm', '111.76',
            '--ball-diameter-cm', '5.715', '--pocket-width-cm', '11.5']
    original = fit_options(build_parser().parse_args(argv))
    rotated = fit_options(build_parser().parse_args(argv + ['--table-long-axis', 'top-to-bottom']))
    assert (original['width_cm'], original['height_cm']) == (223.52, 111.76)
    assert (rotated['width_cm'], rotated['height_cm']) == (111.76, 223.52)
    assert rotated['ball_radius_cm'] == original['ball_radius_cm']


def test_four_reference_centers_and_measured_radii_survive_selection():
    balls = []
    for x, y in ((100, 100), (500, 100), (500, 350), (100, 350)):
        balls.append(manual_ball((x, y), (x + 8, y), balls, (640, 480)))
    ordered = order_ball_candidates(balls)
    assert [b['name'] for b in ordered] == ['TL', 'TR', 'BR', 'BL']
    assert [b['radius_px'] for b in ordered] == [8]*4
    assert all(b['source'] == 'operator_center_and_edge' for b in ordered)
    assert not any(b['identity_confirmed'] for b in ordered)


@pytest.mark.parametrize('center,edge', [((100,100),(100,100)), ((100,100),(102,100)),
                                       ((2,100),(10,100)), ((100,100),(150,100)),
                                       ((float('nan'),100),(108,100))])
def test_invalid_silhouette_is_rejected(center, edge):
    with pytest.raises(ValueError):
        manual_ball(center, edge, [], (640,480))


def test_duplicate_ball_rejected_without_mutating_selection():
    first = manual_ball((100,100),(108,100),[],(640,480))
    selected = [first]
    with pytest.raises(ValueError, match='overlaps'):
        manual_ball((101,100),(109,100),selected,(640,480))
    assert selected == [first]


def test_four_collinear_selections_still_fail_geometry():
    balls = [manual_ball((x,100),(x+8,100),[],(640,480)) for x in (100,200,300,400)]
    with pytest.raises(ValueError, match='distinct'):
        order_ball_candidates(balls)
