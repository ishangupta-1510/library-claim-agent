import pytest

from library_claim.stages.room import measure_room, to_ft2


def test_rectangular_room():
    corners = [(0, 0, 0), (4.2, 0, 0), (4.2, 0.01, 3.1), (0, -0.01, 3.1)]
    room = measure_room(corners, ceiling_y_m=2.75)
    assert room.shape == "rectangle"
    assert room.length_m == pytest.approx(4.2, abs=0.01)
    assert room.width_m == pytest.approx(3.1, abs=0.01)
    assert room.floor_area_m2 == pytest.approx(13.02, abs=0.05)
    assert room.wall_area_m2 == pytest.approx(2 * (4.2 + 3.1) * 2.75, abs=0.1)


def test_l_shaped_room_area_is_exact_not_the_bounding_box():
    # 5x4 room with a 2x2 notch: 20 - 4 = 16 m2.
    corners = [(0, 0, 0), (5, 0, 0), (5, 0, 2), (3, 0, 2), (3, 0, 4), (0, 0, 4)]
    room = measure_room(corners, ceiling_y_m=2.6)
    assert room.floor_area_m2 == pytest.approx(16.0)
    assert room.shape.startswith("polygon (6 corners")
    assert room.wall_area_m2 == pytest.approx(18 * 2.6)  # perimeter is still 18 m


def test_out_of_order_taps_are_untangled():
    # Diagonal order (0,2,1,3) would self-intersect and halve the area.
    corners = [(0, 0, 0), (4, 0, 3), (4, 0, 0), (0, 0, 3)]
    assert measure_room(corners, ceiling_y_m=None).floor_area_m2 == pytest.approx(12.0)


def test_rotated_room_still_measures_true_sides():
    import math
    a = math.radians(30)
    rot = lambda x, z: (x * math.cos(a) - z * math.sin(a), 0, x * math.sin(a) + z * math.cos(a))
    room = measure_room([rot(0, 0), rot(5, 0), rot(5, 3), rot(0, 3)], ceiling_y_m=2.5)
    assert room.length_m == pytest.approx(5, abs=0.01) and room.width_m == pytest.approx(3, abs=0.01)


def test_no_ceiling_point_means_no_wall_area():
    room = measure_room([(0, 0, 0), (3, 0, 0), (3, 0, 3), (0, 0, 3)], ceiling_y_m=None)
    assert room.height_m is None and room.wall_area_m2 is None


def test_too_few_corners_is_an_error():
    with pytest.raises(ValueError):
        measure_room([(0, 0, 0), (1, 0, 0)], 2.5)


def test_ft2_conversion():
    assert to_ft2(10.0) == pytest.approx(107.6, abs=0.1)
    assert to_ft2(None) is None


def test_a_ceiling_tap_on_a_table_gives_no_height_rather_than_a_wrong_one():
    from library_claim.stages.room import measure_room

    corners = [(0, 0, 0), (4, 0, 0), (4, 0, 3), (0, 0, 3)]
    room = measure_room(corners, 0.75)
    assert room.height_m is None and room.wall_area_m2 is None and room.floor_area_m2 == 12.0
