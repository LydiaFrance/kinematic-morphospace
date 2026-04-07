"""Filtering utilities for motion-capture frame metadata.

Provides composable boolean-mask filters for bird ID, perch distance,
obstacle presence, horizontal distance, and other frame-level attributes.
"""

import numpy as np

# ------- Filtering data -------

def filter_by(frame_info, **filters):
    """Build a combined boolean mask from one or more frame-level filters.

    Supported keyword filters:

    * **birdID** -- bird ID integer (1-5).
    * **hawkname** -- hawk name string (e.g. ``'Drogon'``).
    * **perchDist** -- perch distance as ``str``, ``int``, or ``list``.
    * **obstacle** -- obstacle presence (0 or 1).
    * **IMU** -- IMU presence (0 or 1).
    * **left** -- left-side indicator (0 or 1).
    * **year** -- recording year (e.g. 2017, 2020).
    * **naive** -- naive status (0 or 1).
    * **turn** -- turn direction (``'left'``, ``'right'``, ``'straight'``).
    * **horzdist** -- horizontal distance; single value, ``(min, max)``
      tuple, or keyword string.

    Args:
        frame_info: Dict or DataFrame of frame metadata keyed by column name.
        **filters: Keyword arguments corresponding to the filter names above.

    Returns:
        Boolean mask of shape ``(n_frames,)`` where True indicates frames
        that pass all specified filters.
    """
    filter_mask = np.ones(len(frame_info), dtype=bool)

    filter_map = {
        'birdID':    ('BirdID',        filter_by_bool),
        'hawkname':  ('BirdID',        filter_by_hawkname),
        'perchDist': ('PerchDistance', filter_by_perchDist),
        'obstacle':  ('Obstacle',      filter_by_bool),
        'IMU':       ('IMU',           filter_by_bool),
        'left':      ('Left',          filter_by_bool),
        'year':      ('Year',          filter_by_bool),
        'naive':     ('Naive',         filter_by_bool),
        'turn':      ('Turn',          filter_by_turn),
        'horzdist':  ('HorzDistance',  filter_by_horzdist)
    }

    for key, (data_key, func) in filter_map.items():
        filter_value = filters.get(key)
        if filter_value is not None:
            if key == 'left' and frame_info.get("Left") is None:
                msg = "Left filter is only available for unilateral datasets."
                raise ValueError(msg)
            filter_mask &= func(frame_info[data_key], filter_value)

    # Check for unrecognised filters
    unrecognised_filters = set(filters.keys()) - set(filter_map.keys())
    if unrecognised_filters:
        print(f"Warning: Ignoring unrecognised filters: {unrecognised_filters}")
        print(f"Valid filters are: {list(filter_map.keys())}")

    return filter_mask


# ....... Helper functions .......

def filter_by_bool(variable, bool_value):
    """Create a boolean mask by comparing ``variable`` to ``bool_value``.

    Args:
        variable: Array of values to compare.
        bool_value: Value to match against. If ``None``, returns an all-True
            mask (i.e. no filtering is applied).

    Returns:
        Boolean mask with the same length as ``variable``.
    """
    if bool_value is None:
        return np.ones(variable.shape, dtype=bool)
    return variable == bool_value


def get_hawkID(hawk_name):
    """Determine the hawk ID from the provided name or numerical identifier.

    Args:
        hawk_name: Hawk name string (e.g. ``'Drogon'``) or digit string
            (e.g. ``'1'``).

    Returns:
        Integer hawk ID (1-5), or ``None`` if no match is found.
    """
    if hawk_name.isdigit():
        return int(hawk_name)

    hawk_map = {
        "dr": 1,
        "rh": 2,
        "ru": 3,
        "to": 4,
        "ch": 5
    }

    for key, value in hawk_map.items():
        if key in hawk_name.lower():
            return value

    return None


def filter_by_hawkname(birdID_df, hawk):
    """Create a boolean mask selecting frames belonging to a specific hawk.

    Args:
        birdID_df: Series of integer bird IDs.
        hawk: Hawk name string or digit string. If ``None`` or unrecognised,
            returns an all-True mask.

    Returns:
        Boolean array of the same length as ``birdID_df``.
    """
    if hawk is None:
        return np.ones(len(birdID_df), dtype=bool)

    hawk_ID = get_hawkID(hawk)

    if hawk_ID is None:
        return np.ones(len(birdID_df), dtype=bool)

    return birdID_df == hawk_ID


def filter_by_perchDist(perchDist_df, perchDist):
    """Filter frames by perch distance.

    Args:
        perchDist_df: Series containing perch distance values.
        perchDist: Distance specification. Accepts a string (e.g. ``'12m'``),
            an integer, or a list of either. If ``None``, returns an all-True
            mask.

    Returns:
        Boolean array where True indicates matching perch distances.
    """
    if perchDist is None:
        return np.ones(len(perchDist_df), dtype=bool)

    if not isinstance(perchDist, list):
        perchDist = [perchDist]

    normalized_distances = []
    for dist in perchDist:
        if isinstance(dist, str):
            normalized_distances.append(int(''.join(filter(str.isdigit, dist))))
        elif isinstance(dist, (int, float)):
            normalized_distances.append(int(dist))
        else:
            msg = (
                "Each perch distance must be either an integer or "
                "a string containing digits."
            )
            raise ValueError(msg)

    return perchDist_df.isin(normalized_distances)



def filter_by_turn(turn_df, turn):
    """Create a boolean mask selecting frames by turn direction.

    Args:
        turn_df: Series of turn-direction strings.
        turn: Turn direction to match (``'left'``, ``'right'``, or
            ``'straight'``). Case-insensitive. If ``None``, returns an
            all-True mask.

    Returns:
        Boolean array of the same length as ``turn_df``.
    """
    if turn is None:
        return np.ones(len(turn_df), dtype=bool)

    return turn_df.str.contains(turn, case=False)



def filter_by_horzdist(horzdist_df, horzdist_limit):
    """Filter frames by horizontal distance from the perch.

    All distances in the dataset are negative (perch is at ``x = 0``).

    Args:
        horzdist_df: Series of horizontal distance values.
        horzdist_limit: Filter specification. Accepts:

            * **float** -- keeps frames further than this value
              (e.g. ``4.5`` or ``-4.5`` both mean ``x < -4.5``).
            * **tuple** -- ``(start, end)`` keeps frames between the two
              values (e.g. ``(4.5, 2)`` means ``-4.5 <= x < -2``).
            * **str** -- keyword shortcut: ``'first_half'``,
              ``'second_half'``, ``'landing'``, ``'takeoff'``, or
              ``'in-flight'``.

    Returns:
        Boolean mask for filtering.

    Raises:
        ValueError: If an unrecognised string keyword or invalid type is given.
    """
    if horzdist_limit is None:
        return np.ones(len(horzdist_df), dtype=bool)

    # Handle string keywords for common filters
    if isinstance(horzdist_limit, str):
        keywords = {
            'first_half':  -4.5,
            'second_half': (-4.5, 0),
            'landing':     (-2, 0),
            'takeoff':     -6,
            'in-flight':   -0.7
        }
        if horzdist_limit.lower() not in keywords:
            msg = f"Unknown keyword. Valid options are: {list(keywords.keys())}"
            raise ValueError(msg)
        horzdist_limit = keywords[horzdist_limit.lower()]

    # Handle single value (make sure it's negative)
    if isinstance(horzdist_limit, (int, float)):
        limit = -abs(horzdist_limit)
        return horzdist_df < limit

    # Handle range tuple (make sure values are negative and in right order)
    if isinstance(horzdist_limit, tuple) and len(horzdist_limit) == 2:
        start, end = horzdist_limit
        start, end = -abs(start), -abs(end)
        start, end = min(start, end), max(start, end)
        return (horzdist_df >= start) & (horzdist_df < end)

    msg = (
        "horzdist_limit must be:\n"
        "- a number (e.g., 4.5 for x < -4.5)\n"
        "- a tuple of (start, end)\n"
        "- or one of: 'first_half', 'second_half', 'landing', 'takeoff', 'midflight'"
    )
    raise ValueError(
        msg
    )
