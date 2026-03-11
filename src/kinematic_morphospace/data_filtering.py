"""Filtering utilities for motion-capture frame metadata.

Provides composable boolean-mask filters for bird ID, perch distance,
obstacle presence, horizontal distance, and other frame-level attributes.
"""

import numpy as np

# ------- Filtering data -------

def filter_by(frame_info, **filters):
    """Build a combined boolean mask from one or more frame-level filters.

    Supported keyword filters:

    * **birdID** -- bird ID integer (1–5).
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

    Parameters
    ----------
    frame_info : dict or pd.DataFrame
        Frame metadata keyed by column name.
    **filters
        Keyword arguments corresponding to the filter names above.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_frames,)``.
    """
    filter_mask = np.ones(len(frame_info), dtype=bool)
    
    filter_map = {
        'birdID':    ('BirdID',   filter_by_bool),
        'hawkname':  ('BirdID',   filter_by_hawkname),
        'perchDist': ('PerchDistance', filter_by_perchDist),
        'obstacle':  ('Obstacle', filter_by_bool),
        'IMU':       ('IMU',      filter_by_bool),
        'left':      ('Left',     filter_by_bool),
        'year':      ('Year',     filter_by_bool),
        'naive':     ('Naive',    filter_by_bool), 
        'turn':      ('Turn',     filter_by_turn),
        'horzdist':  ('HorzDistance', filter_by_horzdist)
    }

    # Apply filters based on the filter map. 
    # The dict key is the filter name, and the value is the column name and the filter method
    for key, (data_key, func) in filter_map.items():
        filter_value = filters.get(key)
        if filter_value is not None:
            if key == 'left' and frame_info.get("Left") is None:
                raise ValueError("Left filter is only available for unilateral datasets.")
            filter_mask &= func(frame_info[data_key], filter_value)

    # For debugging
    # print(f"Filtered by {filters}")

    # Check for unrecognised filters
    unrecognised_filters = set(filters.keys()) - set(filter_map.keys())
    if unrecognised_filters:
        print(f"Warning: Ignoring unrecognised filters: {unrecognised_filters}")
        print(f"Valid filters are: {list(filter_map.keys())}")


    return filter_mask

# ....... Helper functions .......
def filter_by_bool(variable, bool_value):
    """Create a boolean mask by comparing *variable* to *bool_value*.

    Parameters
    ----------
    variable : np.ndarray or pd.Series
        Array of values to compare.
    bool_value : object or None
        Value to match against. If ``None``, returns an all-``True`` mask
        (i.e. no filtering is applied).

    Returns
    -------
    np.ndarray
        Boolean mask with the same length as *variable*.
    """
    if bool_value is None:
        return np.ones(variable.shape, dtype=bool) # Aka "do not apply filter"
    return variable == bool_value

def get_hawkID(hawk_name):
    """Determine the hawk ID from the provided name or numerical identifier."""
    if hawk_name.isdigit():
        return int(hawk_name)

    # Mapping from hawk name initials to their IDs
    hawk_map = {
        "dr": 1,
        "rh": 2,
        "ru": 3,
        "to": 4,
        "ch": 5
    }

    # Return the matching ID based on the first two characters (lowercased)
    for key, value in hawk_map.items():
        if key in hawk_name.lower():
            return value

    # Return an empty string if no match is found (this should be handled carefully)
    return None

def filter_by_hawkname(birdID_df, hawk):
    """Filter frameID entries based on a hawk name or ID."""
    if hawk is None:
        # Return a boolean array of True values if no specific hawk is specified
        return np.ones(len(birdID_df), dtype=bool)
    
    hawk_ID = get_hawkID(hawk)

    if hawk_ID is None:
        # Return a boolean array of True values if no specific hawk is specified
        return np.ones(len(birdID_df), dtype=bool)
    
    # Return a boolean array where the frameID starts with the determined hawk_ID
    return birdID_df==hawk_ID
def filter_by_perchDist(perchDist_df, perchDist):
    """Filter frames by perch distance.

    Parameters
    ----------
    perchDist_df : pd.Series
        Series containing perch distance values.
    perchDist : str, int, or list
        Distance specification.  Accepts a string (e.g. ``'12m'``), an
        integer, or a list of either.

    Returns
    -------
    np.ndarray
        Boolean array where ``True`` indicates matching perch distances.
    """
    
    # If perchDist is None, return the full array bool mask
    if perchDist is None:
        return np.ones(len(perchDist_df), dtype=bool)

    # Convert input to list if it's not already
    if not isinstance(perchDist, list):
        perchDist = [perchDist]
    
    # Process each perch distance
    normalized_distances = []
    for dist in perchDist:
        if isinstance(dist, str):
            # Extract digits from the string and convert to integer
            normalized_distances.append(int(''.join(filter(str.isdigit, dist))))
        elif isinstance(dist, (int, float)):
            normalized_distances.append(int(dist))
        else:
            raise ValueError("Each perch distance must be either an integer or a string containing digits.")

    # Create mask for all specified distances
    is_selected = perchDist_df.isin(normalized_distances)

    return is_selected

def filter_by_turn(turn_df, turn):
    """
    Filter frameID entries based on a specified turn.
    Possible values are 'left', 'right', or 'straight'.
    """
    
    if turn is None:
        return np.ones(len(turn_df), dtype=bool)
    

    # Find the rows where the column "Turn" contains the string
    is_selected = turn_df.str.contains(turn, case=False)

    return is_selected

def filter_by_horzdist(horzdist_df, horzdist_limit):
    """Filter frames by horizontal distance from the perch.

    All distances in the dataset are negative (perch is at ``x = 0``).

    Parameters
    ----------
    horzdist_df : pd.Series
        Series of horizontal distance values.
    horzdist_limit : float, tuple, or str
        Filter specification:

        * **float** -- keeps frames further than this value
          (e.g. ``4.5`` or ``-4.5`` both mean ``x < -4.5``).
        * **tuple** -- ``(start, end)`` keeps frames between the two
          values (e.g. ``(4.5, 2)`` means ``-4.5 <= x < -2``).
        * **str** -- keyword shortcut: ``'first_half'``,
          ``'second_half'``, ``'landing'``, ``'takeoff'``, or
          ``'in-flight'``.

    Returns
    -------
    np.ndarray
        Boolean mask for filtering.
    """
    if horzdist_limit is None:
        return np.ones(len(horzdist_df), dtype=bool)
    
    # Handle string keywords for common filters
    if isinstance(horzdist_limit, str):
        keywords = {
            'first_half': -4.5,      # x < -4.5
            'second_half': (-4.5, 0), # -4.5 <= x < 0
            'landing': (-2, 0),      # -2 <= x < 0
            'takeoff': -6,           # x < -9
            'in-flight': -0.7    # x < -0.7
        }
        if horzdist_limit.lower() not in keywords:
            raise ValueError(f"Unknown keyword. Valid options are: {list(keywords.keys())}")
        horzdist_limit = keywords[horzdist_limit.lower()]
    
    # Handle single value (make sure it's negative)
    if isinstance(horzdist_limit, (int, float)):
        limit = -abs(horzdist_limit)  # Convert to negative if positive
        return horzdist_df < limit
    
    # Handle range tuple (make sure values are negative and in right order)
    elif isinstance(horzdist_limit, tuple) and len(horzdist_limit) == 2:
        start, end = horzdist_limit
        start, end = -abs(start), -abs(end)  # Convert to negative if positive
        # Swap if needed to ensure start is more negative than end
        start, end = min(start, end), max(start, end)
        return (horzdist_df >= start) & (horzdist_df < end)
    
    else:
        raise ValueError(
            "horzdist_limit must be:\n"
            "- a number (e.g., 4.5 for x < -4.5)\n"
            "- a tuple of (start, end)\n"
            "- or one of: 'first_half', 'second_half', 'landing', 'takeoff', 'midflight'"
        )
    