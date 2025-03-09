import re

import unicodedata
from fuzzywuzzy import fuzz


def normalize_name(name: str):
    """

    Parameters
    ----------
    name: str, name we want to normalise

    Returns
    -------
    str - normalised name
    """

    # Normalise the name in a consistent format
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

    return name.lower().strip()


def fuzzy_match(name1, name2, threshold=85):
    """
    Returns whether two names have a similarity score above a certain threshold
    Parameters
    ----------
    name1: str, first name
    name2: str, second name
    threshold: int, threshold above which the names are considered to be the same person's

    Returns
    -------
    bool - whether the similarity score between the names is above the given threshold
    """

    # partial_ratio works better when there are initials, but we use token_sort_ratio since this function is only called
    # when org_match is False, so we do not want to match on initials just in case, we match on the name
    return fuzz.token_sort_ratio(name1, name2) >= threshold


def is_name_match(name_variations, person_name, org_match):
    """
    Given a person's name variations from OpenAlex, the name from the other source, and whether the organisation metadata
    matches, this returns true if we are confident that the names refer to the same person
    Parameters
    ----------
    name_variations: List[str], all known variations of the name according to OpenAlex
    person_name: str, name of person from new source
    org_match: bool, whether the organisations matched

    Returns
    -------
    bool - True if we are confident the names match
    """

    # Normalise the name given by GtR
    person_full_name = normalize_name(person_name)

    # Go through all name variations provided by OpenAlex
    for name in name_variations:

        # Normalise this variation
        norm_name = normalize_name(name)

        # Split into parts of names and filter out empty strings
        sub_var_names = [part for part in re.split("[ .-]+", norm_name) if part]
        sub_full_name = [part for part in re.split("[ .-]+", person_full_name) if part]

        # If the organisations match, we are going to check that any initial matches with the corresponding
        # sub-name in the other name, e.g., between John K Smith and John Kennedy Smith, the K and Kennedy will
        # match however John Calvin Smith will not match with John K Smith, we also match all the non-initial sub
        # names
        if org_match:
            if all(n[0] == g[0] for n, g in zip(sub_var_names, sub_full_name) if len(n) == 1 or len(g) == 1) and \
                    all(n[0] == g[0] for n, g in zip(sub_var_names, sub_full_name) if len(n) > 1 and len(g) > 1):
                return True

        # If the organisations do not match, we use fuzzywuzzy to see if the names match given a certain threshold
        else:
            if fuzzy_match(person_full_name, norm_name, 85):
                return True

    return False
