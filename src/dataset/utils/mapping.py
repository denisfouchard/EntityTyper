def generate_mapping(file_path: str) -> tuple[dict[str, int], list[str]]:
    """
    Generates a mapping from a file containing tab-separated values.

    Args:
        file_path (str): The path to the file containing the mappings. Each line in the file should contain an index and a class name separated by a tab.

    Returns:
        tuple: A tuple containing:
            - dict[str, int]: A dictionary mapping class names to their corresponding indices.
            - list[str]: A list of class names in the order they appear in the file.
    """
    class2idx = {}
    with open(file=file_path) as f:
        for line in f:
            idx, ent = line.strip().split(" ")
            class2idx[ent] = int(idx)
    idx2class = list(class2idx.keys())

    return class2idx, idx2class


def generate_hierarchical_mapping(
    file_path: str,
) -> tuple[dict[str, int], list[str]]:
    """
    Generates a mapping from a file containing tab-separated values.

    Args:
        file_path (str): The path to the file containing the mappings. Each line in the file should contain an index and a class name separated by a tab.

    Returns:
        tuple: A tuple containing:
            - dict[str, int]: A dictionary mapping class names to their corresponding indices.
            - list[str]: A list of class names in the order they appear in the file.
    """
    class2idx = {}
    idx2class = []
    with open(file=file_path) as f:
        types = f.readlines()

    # Match lines of patern idx1 idx2 idx3 idxn class_name
    idxs_type = []
    for line in types:
        line_strip = line.strip().split(" ")
        idxs = line_strip[:-1]
        idxs = [int(idx) for idx in idxs]
        type_name = line_strip[-1]
        idxs_type.append((idxs, type_name))

    for idxs, type_name in idxs_type:
        hierarchical_type_name = ""
        for idx in reversed(idxs):
            hierarchical_type_name += idxs_type[idx][-1] + "/"
        hierarchical_type_name = hierarchical_type_name[:-1]
        class2idx[hierarchical_type_name] = idxs[0]
        idx2class.append(hierarchical_type_name)

    return class2idx, idx2class
