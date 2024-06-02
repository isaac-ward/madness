from path import Path

def test_deviation_from_other_path(p1, p2):
    path1 = Path(p1)
    path2 = Path(p2)
    print(f"P1: {path1.path_metres}")
    print(f"P2: {path2.path_metres}")
    #deviation = path1.deviation_from_path(path2, verbose=False)
    deviation = path1.dtw_distance(path2)
    print(f"Deviation Score: {deviation}")

if __name__ == "__main__":
    path1 = [(0, 0), (1, 1), (2, 2)]
    path2 = [(0, 0), (1, 1), (2, 2)]
    # Deviation Score: 0
    test_deviation_from_other_path(path1, path2)

    path1 = [(0, 0), (1, 1), (2, 2)]
    path2 = [(0, 0), (1, 1), (2, 3)]
    # Deviation Score: Small positive value
    test_deviation_from_other_path(path1, path2)

    path1 = [(0, 0), (1, 1), (2, 2)]
    path2 = [(0, 0), (2, 2), (4, 4)]
    # Deviation Score: Larger positive value
    test_deviation_from_other_path(path1, path2)

    path1 = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    path2 = [(0, 0), (1, 1), (2, 2)]
    # Paths with perfect alignment but different lengths - score should be zero
    test_deviation_from_other_path(path1, path2)

    path1 = [(0, 0), (1, 1), (2, 2)]
    path2 = [(2, 2), (1, 1), (0, 0)]
    # Paths with perfect alignment but going backwards - score should be large
    test_deviation_from_other_path(path1, path2)
    
    path1 = [(0, 0), (1, 1), (2, 2)]
    path2 = [(0, 0), (-1, -1), (-2, -2)]
    # Paths going in completely opposite directions - score should be large
    test_deviation_from_other_path(path1, path2)

