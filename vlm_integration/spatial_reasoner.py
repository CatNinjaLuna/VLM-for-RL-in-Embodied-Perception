# spatial_reasoner.py
'''
relations like left_of/right_of between detected objects.
Determine scores and select best matching object.
'''
def select_by_relation(targets, refs, relation):
    if not targets or not refs: 
        return None
    # use centroids
    if relation == "left_of":
        # choose target with max (ref.x - target.x) > 0
        best, best_score = None, -1e9
        for t in targets:
            tx, _ = t["centroid"]
            score = max((rx - tx) for r in refs for rx,_ in [r["centroid"]])
            if score > best_score:
                best, best_score = t, score
        return best
    if relation == "right_of":
        best, best_score = None, -1e9
        for t in targets:
            tx, _ = t["centroid"]
            score = max((tx - rx) for r in refs for rx,_ in [r["centroid"]])
            if score > best_score:
                best, best_score = t, score
        return best
    # simple placeholders for in_front_of/behind (needs camera model or y-axis convention)
    return None
