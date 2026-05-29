def generate_search_space(trail, args):
    use_cluster = args.get('use_cluster', False) if args.get('use_cluster', False) else trail.suggest_categorical(
        "use_cluster", [True, False])
    cluster_num = args.get("cluster_num", None) if args.get("cluster_num", None) else trail.suggest_int("cluster_num",
                                                                                                        args.get(
                                                                                                            "cluster_num_min",
                                                                                                            10),
                                                                                                        args.get(
                                                                                                            "cluster_num_max",
                                                                                                            50)) if use_cluster else None
    use_threshold = args.get("use_threshold", False) if args.get("use_threshold", False) else trail.suggest_categorical(
        "use_threshold", [False, True])
    threshold = args.get("threshold", None) if args.get("threshold", None) else trail.suggest_float("threshold",
                                                                                                    args.get(
                                                                                                        "threshold_min",
                                                                                                        0.5),
                                                                                                    args.get(
                                                                                                        "threshold_max",
                                                                                                        1)) if use_threshold else None
    use_dynamic = args.get("use_dynamic", False) if args.get("use_dynamic", False) else trail.suggest_categorical(
        "use_dynamic", [False, True])

    sample_ratio = args.get("sample_ratio", 200) if args.get("sample_ratio", None) else trail.suggest_int(
        "sample_ratio",
        args.get(
            "sample_ratio_min",
            200),
        args.get(
            "sample_ratio_max",
            500)) if not use_dynamic else "dynamic"
    dynamic_ratio = args.get("dynamic_ratio", None) if args.get("dynamic_ratio", None) else trail.suggest_float(
        "dynamic_ratio",
        args.get("dynamic_ratio_min", 0.1),
        args.get("dynamic_ratio_max", 0.5)) if use_dynamic else None
    mixed_method = args.get("mixed_method", None) if args.get("mixed_method", None) else trail.suggest_categorical(
        "mixed_method", ["max", "min"])
    param = {
        "use_cluster": use_cluster,
        "cluster_num": cluster_num,
        "threshold": threshold,
        "retrieval_len": sample_ratio,
        "dynamic_ratio": dynamic_ratio,
        "mixed_method": mixed_method,
        "sub_feature_ratio": 1
    }
    return param
