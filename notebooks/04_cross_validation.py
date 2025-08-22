import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    import sys

    # sys.path = ['/Users/tzuchi/Documents/Workspace/regrank/'] + sys.path
    sys.path.append("/Users/tzuchi/Documents/Workspace/regrank/")
    # sys.path.append("..")

    import numpy as np
    from collections import defaultdict, Counter
    import graph_tool.all as gt
    import matplotlib.pyplot as plt
    import regrank as rr
    return gt, np, rr


@app.cell
def _():
    from regrank.stats import PeerInstitution

    pde = PeerInstitution()
    # pde = PhDExchange()
    # g = pde.get_data(goi="c18basic")
    g = pde.get_data(goi="sector")
    # g = pde.get_data(goi="stabbr")
    vp_goi = g.vertex_properties["goi"]
    vp_instnm = g.vertex_properties["instnm"]
    return g, vp_goi


@app.cell
def _(g, vp_goi):
    goi_s = []
    for _ in g.vertices():
        goi_s.append(vp_goi[_])
        # print(vp_goi[v], vp_instnm[v])
    return


@app.cell
def _(g, gt, np, rr):
    # import numpy as np
    # import graph_tool.all as gt
    # from collections import Counter
    # Assuming 'regrank' is your package name and the class is in 'stats'
    from regrank.stats import CrossValidation
    # import regrank as rr

    # --- Assume these variables are pre-loaded in your environment ---
    # g: your full graph-tool Graph
    # vp_goi: a vertex property map for 'group of interest' tags
    # vp_instnm: a vertex property map for 'institution name' tags
    # rr: the regrank library
    # -----------------------------------------------------------------

    # 1. Initialize and run Cross-Validation
    cv = CrossValidation(g, n_folds=5, n_reps=10, n_subfolds=5, seed=42)
    cv.gen_all_train_validate_splits()

    # 2. Select a specific split for analysis
    fold_id = 3  # 0..4
    rep_id = 0  # 0..9
    subfold_id = 0  # 0..4

    main_train_view = gt.GraphView(cv.g, efilt=cv.main_cv_splits[fold_id])
    efilter = cv.sub_cv_splits[fold_id][rep_id][subfold_id]

    g_train = gt.GraphView(main_train_view, efilt=efilter)
    g_test = gt.GraphView(main_train_view, efilt=np.logical_not(efilter.a))

    # 3. Fit the models on the training graph
    print("Fitting models...")
    g_train.vp.goi = g.vp.goi
    ranking_null = rr.optimize.SpringRank(method="annotated").fit(
        g_train, alpha=1e-12, lambd=1e-15, goi="goi"
    )["primal"]
    ranking_alt = rr.optimize.SpringRank(method="annotated").fit(
        g_train, alpha=1e-12, lambd=10, goi="goi"
    )["primal"]
    print("Fitting complete.")

    # 4. Evaluate scores on the test graph
    g_test.vp.goi = g.vp.goi
    g_test.vp.instnm = g.vp.instnm

    if "weights" in g.ep:
        print("Found 'weights' property map in the graph. Using it for scoring.")
        g_test.ep["weights"] = g.ep["weights"]
    else:
        print("\nWarning: Edge property 'weights' not found in the graph.")
        print(
            "Scoring will proceed by assuming a default weight of 1.0 for all edges.\n"
        )

    # --- CORRECTED SECTION ---
    # The previous line `set(g.vp.goi.a)` failed because .a is None for string properties.
    # The correct way is to iterate over the vertices and access the property map values.
    if "goi" not in g.vp:
        raise ValueError(
            "Vertex property 'goi' not found in the graph. Cannot proceed with analysis."
        )

    all_tags = set(g.vp.goi[v] for v in g.vertices())
    # --- END CORRECTION ---

    counter_better = 0
    counter_worse = 0
    total_score_null = 0
    total_score_alt = 0

    print("\n--- Comparison of Scores per Group ---")
    for tag in all_tags:
        score_null, (n_nodes, n_edges) = cv._compute_score_per_tag(
            g_test, ranking_null, tag
        )
        score_alt, _ = cv._compute_score_per_tag(g_test, ranking_alt, tag)

        if n_edges > 0:
            norm_score_null = score_null / n_edges
            norm_score_alt = score_alt / n_edges

            total_score_null += norm_score_null
            total_score_alt += norm_score_alt

            score_diff = norm_score_alt - norm_score_null

            if score_diff > 0:
                counter_better += n_nodes
                print(
                    f"Reg is BETTER for goi={tag}: Δ_score={score_diff:.3e} (#nodes={n_nodes}, #edges={n_edges})"
                )
            else:
                counter_worse += n_nodes
                print(
                    f"Reg is WORSE for goi={tag}: Δ_score={score_diff:.3e} (#nodes={n_nodes}, #edges={n_edges})"
                )

    # 5. Print final summary
    print("\n--- Overall Results ---")
    if counter_better > counter_worse:
        print(
            f"Regularization is better for more nodes: {counter_better} > {counter_worse}"
        )
    else:
        print(
            f"No regularization is better for more nodes: {counter_worse} > {counter_better}"
        )

    if abs(total_score_null) > 1e-12:
        percent_improvement = (total_score_alt - total_score_null) / abs(
            total_score_null
        )
        comparison = "better" if total_score_alt > total_score_null else "worse"
        print(
            f"OVERALL, Regularization is {comparison}: {total_score_alt:.5f} vs {total_score_null:.5f} ({percent_improvement:+.2%})"
        )
    else:
        print(
            f"OVERALL, Score comparison: Reg={total_score_alt:.5f}, NoReg={total_score_null:.5f} (relative change cannot be computed)"
        )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
