import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    # import sys

    # sys.path.append("/Users/tzuchi/Documents/Workspace/regrank/")

    import regrank as rr
    from regrank.draw import plot_hist
    from regrank.models import SpringRank

    return SpringRank, plot_hist, rr


@app.cell
def _(rr):
    g = rr.datasets.us_air_traffic()
    # g = rr.datasets.at_migrations()
    # g = rr.datasets.parakeet()
    print(g)
    return (g,)


@app.cell
def _(g):
    g.list_properties()
    return


@app.cell
def _(SpringRank, g):
    model = SpringRank(method="annotated")
    result = model.fit(g, alpha=1, lambd=0.5, goi="state_abr")
    return model, result


@app.cell
def _(g, model, result):
    summary = model.compute_summary(g, "state_abr", primal_s=result["primal"])
    return (summary,)


@app.cell
def _(plot_hist, summary):
    plot_hist(summary)
    return


@app.cell
def _(rr, summary):
    rr.print_summary_table(summary, max_width=40)
    return


@app.cell
def _(SpringRank, g):
    # Assuming 'model' is an instance of your SpringRank class
    model = SpringRank(method="porder")

    # Run the fit method, providing the new hyperparameters 'k' and 'lambda_x'
    # 'data' can be your graph-tool object or a NumPy array
    # 'b_vec' is optional.
    results = model.fit(g, k=50, lambda_x=0.01)

    # Access the results
    pruned_A = results["primal_A"]
    ranks_x = results["primal_x"]

    print("Pruned Matrix A:")
    print(pruned_A)
    print("\nFinal Ranks x:")
    print(ranks_x)
    return (model,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
