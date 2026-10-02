// There is an interaction between plotly and the pydata-sphinx-theme
// (secondary sidebar) that causes plotly figures to render cropped until a
// resize event is fired. Plotly figures are responsive, so this simply
// dispatches a resize event once the DOM has finished loading so that plotly
// recomputes the figure width.
//
// Workaround from probabl-ai/skore#748 and scikit-learn#30778.
//
// Additionally, the compute-trade-offs figures embed both a desktop and a
// mobile layout (swapped via CSS @media). The hidden variant renders at zero
// size; when the viewport crosses the breakpoint the CSS swaps visibility, so
// we fire a resize event on matchMedia change to make plotly redraw the
// newly-visible figure at the correct size.
//
// Plotly's legend does not re-wrap on Plotly.Plots.resize (it only resizes
// the existing layout). A full re-render via Plotly.react re-wraps it, but
// firing it on every intermediate resize event during a fast drag leaves
// plotly in a bad state at some widths. So we debounce the re-render: it
// only fires once the drag has settled (~200ms after the last event).
document.addEventListener("DOMContentLoaded", () => {
    window.dispatchEvent(new Event("resize"));
    const mq = window.matchMedia("(max-width: 576px)");
    mq.addEventListener("change", () => window.dispatchEvent(new Event("resize")));

    let timer = null;
    window.addEventListener("resize", () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            if (typeof window.Plotly === "undefined") return;
            document.querySelectorAll("div.plotly-graph-div").forEach((gd) => {
                if (gd.clientWidth > 0 && gd.clientHeight > 0 && gd.data && gd.layout) {
                    window.Plotly.react(gd, gd.data, gd.layout);
                }
            });
        }, 200);
    });
});
