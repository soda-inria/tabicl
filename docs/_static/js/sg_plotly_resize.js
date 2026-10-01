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
document.addEventListener("DOMContentLoaded", () => {
    window.dispatchEvent(new Event("resize"));
    const mq = window.matchMedia("(max-width: 576px)");
    mq.addEventListener("change", () => window.dispatchEvent(new Event("resize")));
});
