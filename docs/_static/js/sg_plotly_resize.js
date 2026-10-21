// There is an interaction between plotly and the pydata-sphinx-theme
// (secondary sidebar) that causes plotly figures to render cropped until a
// resize event is fired. Plotly figures are responsive, so this simply
// dispatches a resize event once the DOM has finished loading so that plotly
// recomputes the figure width.
//
// Workaround from probabl-ai/skore#748 and scikit-learn#30778.
//
// Additionally, the compute-trade-offs figures embed four pre-rendered
// variants: desktop / mobile layouts (swapped via CSS @media) and light /
// dark colour sets (swapped via the theme's html[data-theme] attribute).
// Only one variant of each figure is displayed; the others are display:none
// and therefore laid out at zero size, so whenever the swap changes (viewport
// crossing the breakpoint, or the user switching the theme) the newly visible
// figure must be redrawn at the correct width.
//
// Plotly's legend does not re-wrap on Plotly.Plots.resize (it only resizes
// the existing layout). A full re-render via Plotly.react re-wraps it, but
// firing it on every intermediate resize event during a fast drag leaves
// plotly in a bad state at some widths. So we debounce the re-render: it
// only fires once the drag has settled (~200ms after the last event).
const REDRAW_DELAY_MS = 200;

const redrawVisibleFigures = () => {
    if (typeof window.Plotly === "undefined") return;
    document.querySelectorAll("div.plotly-graph-div").forEach((gd) => {
        if (gd.clientWidth > 0 && gd.clientHeight > 0 && gd.data && gd.layout) {
            window.Plotly.react(gd, gd.data, gd.layout);
        }
    });
};

let redrawTimer = null;
const scheduleRedraw = () => {
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(redrawVisibleFigures, REDRAW_DELAY_MS);
};

document.addEventListener("DOMContentLoaded", () => {
    window.dispatchEvent(new Event("resize"));
    const mq = window.matchMedia("(max-width: 576px)");
    mq.addEventListener("change", () => window.dispatchEvent(new Event("resize")));

    window.addEventListener("resize", scheduleRedraw);

    // pydata-sphinx-theme resolves the light/dark choice into the
    // data-theme attribute of <html> (data-mode keeps "auto" / "light" /
    // "dark") and fires no event when it does; watch the attribute instead.
    // The observer fires on every flip, including the initial one set by the
    // theme script, which is what we want: it forces a draw of the variant
    // that only becomes visible once the attribute is in place.
    new MutationObserver(scheduleRedraw).observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme", "data-mode"],
    });
});
