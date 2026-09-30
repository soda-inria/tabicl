// There is an interaction between plotly and the pydata-sphinx-theme
// (secondary sidebar) that causes plotly figures to render cropped until a
// resize event is fired. Plotly figures are responsive, so this simply
// dispatches a resize event once the DOM has finished loading so that plotly
// recomputes the figure width.
//
// Workaround from probabl-ai/skore#748 and scikit-learn#30778.
document.addEventListener("DOMContentLoaded", () => {
    window.dispatchEvent(new Event("resize"));
});
