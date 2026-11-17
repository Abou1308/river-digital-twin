// Loaded automatically by Dash (assets folder). Marks when a mouse button or finger is down on a chart,
// so the live refresh can wait until a zoom/pan drag is finished (see the clientside callback in app.py).
(function () {
  function down(e) {
    if (e.target.closest && e.target.closest(".js-plotly-plot")) {
      window.rdtPointerDownSince = Date.now();
    }
  }
  function up() {
    window.rdtPointerDownSince = null;
  }
  document.addEventListener("mousedown", down, true);
  document.addEventListener("touchstart", down, true);
  document.addEventListener("mouseup", up, true);
  document.addEventListener("touchend", up, true);
  document.addEventListener("touchcancel", up, true);
  window.addEventListener("blur", up);
})();
