# Vendored frontend assets

The charting library [uPlot](https://github.com/leeoniya/uPlot) (MIT,
dependency-free) is vendored here and committed so the UI works fully offline with no
npm / Node / build step.

- `uplot.iife.min.js` — uPlot v1.6.31 (~50 KB)
- `uPlot.min.css`

To update: re-download the two files from
`https://cdn.jsdelivr.net/npm/uplot@<version>/dist/` and bump the version above.
