// Apply the stored appearance before first paint so the page never flashes the wrong theme.
(function () {
  var pref = null
  try { pref = localStorage.getItem('magneye-theme') } catch (e) {}
  var dark = pref === 'dark' || (pref !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
})()
