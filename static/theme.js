// Shared Dark Mode Manager for F.R.E.D.
// Applies persisted theme across all pages and syncs state between tabs.
(function () {
  function getSaved() {
    const v = localStorage.getItem('darkMode');
    return v === null ? true : v === 'true'; // Default to dark mode
  }

  function applyNow(isDark) {
    const enabled = !!isDark;
    const body = document.body;
    if (!body) return false;
    body.classList.toggle('dark-mode', enabled);
    const sw = document.getElementById('dark-mode-switch');
    if (sw) sw.checked = enabled;
    return true;
  }

  function apply(isDark) {
    if (!applyNow(isDark)) {
      document.addEventListener('DOMContentLoaded', function onReady() {
        document.removeEventListener('DOMContentLoaded', onReady);
        applyNow(isDark);
      });
    }
  }

  // Initial apply ASAP (or defer until DOM ready if needed)
  apply(getSaved());

  // Expose a small API for other scripts if needed
  window.FREDTheme = window.FREDTheme || {};
  window.FREDTheme.apply = apply;
  window.FREDTheme.setDarkMode = function (isDark) {
    const enabled = !!isDark;
    localStorage.setItem('darkMode', enabled);
    apply(enabled);
  };

  // Listen for cross-tab changes
  window.addEventListener('storage', function (e) {
    if (e.key === 'darkMode') {
      const val = e.newValue === 'true';
      if (!applyNow(val)) {
        document.addEventListener('DOMContentLoaded', () => applyNow(val), { once: true });
      }
    }
  });
})();
